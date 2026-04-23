import numpy as np
from redis.asyncio import Redis
from redis.commands.search.query import Query
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from tavily import TavilyClient
from database import get_db
import cohere
import operator
import asyncio
import uuid
from langgraph.types import RetryPolicy
import math

load_dotenv()

class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], operator.add]
  in_cache: int
  topics: Annotated[List[str], operator.add]
  optimised_query: Optional[str]
  rag_query: Optional[str] = None
  initial_rag_query: Optional[str] = None
  web_query: Optional[str] = None
  web_context: Optional[Annotated[List[str], operator.add]] = None
  rag_context: Optional[List[str]] = None
  reranked_rag_context: Optional[List[str]] = None
  reranked_web_context: Optional[List[str]] = None
  loop_number: Optional[int] = 0
  break_loop: Optional[bool] = False
  critique: Optional[str]
  fallback_to_web: Optional[bool]

class AnalyzerChoice(BaseModel):
  stop_now : Annotated[bool, Field(description="Set to True ONLY if greeting or off-topic.")]
  stop_reply : Optional[Annotated[str, Field(description="Response to user if stop_now is True.")]] = None
  web_query : Optional[Annotated[str, Field(description="Query to search the web.")]] = None
  rag_query : Optional[Annotated[str, Field(description="Query to search the RAG database.")]] = None

  @model_validator(mode='after')
  def validate_routing_logic(self):
    if self.stop_now:
      if self.web_query or self.rag_query:
        raise ValueError("State Conflict: Cannot have search queries if stop_now is True.")
      if not self.stop_reply:
        raise ValueError("State Conflict: stop_reply is required if stop_now is True.")
    
    else:
      if not self.web_query and not self.rag_query:
        raise ValueError("State Conflict: Must provide at least web_query, rag_query, or both.")
    
    return self

class GraderFormat(BaseModel):
  loop: Annotated[bool, Field(description="True if loop again")]
  critique: Optional[Annotated[str, Field(description="critique of the current context")]]

class Workflow:
  def __init__(self, embedding_model):
    self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    self.embedding_model = embedding_model
    self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    self.redis_cache = Redis(host=self.REDIS_HOST, port=6379, decode_responses=False)
    self.client = AsyncOpenAI(
      api_key=self.OPENAI_API_KEY,
    )
    self.tavily_client = TavilyClient(api_key=self.TAVILY_API_KEY)
    self.co = cohere.AsyncClientV2(api_key=self.COHERE_API_KEY)
    self.retry_policy = RetryPolicy(
      max_attempts=3,
      initial_interval=2.0,
      backoff_factor=2.0,
    )

    rag_graph = StateGraph(AgentState)
    rag_graph.add_node("get_chunks", self.get_rag_chunks, retry=self.retry_policy)
    rag_graph.add_node("reranker", self.reranker, retry=self.retry_policy)
    rag_graph.add_node("critique", self.critique, retry=self.retry_policy)
    rag_graph.add_node("rewriter", self.rewriter, retry=self.retry_policy)

    rag_graph.add_edge(START, "get_chunks")
    rag_graph.add_edge("get_chunks", "reranker")
    
    rag_graph.add_conditional_edges(
        "reranker",
        self.relevance_condition,
        {"critique": "critique", "web": END} 
    )
    rag_graph.add_conditional_edges(
        "critique",
        self.rewrite_loop_condition,
        {"rewriter": "rewriter", "final": END} 
    )
    rag_graph.add_edge("rewriter", "get_chunks")
    
    rag_app = rag_graph.compile()

    self.agentic_workflow = StateGraph(AgentState)
    self.agentic_workflow.add_node("check_cache", self.get_from_redis_cache)
    self.agentic_workflow.add_node("analyzer", self.analyzer, retry=self.retry_policy)
    self.agentic_workflow.add_node("web_search", self.tavily_search, retry=self.retry_policy)
    
    self.agentic_workflow.add_node("rag_pipeline", rag_app) 
    
    self.agentic_workflow.add_node("final", self.draft_final, retry=self.retry_policy)
    self.agentic_workflow.add_node("store_cache", self.store_in_cache)
    self.agentic_workflow.add_node("web_reranker", self.web_reranker, retry=self.retry_policy)
    self.agentic_workflow.add_node("query_rewriter", self.query_rewriter, retry=self.retry_policy)

    self.agentic_workflow.add_edge(START, "query_rewriter")
    self.agentic_workflow.add_edge("query_rewriter", "check_cache")
    self.agentic_workflow.add_conditional_edges(
        "check_cache", 
        self.cache_condition, 
        ["analyzer", END]
    )
    
    self.agentic_workflow.add_conditional_edges(
        "analyzer",
        self.parallel_router,
        ["web_search", "rag_pipeline", END]
    )
    self.agentic_workflow.add_edge("web_search", "web_reranker")
    self.agentic_workflow.add_edge("web_reranker", "final")
    self.agentic_workflow.add_conditional_edges(
      "rag_pipeline",
      self.fallback_condition,
      ["final", "web_search"]
    )
    
    self.agentic_workflow.add_edge("final", "store_cache")
    self.agentic_workflow.add_edge("store_cache", END)
    self.agentic_workflow = self.agentic_workflow.compile()

  async def query_rewriter(self, state):
    messages = state.get("messages", [])
    if not messages:
        return {}

    original_query = messages[-1].content

    context_messages = messages[-6:-1] if len(messages) > 1 else []
    
    # Format history cleanly so the LLM knows who said what
    history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in context_messages])

    system_prompt = """You are an expert search query generator for a technical RAG (Retrieval-Augmented Generation) system.
      Your sole task is to convert a conversational user utterance into a highly targeted, standalone search query optimized for vector database retrieval.

      CORE DIRECTIVES:
      1. Strict Entity Resolution: Replace all pronouns (it, that, they, 'the first one') with the EXACT Proper Nouns or specific terminology they refer to from the chat history. 
      2. No Abstraction: DO NOT generalize entities.
      3. Contextual Independence: The final output MUST make perfect sense to a search engine that has no access to the chat history.
      4. Zero Generation: DO NOT answer the question. DO NOT add new technical concepts or keywords that the user didn't ask for.
      5. Pass-through: If the query does NOT need any rewriting, output the exact same query.

      OUTPUT FORMAT:
      Return ONLY the raw search string. No markdown, no quotes, no conversational prefixes.

      EXAMPLES:
      History:
      USER: How do I configure FUSE?
      AI: You can use the llfuse or fusepy bindings. Which are you using?
      Latest Query: the first one, specifically to bypass the kernel cache.
      Output: configure llfuse to bypass the kernel cache

      History:
      USER: What is the Indian Constitution used for?
      AI: It is the supreme law of India, laying down the framework of fundamental political codes...
      Latest Query: who wrote it?
      Output: who wrote the Indian Constitution?

      History:
      USER: What is the latency of Redis?
      AI: Redis achieves sub-millisecond latency.
      Latest Query: wow, that is really fast. thanks!
      Output: wow, that is really fast. thanks!
      """
    user_prompt = f"Chat History:\n{history_text}\n\nLatest Query: {original_query}"

    response = await self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    print(f"DEBUG: rewritten query: {response.choices[0].message.content}")

    return {"optimised_query": response.choices[0].message.content}

  async def store_in_cache(self, state):
    query = state["optimised_query"]
    answer = state["messages"][-1].content
    emb_array = await asyncio.to_thread(self.embedding_model.encode, query)
    emb = emb_array.astype(np.float32).tobytes()
    entry_id = uuid.uuid4().hex
    redis_key = f"cache:{entry_id}"

    await self.redis_cache.hset(
        name=redis_key,
        mapping={
            "vector": emb,
            "answer": answer,
            "original_query": query 
        }
    )
    await self.redis_cache.expire(redis_key, 86400)
    print(f"DEBUG: Successfully stored {redis_key} in cache.")
      
    return {}

  async def get_from_redis_cache(self, state):
    query = state['optimised_query']
    emb_array = await asyncio.to_thread(self.embedding_model.encode, query)
    query_vector = emb_array.astype(np.float32).tobytes()
    redis_query = (
      Query("*=>[KNN 1 @vector $query_vec AS vector_score]")
      .sort_by("vector_score")
      .return_fields("answer", "vector_score")
      .dialect(2)
    )
    results = await self.redis_cache.ft("idx:cache").search(
      redis_query,
      query_params={"query_vec": query_vector}
    )

    print(f"DEBUG: Cache search returned {results.total} results.")

    if results.docs:
      distance = float(results.docs[0].vector_score)
      
      if distance < 0.05: 
        cached_answer = results.docs[0].answer
        if isinstance(cached_answer, bytes):
          cached_answer = cached_answer.decode('utf-8')
        return {
          "messages": [AIMessage(content=cached_answer)],
          "in_cache": 1
        }
    return {
      "in_cache": 0
    }

  def cache_condition(self, state):
    if state["in_cache"] == 1:
      return END
    return "analyzer"

  def parallel_router(self,state):
    branches = []
    if state.get("web_query"):
      branches.append("web_search")
    if state.get("rag_query"):
      branches.append("rag_pipeline")
        
    if not branches:
      return [END] 
        
    return branches

  async def analyzer(self, state):
    user_query = state["optimised_query"]
    topics = state["topics"]
    print("TOPICS :", topics)

    prompt = f""" You are an expert routing analyzer. You evaluate user queries and determine the  required data sources.
      
      Names of PDFs in the database: {"\n".join(topics) if topics else "None"}
      
      1. STOP CONDITION: Set stop_now = True ONLY if the query is a generic greeting (e.g., "hello") or absolute gibberish. Provide a polite stop_reply. Do NOT stop for academic or factual questions.
      
      2. ROUTING LOGIC:
        - If the query relates to the PDF's in the database, generate a 'rag_query'.
        - If the query requires general internet knowledge, coding facts, or recent news or if the database is empty, generate a 'web_query'.
        - If it requires both, generate both.
      The database should always get preference.
    """
    response = await self.client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user","content": user_query},
      ],
      response_format=AnalyzerChoice,
    )

    parsed_response = response.choices[0].message.parsed
    print(f"DEBUG: Analyzing.....")
    if parsed_response.stop_now:
      return {
        "messages": [AIMessage(content=parsed_response.stop_reply)],
        "rag_query": None, 
        "web_query": None
      }
    return {
      "rag_query": parsed_response.rag_query,
      "initial_rag_query": parsed_response.rag_query,
      "web_query": parsed_response.web_query
    }

  async def tavily_search(self, state):

    response = await asyncio.to_thread(
      self.tavily_client.search,
      state["web_query"]
    )
    extracted_content = [result["content"] for result in response.get("results", [])]
    print(f"DEBUG: Searching with Tavily")
    return {
      "web_context": extracted_content
    }

  async def get_rag_chunks(self, state):
    query = state["rag_query"]

    retrieved_chunks = await asyncio.to_thread(
      get_db().get_similar,
      query=query,
      needed=10
    )

    print(f"DEBUG: Getting RAG chunks...")

    return {
      "rag_context": retrieved_chunks
    }

  async def reranker(self, state):
    docs = state["rag_context"]
    rag_query = state["rag_query"]

    if not docs:
      return {"reranked_rag_context": []}

    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in docs]
    response = await self.co.rerank(
      model="rerank-v4.0-pro",
      query=rag_query,
      documents=texts,
      top_n=3,
    )

    reranked_texts = [texts[res.index] for res in response.results]

    print(f"DEBUG: Reranking....")
    if response.results[0].relevance_score < 0.5:
      return {
        "reranked_rag_context": reranked_texts,
        "fallback_to_web": True,
        "web_query": state["initial_rag_query"]
      }
    return {
      "reranked_rag_context": reranked_texts,
      "fallback_to_web": False
    }

  async def web_reranker(self, state):
    docs = state["web_context"]
    web_query = state["web_query"]
    n = len(docs)


    if not docs:
      return {"reranked_web_context": []}

    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in docs]
    response = await self.co.rerank(
      model="rerank-v4.0-pro",
      query=web_query,
      documents=texts,
      top_n=max(1, int(math.log(n, 1.8))),
    )

    reranked_texts = [texts[res.index] for res in response.results]

    print(f"DEBUG: Tavily Reranking....")

    return {
      "reranked_web_context": reranked_texts
    }

  def relevance_condition(self, state):
    if state["fallback_to_web"]:
      return "web"
    return "critique"

  async def critique(self, state):
    extracted_content = state["reranked_rag_context"]
    query = state["rag_query"]
    initial = state["initial_rag_query"]
    loop_number = state["loop_number"]
    
    system_prompt = f"""
      Your role is to take a decision on whether the given context from a rag architecture is good and relateed to the given query.

      The prompt has already been rewritten {loop_number}

      1. loop: True if the context is not good and if changes to the query can get better results, otherwise False
      2. Critique: If you decide to loop back to change the query, provide a critique so that the query can be changed.

      Do not drift away from the user prompt.
      You should loop only if the changes will cause good improvement
    """
    if loop_number == 0:
      user_prompt = f"""
        Query: {query}
        Context: {"\n".join(extracted_content)}
      """
    else:
      user_prompt = f"""
        User Query: {initial}
        Rewritten Query: {query}
        Context: {"\n".join(extracted_content)}
      """

    response = await self.client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user","content": user_prompt},
      ],
      response_format=GraderFormat,
    )

    parsed_response = response.choices[0].message.parsed

    print(f"DEBUG: Giving critique....")

    if parsed_response.loop:
      print(f"CRTIQUE: {parsed_response.critique}")
      return {
        "break_loop": False,
        "critique": parsed_response.critique
      }
    return {
      "break_loop": True,
    }

  def rewrite_loop_condition(self, state):
    if state["break_loop"] or state["loop_number"] == 2:
      return "final"
    return "rewriter"

  async def rewriter(self, state):
    current_query = state["rag_query"]
    critique = state["critique"]
    system_prompt = """
      You are a proffessional prompt rewriter.
      You will be given an old prompt and a critique of that old prompt.
      Your job is to fix the old prompt by referencing the critique.
      Do not hallucinate. Rewrite only based on the critique.
    """
    user_prompt = f"""
      old prompt: {current_query}
      critique: {critique}

      OUTPUT ONLY THE REWRITTEN PROMPT DO NOT ADD ANYTHING ELSE
    """
    response = await self.client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user","content": user_prompt},
      ]
    )

    print(f"DEBUG: Rewriting..... {response.choices[0].message.content}")

    return {
      "rag_query": response.choices[0].message.content,
      "loop_number": state["loop_number"] + 1
    }

  def fallback_condition(self, state):
    if state["fallback_to_web"] and not state.get("web_query"):
      print("DUBUG: falling back to web search...")
      return "web_search"
    return "final"
  
  async def draft_final(self, state):
    user_question = state["optimised_query"]

    rag_text = []
    web_text = []
    if state.get("reranked_rag_context"):
      rag_text = state["reranked_rag_context"]
    if state.get("reranked_web_context"): 
      web_text = state["reranked_web_context"]

    system_prompt = """
      You have the most important job of compiling all the context and formatting it properly based on the users question to give a perfect response to the user. 
      Make sure to specify where you got the context from (Eg. According to the database.... or According to the internet....)

      DO NOT ADD NEW DATA. USE ONLY THE GIVEN CONTEXT
    """
    user_prompt = f"""
      question = {user_question}
      Database Context = {rag_text}
      Web Context = {web_text}
    """

    response = await self.client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user","content": user_prompt},
      ]
    )

    final_answer = response.choices[0].message.content

    print(f"DEBUG: Drafting final response...")

    return {"messages": [AIMessage(content=final_answer)]}
