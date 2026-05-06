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
  rag_query : Optional[Annotated[str, Field(description="Query to search the RAG database.")]] = None

class GraderFormat(BaseModel):
  loop: Annotated[bool, Field(description="True if loop again")]
  critique: Optional[Annotated[str, Field(description="critique of the current context")]]
  web_query: Optional[Annotated[str, Field(description="web query")] ]

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
    rag_graph.add_edge("reranker", "critique")
    
    rag_graph.add_conditional_edges(
        "critique",
        self.relevance_condition,
        {"rewriter": "rewriter", "web": END} 
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

    self.agentic_workflow.add_edge(START, "analyzer")
    
    self.agentic_workflow.add_conditional_edges(
        "analyzer",
        self.parallel_router,
        ["check_cache", END]
    )

    self.agentic_workflow.add_conditional_edges(
        "check_cache", 
        self.cache_condition, 
        ["rag_pipeline", END]
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

  async def store_in_cache(self, state):
    query = state["initial_rag_query"]
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
    query = state['rag_query']
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
    return "rag_pipeline"

  def parallel_router(self,state):
    if state.get("rag_query"):
      return "check_cache"
    return END

  async def analyzer(self, state):
    messages = state.get("messages", [])
    if not messages:
        return {}

    user_query = messages[-1].content

    context_messages = messages[-6:-1] if len(messages) > 1 else []
    history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in context_messages])

    topics = state["topics"]
    print("TOPICS :", topics)

    prompt = f"""You are a strict routing engine for a RAG system.

      Your job is to output EXACTLY ONE of the following states:

      1. STOP
      2. SEARCH

      ---

      RULES:

      STOP:
      - Only if the query is a greeting, small talk, or meaningless
      - Must include a short natural reply to the user

      SEARCH:
      - Must produce a single, self-contained search query
      - The query must be understandable WITHOUT chat history
      - Only resolve ambiguity using history
      - Do NOT expand, explain, or improve the query
      - Do NOT add new information

      ---

      STRICT CONSTRAINTS:
      - NEVER output both STOP and SEARCH
      - NEVER leave required fields empty
      - NEVER explain your reasoning
      - Output must strictly follow the schema

      ---

      EXAMPLES:

      [Good]
      History: "What is Redis?"
      Query: "who made it?"
      → SEARCH: "who created Redis?"

      [Bad]
      → "Redis was created by..." ❌ (this is answering, not routing)

      [Bad]
      → "Tell me more about Redis creators" ❌ (added info)

      ---

      Now process the input.
    """

    user_query = f"Chat History:\n{history_text}\n\nLatest Query: {user_query}"

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
        "rag_query": None
      }
    return {
      "rag_query": parsed_response.rag_query,
      "initial_rag_query": parsed_response.rag_query,
    }

  async def tavily_search(self, state):

    print(f"DEBUG: web query - {state['web_query']}")

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
    print(f"DEBUG: rag query - {query}")
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
      top_n=5,
    )

    reranked_texts = [texts[res.index] for res in response.results]

    return {
      "reranked_rag_context": reranked_texts,
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
    if state["break_loop"] or state.get("loop_number", 0) >= 3:
      return "web"
    return "rewriter"

  async def critique(self, state):
    extracted_content = state["reranked_rag_context"]
    query = state["rag_query"]
    initial = state["initial_rag_query"]
    loop_number = state["loop_number"]
    
    system_prompt = f"""You are a strict retrieval evaluator.

      Your ONLY goal:
      Decide whether the retrieved context is sufficient to answer the query EXACTLY.

      ---

      DEFINITION OF SUFFICIENT:
      The context must contain:
      - All required entities
      - All required relationships
      - Any numbers, dates, or attributes explicitly asked

      If ANY required detail is missing → NOT sufficient.

      ---

      YOU MUST CHOOSE ONE:

      1. LOOP
      - Only if a BETTER query can retrieve the missing information from the database
      - Provide a precise critique explaining:
        - what is missing
        - what exact keywords should be used

      2. WEB
      - If:
        - context is irrelevant
        - answer requires real-time or external knowledge
        - or after multiple failed attempts

      3. SUCCESS
      - If context fully answers the query
      - No critique, no web query

      ---

      STRICT RULES:

      - Do NOT say "partially relevant"
      - Do NOT be vague
      - Do NOT suggest broad improvements
      - Critique must directly map to a better search query

      ---

      EXAMPLE:

      Query: "Age of Elon Musk"
      Context: "Elon Musk is CEO of Tesla"
      → LOOP
      Critique: "Missing age. Query should include 'Elon Musk age'"

      ---

      Now evaluate.
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
    elif parsed_response.web_query:
      return {
        "fallback_to_web": True,
        "break_loop": True,
        "web_query": parsed_response.web_query
      }
    return {
      "break_loop": True,
    }

  async def rewriter(self, state):
    current_query = state["rag_query"]
    critique = state["critique"]
    system_prompt = """
      You are a deterministic query rewriter.

      Input:
      - Original query
      - Critique

      Your job:
      Produce a corrected query that FIXES the critique.

      ---

      RULES:

      - Only modify parts mentioned in the critique
      - Do NOT rephrase the entire query
      - Do NOT add new concepts
      - Do NOT remove original intent
      - Output must be a single query string

      ---

      FAIL CONDITIONS (STRICTLY AVOID):

      - Adding unrelated terms
      - Making the query broader
      - Changing meaning

      ---

      EXAMPLE:

      Original: "prime minister of India"
      Critique: "Missing age"
      Output: "prime minister of India age"

      ---

      Output ONLY the rewritten query.
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
    if state.get("fallback_to_web"):
      print("DUBUG: falling back to web search...")
      return "web_search"
    return "final"
  
  async def draft_final(self, state):
    user_question = state["initial_rag_query"]

    rag_text = []
    web_text = []
    if state.get("reranked_rag_context"):
      rag_text = state["reranked_rag_context"]
    if state.get("reranked_web_context"): 
      web_text = state["reranked_web_context"]

    system_prompt = """
    You are a strict factual answer generator. Your only job is to answer questions using the provided context.

    ---

    STEP 1 — GROUND YOURSELF:
    Before answering, find and internally note the exact sentence(s) from the context that support your answer.
    If no such sentence exists, do not answer.

    ---

    STEP 2 — ANSWER RULES:
    - Answer ONLY using information explicitly stated in the context
    - Do NOT infer, connect dots, or generalize
    - Do NOT use any outside or prior knowledge
    - Do NOT add explanations not directly present in the context
    - If database context exists on this topic, ignore web context entirely
    - If context is insufficient → respond exactly with:
      "The available context is insufficient to answer this question."

    ---

    CONTEXT PRIORITY:
    1. Database context (highest priority — if present, discard web context on that topic)
    2. Web context (only if database context is absent)

    ---

    OUTPUT FORMAT:
    - State your answer clearly and directly
    - Cite your source:
      - "According to the database: ..."
      - "According to the internet: ..."
    - Do not add anything beyond what the cited source explicitly states

    ---

    REMEMBER:
    - No hallucination
    - No inference
    - No generalization
    - Silence is better than a wrong answer
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
