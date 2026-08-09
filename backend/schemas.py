from typing import TypedDict, Annotated, Optional, List, Dict
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator
import operator


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
  web_query: Optional[Annotated[str, Field(description="web query")]]

class User_question(BaseModel):
  question: str
  session_id: str = "default_session"
