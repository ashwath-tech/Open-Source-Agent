from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
import io
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
from agent import Workflow 
from database import VectorDBClient, get_db
from document_ingestion import process
import os
import json
from contextlib import asynccontextmanager
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import ResponseError
from sentence_transformers import SentenceTransformer

def setup_redis_cache():
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    schema = (
        TextField("answer"),
        VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
    )
    definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
    
    try:
        client.ft("idx:cache").create_index(fields=schema, definition=definition)
        print("System: Redis Index 'idx:cache' initialized.")
    except ResponseError as e:
        if "Index already exists" not in str(e):
            raise e
ml_model = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['embedding_model'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        setup_redis_cache()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Redis cache. Is Docker running? Error: {e}")
    
    yield 
    ml_model.clear()

class ChatMemoryManager:
    def __init__(self):
        # We create a specific client for history with decode_responses=True 
        # so we can easily read/write JSON strings.
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.redis = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

    async def save_message(self, session_id: str, message):
        """Appends a message to the right side of a Redis List."""
        redis_key = f"chat_history:{session_id}"
        msg_data = {"type": message.type, "content": message.content}
        
        # We use sync redis calls here since standard redis-py isn't async by default 
        # (unless using redis.asyncio). Using asyncio.to_thread prevents blocking.
        await asyncio.to_thread(self.redis.rpush, redis_key, json.dumps(msg_data))
        await asyncio.to_thread(self.redis.expire, redis_key, 86400) # 24 hr TTL

    async def get_history(self, session_id: str, window_size: int = 10):
        """Retrieves the last N messages from Redis."""
        redis_key = f"chat_history:{session_id}"
        
        raw_messages = await asyncio.to_thread(self.redis.lrange, redis_key, -window_size, -1)
        
        history = []
        for raw in raw_messages:
            msg_data = json.loads(raw)
            if msg_data["type"] == "human":
                history.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                history.append(AIMessage(content=msg_data["content"]))
        return history
    async def add_topic(self, session_id: str, filename: str):
        """Adds a filename to the user's session using a Redis Set to prevent duplicates."""
        redis_key = f"session_topics:{session_id}"
        
        # SADD adds the item to a set. If it's already there, it does nothing.
        await asyncio.to_thread(self.redis.sadd, redis_key, filename)
        await asyncio.to_thread(self.redis.expire, redis_key, 86400) # 24 hr TTL

    async def get_topics(self, session_id: str) -> list:
        """Retrieves all filenames associated with this session."""
        redis_key = f"session_topics:{session_id}"
        
        # SMEMBERS returns all items in the set
        topics = await asyncio.to_thread(self.redis.smembers, redis_key)
        return list(topics)

app = FastAPI(lifespan=lifespan)
file_name = []

@app.get("/")
def root():
  return "SystemOnline"

class User_question(BaseModel):
  question: str
  session_id: str = "default_session"

@app.post("/question")
async def upload_question(ques : User_question):
  memory = ChatMemoryManager()

  history = await memory.get_history(session_id=ques.session_id)
  session_topics = await memory.get_topics(session_id=ques.session_id)

  current_message = HumanMessage(content=ques.question)
  full_messages = history + [current_message]

  workflow = Workflow(ml_model['embedding_model'])
  
  initial_state = {
    "topics": session_topics,
    "messages": full_messages,
    "in_cache": 0,
    "loop_number": 0,
    "break_loop": False
  }
  
  final_state = await workflow.agentic_workflow.ainvoke(initial_state)
  
  final_answer_object = final_state["messages"][-1]
  final_answer = final_answer_object.content

  await memory.save_message(ques.session_id, current_message)
  await memory.save_message(ques.session_id, final_answer_object)
  
  return {"answer": final_answer}

@app.post("/file-upload")
async def upload_file(file : UploadFile, session_id: str = Form(...)):
  file_bytes = await file.read()
  pdf_stream = io.BytesIO(file_bytes)
  
  reader = PdfReader(pdf_stream)
  
  extracted_text_chunks = []
  for page in reader.pages:
      text = page.extract_text()
      if text:
          extracted_text_chunks.append(text)
          
  fulltext = "\n".join(extracted_text_chunks)
  
  if not fulltext.strip():
      raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

  process(fulltext, file.filename)
  memory = ChatMemoryManager()
  await memory.add_topic(session_id, file.filename)

  return {"status": "success", "filename": file.filename, "chars_extracted": len(fulltext)}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

@app.post("/reset")
def reset_system():
    try:
        get_db().reset_collection()
        redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="cache:*", count=100)
            if keys:
                redis_client.delete(*keys)

            cursor_hist, keys_hist = redis_client.scan(cursor, match="chat_history:*", count=100)
            if keys_hist:
                redis_client.delete(*keys_hist)
            
            cursor_top, keys_top = redis_client.scan(cursor, match="session_topics:*", count=100)
            if keys_top: 
                redis_client.delete(*keys_top)
            if cursor == 0 and cursor_hist == 0:
                break
        
        
        return {
            "status": "success", 
            "message": "System fully reset. ChromaDB, Redis, and Memory are clear."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")