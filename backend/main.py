from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import io
from pypdf import PdfReader
from langchain_core.messages import HumanMessage
import asyncio
from agent import Workflow 
from database import VectorDBClient, get_db
from document_ingestion import process

from contextlib import asynccontextmanager
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import ResponseError

def setup_redis_cache():
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        setup_redis_cache()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Redis cache. Is Docker running? Error: {e}")
    
    yield 

app = FastAPI(lifespan=lifespan)
file_name = []

@app.get("/")
def root():
  return "SystemOnline"

class User_question(BaseModel):
  question: str

@app.post("/question")
async def upload_question(ques : User_question):
  workflow = Workflow(query=ques.question)
  
  initial_state = {
    "topics": file_name,
    "messages": [HumanMessage(content=ques.question)],
    "in_cache": 0,
    "loop_number": 0,
    "break_loop": False
  }
  
  final_state = await workflow.agentic_workflow.ainvoke(initial_state)
  
  final_answer = final_state["messages"][-1].content
  
  return {"answer": final_answer}

@app.post("/file-upload")
async def upload_file(file : UploadFile):
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

  if file.filename not in file_name:
    file_name.append(file.filename)
  process(fulltext)

  return {"status": "success", "filename": file.filename, "chars_extracted": len(fulltext)}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

@app.post("/reset")
def reset_system():
    try:
        get_db().reset_collection()
        redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        redis_client.flushdb()
        
        global file_name
        file_name.clear()
        
        return {
            "status": "success", 
            "message": "System fully reset. ChromaDB, Redis, and Memory are clear."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")