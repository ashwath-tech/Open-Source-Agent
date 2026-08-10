from fastapi import FastAPI, UploadFile, HTTPException, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from utils import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from model import User
from database import SessionLocal, engine, Base
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
from schemas import User_question, TokenData, Token, UserCreate, UserResponse

Base.metadata.create_all(bind=engine)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

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
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.redis = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

    async def save_message(self, session_id: str, message):
        """Appends a message to the right side of a Redis List."""
        redis_key = f"chat_history:{session_id}"
        msg_data = {"type": message.type, "content": message.content}
        
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
        
        await asyncio.to_thread(self.redis.sadd, redis_key, filename)
        await asyncio.to_thread(self.redis.expire, redis_key, 86400) # 24 hr TTL

    async def get_topics(self, session_id: str) -> list:
        """Retrieves all filenames associated with this session."""
        redis_key = f"session_topics:{session_id}"
        
        topics = await asyncio.to_thread(self.redis.smembers, redis_key)
        return list(topics)

app = FastAPI(lifespan=lifespan)
file_name = []

@app.get("/")
def root():
  return "SystemOnline"

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordBearer = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/question")
async def upload_question(ques : User_question, current_user: User = Depends(get_current_user)):
  
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
  context_used = final_state.get("reranked_rag_context")

  await memory.save_message(ques.session_id, current_message)
  await memory.save_message(ques.session_id, final_answer_object)

  print("----------------------------------")
  print(type(context_used))
  print(context_used)
  print("----------------------------------")
  return {"answer": final_answer, "context": context_used}

@app.post("/file-upload")
async def upload_file(file : UploadFile, session_id: str = Form(...), current_user: User = Depends(get_current_user)):
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
def reset_system(current_user: User = Depends(get_current_user)):
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