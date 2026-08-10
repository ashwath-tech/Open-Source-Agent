from chromadb import EmbeddingFunction, Embeddings
import chromadb
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import register_embedding_function
import chromadb.errors
import asyncio
import redis
import pickle
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@db:5432/enterprise_rag"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

_db_instance = None

@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):
  def __init__(self):
    self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

  def __call__(self, chunks) -> Embeddings:
    return self.model.encode(chunks, normalize_embeddings=True).tolist()

  @staticmethod
  def name() -> str:
      return "my-ef"

  def get_config(self) -> Dict[str, Any]:
      return dict(model=self.model)

  @staticmethod
  def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
      return MyEmbeddingFunction(config['model'])

class VectorDBClient:
  def __init__(self):
    self.chroma_client = chromadb.PersistentClient(path="./db")
    self.collection = self.chroma_client.get_or_create_collection(
      name="RAG",
      embedding_function=MyEmbeddingFunction()
    )
    self.redis = redis.Redis()
    self._bm25 = None

  def _get_bm25_data(self):
      raw = self.redis.get("bm25_index")
      return pickle.loads(raw) if raw else {"corpus": [], "ids": []}

  def _save_bm25_data(self, data):
      self.redis.set("bm25_index", pickle.dumps(data))
      self._bm25 = None 

  def _get_bm25(self):
      if self._bm25 is None:
          data = self._get_bm25_data()
          if not data["corpus"]:
              return None, []
          self._bm25 = (BM25Okapi(data["corpus"]), data["ids"])
      return self._bm25

  async def get_similar(self, query, needed=10):
      dense_task = asyncio.to_thread(
          self.collection.query, query_texts=[query], n_results=needed
      )
      bm25_task = asyncio.to_thread(self._bm25_query, query, needed)

      dense_results, bm25_ids = await asyncio.gather(dense_task, bm25_task)

      dense_ids = dense_results["ids"][0] if dense_results["ids"] else []
      dense_docs = dense_results["documents"][0] if dense_results["documents"] else []

      fused_ids = self._rrf(dense_ids, bm25_ids)

      # fetch docs for bm25-only results
      id_to_doc = dict(zip(dense_ids, dense_docs))
      if bm25_ids:
          missing = [i for i in bm25_ids if i not in id_to_doc]
          if missing:
              bm25_docs = self.collection.get(ids=missing)
              for id_, doc in zip(bm25_docs["ids"], bm25_docs["documents"]):
                  id_to_doc[id_] = doc

      return [id_to_doc[i] for i in fused_ids if i in id_to_doc]

  def _bm25_query(self, query, needed):
      result = self._get_bm25()
      if result[0] is None:
          return []
      bm25, ids = result
      scores = bm25.get_scores(query.split())
      top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:needed]
      return [ids[i] for i in top_indices]

  def create_and_store(self, ids, allchunks, metadatas):
      # Vector DB (unchanged)
      self.collection.add(ids=ids, documents=allchunks, metadatas=metadatas)

      # BM25
      data = self._get_bm25_data()
      data["corpus"].extend([chunk.split() for chunk in allchunks])
      data["ids"].extend(ids)
      self._save_bm25_data(data)

  def reset_collection(self):
      try:
          self.chroma_client.delete_collection(name="RAG")
          self.collection = self.chroma_client.get_or_create_collection(
              name="RAG", embedding_function=MyEmbeddingFunction()
          )
          self.redis.delete("bm25_index")
          self._bm25 = None
          return True
      except Exception as e:
          print(f"Failed to reset: {e}")
          return False
  def _rrf(self, dense_ids, bm25_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def get_db():
    global _db_instance
    if _db_instance is None:
        print("Initializing Vector Database and Model...")
        _db_instance = VectorDBClient()
    return _db_instance