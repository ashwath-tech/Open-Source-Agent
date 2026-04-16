from chromadb import EmbeddingFunction, Embeddings
import chromadb
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import register_embedding_function
import chromadb.errors

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

  def get_similar(self, query, needed = 10):
    '''get "needed" similar chunks which will be reranked later'''
    results = self.collection.query(
      query_texts=[query],
      n_results=needed
    )
    if not results["documents"]:
      return []
    return results["documents"][0]
  
  def create_and_store(self, ids, allchunks):
    '''store in chromadb'''
    try:
      self.collection.add(
        ids = ids,
        documents = allchunks
      )
    except:
      print("Warning: Collection reference lost. Re-initializing...")
      self.collection = self.chroma_client.get_or_create_collection(
        name="RAG",
        embedding_function=MyEmbeddingFunction() 
      )
      self.collection.add(
        ids=ids,
        documents=allchunks
      )
  def reset_collection(self):
    """Deletes the existing RAG collection and creates a fresh one."""
    try:
      self.chroma_client.delete_collection(name="RAG")
      self.collection = self.chroma_client.get_or_create_collection(
          name="RAG",
          embedding_function=MyEmbeddingFunction()
      )
      return True
    except Exception as e:
      print(f"Failed to reset ChromaDB: {e}")
      return False

def get_db():
    global _db_instance
    if _db_instance is None:
        print("Initializing Vector Database and Model...")
        _db_instance = VectorDBClient()
    return _db_instance