from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import get_db

def process(all_text, size_of_chunk = 800, overlap_percent = 0.15):
  '''split using recursive text splitter'''

  overlap = int(size_of_chunk * overlap_percent)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=size_of_chunk, chunk_overlap=overlap)
  chunks = text_splitter.split_text(all_text)
  
  if not chunks:
    return
  
  ids = [f"{i}" for i in range(len(chunks))]
  get_db().create_and_store(ids, chunks)

  
  