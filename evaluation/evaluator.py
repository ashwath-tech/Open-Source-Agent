import json
import pandas as pd
import asyncio
import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset
import os
import requests
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
load_dotenv() 
API_BASE_URL = "http://localhost:8000"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
api_semaphore = asyncio.Semaphore(1)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
llm = llm_factory("gpt-4o-mini", client=openai_client)

client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

dataset_rows = []

with open('dataset.jsonl', 'r') as json_file:
    for line in json_file:
        try:
            dataset_rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line}")


async def fetch_answer(client, row):
    async with api_semaphore:
        payload = {"question": row["question"], "session_id": "1"}
        try:
            response = await client.post(f"{API_BASE_URL}/question", json=payload, timeout=60.0)
            response.raise_for_status()
            row["generated_answer"] = response.json().get("answer", "")
            row["context"] = response.json().get("context", "")
        except Exception as e:
            print(f"API Error for question '{row['question']}': {e}")
            row["generated_answer"] = "API_ERROR"
            row["context"] = []
            
        await asyncio.sleep(15)
    return row

async def run_generation_pipeline(rows):
    async with httpx.AsyncClient() as client:
        tasks = [fetch_answer(client, row) for row in rows]
        return await asyncio.gather(*tasks)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # --- PHASE 0: BACKEND INITIALIZATION ---
    print("Phase 0: Initializing backend state...")
    
    pdf_file_path = "NVIDIA-2025-Annual-Report.pdf" 
    
    try:
        print(" -> Resetting backend...")
        reset_res = requests.post(f"{API_BASE_URL}/reset")
        reset_res.raise_for_status()
        
        print(f" -> Uploading {pdf_file_path} (This may take a moment)...")
        with open(pdf_file_path, "rb") as f:
            files = {"file": (pdf_file_path.split("/")[-1], f, "application/pdf")}
            data = {"session_id": "1"}
            upload_res = requests.post(f"{API_BASE_URL}/file-upload", files=files, data=data)
            upload_res.raise_for_status()
            
        print("Backend reset and file uploaded successfully.\n")
        
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find the PDF at {pdf_file_path}")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL ERROR: Failed to communicate with FastAPI backend: {e}")
        exit(1)

    print(f"Phase 1: Generating answers for {len(dataset_rows)} questions via API...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    completed_rows = loop.run_until_complete(run_generation_pipeline(dataset_rows))
    loop.close()
    eval_data = {
        "question": [row["question"] for row in completed_rows],
        "answer": [row["generated_answer"] for row in completed_rows],
        "ground_truth": [row["answer"] for row in completed_rows],
        "contexts": [
          row.get("context") if isinstance(row.get("context"), list) else [row.get("context") or ""]
          for row in completed_rows
        ]
    }
    hf_dataset = Dataset.from_dict(eval_data)
    print(hf_dataset)
    print("Phase 2: Running LLM-as-a-judge evaluation...")
    for i, row in enumerate(eval_data["answer"]):
        if row == "" or row == "API_ERROR":
            print("Bad response at", i)

    for i, ref in enumerate(eval_data["ground_truth"]):
        if not ref:
            print("Bad reference at", i)
    metrics = [
        answer_correctness,
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    ]

    result = evaluate(dataset=hf_dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    print(result)
    results_df = result.to_pandas()
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation complete. Check evaluation_results.csv")