# Agentic RAG Pipeline with LangGraph

An advanced, fully asynchronous AI agent that intelligently routes queries, performs semantic caching, retrieves internal documents, and searches the web. Built using **LangGraph**, **FastAPI**, **Redis Stack**, and multiple AI models (OpenAI, Cohere).

## System Architecture & Workflow

The system utilizes an agentic workflow with parallel routing and a dedicated RAG feedback loop. 

Below is the execution graph representing the LangGraph state machine:

```mermaid
flowchart TD
    %% Define Styling
    classDef loop stroke:#ff7675,stroke-width:3px,stroke-dasharray: 5 5;
    classDef decision fill:#6c5ce7,color:#fff,stroke:#a29bfe;
    classDef cache fill:#0984e3,color:#fff,stroke:#74b9ff;
    classDef endpoint fill:#d63031,color:#fff,stroke:#ff7675;

    %% Main Nodes
    Start((START)):::endpoint
    End((END)):::endpoint
    
    CheckCache{check_cache}:::cache
    Analyzer{analyzer}:::decision
    WebSearch[web_search]
    Final[draft_final]
    StoreCache[store_cache]:::cache

    %% Main Graph Routing
    Start --> CheckCache
    CheckCache -- "Cache Hit (in_cache == 1)" --> End
    CheckCache -- "Cache Miss" --> Analyzer
    
    Analyzer -- "stop_now == True" --> End
    Analyzer -- "Has web_query" --> WebSearch

    %% RAG Subgraph Definition
    subgraph RAG_Pipeline ["rag_graph (Subgraph)"]
        direction TB
        GetChunks[get_chunks]
        Reranker[reranker]
        Critique{critique}:::decision
        Rewriter[rewriter]:::loop

        GetChunks --> Reranker
        Reranker --> Critique
        
        %% The Feedback Loop
        Critique -- "break_loop == False\n(Poor Context)" --> Rewriter
        Rewriter -- "Rewrite Query\nloop_number += 1" --> GetChunks
    end

    %% Subgraph Connections
    Analyzer -- "Has rag_query" --> GetChunks
    Critique -- "break_loop == True\nOR loop_number == 2" --> Final
    WebSearch --> Final
    
    Final --> StoreCache
    StoreCache --> End
```

### The RAG Critique Loop Explained
Instead of blindly returning vector search results, the `rag_graph` subgraph enforces quality control:
1. **Retrieve & Rerank:** Fetches chunks from ChromaDB and passes them through the Cohere Reranker.
2. **Critique:** An LLM evaluates if the reranked context answers the query.
3. **Rewrite (The Loop):** If the context is poor, the `rewriter` node changes the search query based on the critique and loops the state back to `get_chunks`. It will break automatically after 2 loops to prevent infinite execution.

## Key Concepts & Learnings Applied

* **FastAPI Lifespan Events:** Managed application startup/shutdown gracefully. Used `@asynccontextmanager` to ensure the Redis index (`idx:cache`) initialized before accepting traffic.
* **Redis Semantic Cache:** Implemented a high-performance semantic cache using Redis Stack. Uses `SentenceTransformers` and Cosine Similarity to detect conceptually similar questions, bypassing LLM execution.
* **Tavily API:** Integrated an agentic search engine optimized for LLMs to fetch real-time web context.
* **Asynchronous Execution:** Migrated to a highly concurrent `async/await` architecture. Used `asyncio.to_thread` to offload CPU-bound embeddings to background threads, unblocking the FastAPI event loop.
* **Retry Policies:** Configured LangGraph `RetryPolicy` wrappers to handle transient network failures and HTTP 429 rate limits from external API providers.
* **LangGraph State Management:** Modeled complex agentic behaviors as directed graphs with custom nodes, `operator.add` reducers, and conditional edges to create parallel execution branches.

## Running the Application Locally

1. Clone the repository and navigate to the project directory.
2. Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   COHERE_API_KEY=your_cohere_key
   REDIS_HOST=redis-stack
   ```
3. Run the system using Docker Compose:
   ```bash
   docker-compose up --build -d
   ```
4. The API will be live at `http://localhost:8000`.
