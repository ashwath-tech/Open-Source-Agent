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
    classDef process fill:#00b894,color:#fff,stroke:#55efc4;

    %% Main Nodes
    Start((START)):::endpoint
    End((END)):::endpoint
    
    QueryRewriter[query_rewriter]:::process
    CheckCache{check_cache}:::cache
    Analyzer{analyzer}:::decision
    WebSearch[web_search]
    WebReranker[web_reranker]
    Final[draft_final]
    StoreCache[store_cache]:::cache

    %% Main Graph Routing
    Start --> QueryRewriter
    QueryRewriter --> CheckCache
    CheckCache -- "Cache Hit (in_cache == 1)" --> End
    CheckCache -- "Cache Miss" --> Analyzer
    
    Analyzer -- "stop_now == True" --> End
    Analyzer -- "Has web_query" --> WebSearch

    %% Web Branch
    WebSearch --> WebReranker
    WebReranker --> Final

    %% RAG Subgraph Definition
    subgraph RAG_Pipeline ["rag_graph (Subgraph)"]
        direction TB
        GetChunks[get_chunks]
        Reranker{reranker}:::decision
        Critique{critique}:::decision
        Rewriter[rewriter]:::loop

        GetChunks --> Reranker
        Reranker -- "Score >= 0.5" --> Critique
        
        %% The Feedback Loop
        Critique -- "break_loop == False\n(Poor Context)" --> Rewriter
        Rewriter -- "Rewrite Query\nloop_number += 1" --> GetChunks
    end

    %% Subgraph Connections
    Analyzer -- "Has rag_query" --> GetChunks
    
    %% Dynamic Fallback Edge
    Reranker -- "Score < 0.5\n(Fallback Triggered)" --> WebSearch
    
    Critique -- "break_loop == True\nOR loop_number == 2" --> Final
    
    Final --> StoreCache
    StoreCache --> End
```

### Advanced Pipeline Features Explained
1. **Contextual Query Rewriting:** The query_rewriter node analyzes the user's input against the ongoing chat history. It resolves pronouns (e.g., "how do I configure it?") into standalone, vector-friendly search queries (e.g., "how to configure llfuse direct_io"), ensuring downstream retrieval is highly accurate.
2. **The Dynamic Web Fallback:** The system acts with defensive engineering. If the rag_pipeline retrieves documents from ChromaDB but the Cohere Reranker determines the relevance score is below a strict threshold (0.5), the graph abandons the internal context to prevent hallucinations. It dynamically triggers a fallback edge, routing the query directly to the web_search node.
2. **Critique:** An LLM evaluates if the reranked context answers the query.
3. **Rag Critique Loop:** 
    - *Retrieve & Rerank*: Fetches chunks from the vector database and scores them.
    - *Critique*: An LLM evaluates if the context actually answers the user's intent.
    - *Rewrite (The Loop)*: If the context is poor, the rewriter node adjusts the search query based on the critique and loops the state back to get_chunks. It explicitly breaks after 2 loops to prevent infinite execution.
4. **Dual-Reranking Architecture:** Reranking is applied to both internal and external data. The web_reranker node acts as a noise filter for Tavily Search results, applying Cohere's scoring model to raw web scrapes to extract only the highest-density information before final drafting.

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
