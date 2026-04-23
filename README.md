# Agentic RAG Pipeline with LangGraph

An advanced, fully asynchronous AI agent that intelligently routes queries, performs semantic caching, retrieves internal documents, and searches the web. Built using **LangGraph**, **FastAPI**, **Redis Stack**, and multiple AI models (OpenAI, Cohere).

## System Architecture & Workflow

The system utilizes an agentic workflow with parallel routing and a dedicated RAG feedback loop. 

Below is the execution graph representing the LangGraph state machine:

```mermaid
flowchart TD
    %% ===== Styling =====
    classDef process fill:#00b894,color:#ffffff,stroke:#55efc4,stroke-width:1.5px;
    classDef decision fill:#6c5ce7,color:#ffffff,stroke:#a29bfe,stroke-width:1.5px;
    classDef cache fill:#0984e3,color:#ffffff,stroke:#74b9ff,stroke-width:1.5px;
    classDef endpoint fill:#d63031,color:#ffffff,stroke:#ff7675,stroke-width:2px;
    classDef loop stroke:#ff7675,stroke-width:2px,stroke-dasharray: 5 5;

    %% ===== Main Flow =====
    Start((Start)):::endpoint
    End((End)):::endpoint

    QueryRewriter["Query Rewriter"]:::process
    CacheCheck{"Cache Hit?"}:::cache
    Analyzer{"Analyze Query"}:::decision

    WebSearch["Web Search"]:::process
    WebRerank["Web Reranker"]:::process

    Final["Draft Final Response"]:::process
    CacheStore["Store in Cache"]:::cache

    %% ===== Main Routing =====
    Start --> QueryRewriter
    QueryRewriter --> CacheCheck

    CacheCheck -- Yes --> End
    CacheCheck -- No --> Analyzer

    Analyzer -- "Stop Early" --> End
    Analyzer -- "Needs Web Data" --> WebSearch
    Analyzer -- "Needs RAG" --> GetChunks

    %% ===== Web Pipeline =====
    WebSearch --> WebRerank --> Final

    %% ===== RAG Subgraph =====
    subgraph RAG_Pipeline ["RAG Pipeline"]
        direction TB

        GetChunks["Retrieve Chunks"]:::process
        Rerank{"Relevance Score ≥ 0.5?"}:::decision
        Critique{"Context Sufficient?"}:::decision
        Rewrite["Rewrite Query"]:::process

        GetChunks --> Rerank

        Rerank -- Yes --> Critique
        Rerank -- No --> WebSearch

        Critique -- Yes --> Final
        Critique -- No --> Rewrite

        Rewrite -->|"Iterate (max 2)"| GetChunks
    end

    %% ===== Finalization =====
    Final --> CacheStore --> End
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
