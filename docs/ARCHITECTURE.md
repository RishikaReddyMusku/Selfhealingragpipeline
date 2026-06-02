# Architecture

This project is a self-healing Retrieval-Augmented Generation pipeline. It uses
LangGraph to model RAG as a stateful workflow with explicit validation and repair
routes.

## Runtime Flow

```mermaid
flowchart TD
    A["User Question"] --> B["Rewrite Query"]
    B --> C["Retrieve Documents"]
    C --> D["Grade Context"]
    D -->|Useful context| E["Generate Answer"]
    D -->|Weak or missing context| B
    E --> F["Critique Answer"]
    F -->|Approved| G["Final Answer"]
    F -->|Need better retrieval| B
    F -->|Need better wording/citations| E
    F -->|Retry budget exhausted| H["Safe Fallback"]
```

## Component Diagram

```mermaid
flowchart LR
    UI["FastAPI Dashboard / React UI"] --> API["FastAPI API"]
    CLI["Typer CLI"] --> Graph["LangGraph Workflow"]
    API --> Graph
    Graph --> Nodes["RAG Nodes"]
    Nodes --> Retriever["Local Retriever"]
    Retriever --> Index["JSONL Chunk Index"]
    Nodes --> LLM["LLM Adapter"]
    LLM --> Ollama["Ollama: Qwen/Llama"]
    LLM --> OpenAI["OpenAI-compatible API"]
    API --> Trace["Trace + Critic Verdict"]
    Graph --> Trace
```

## Self-Healing Behavior

```mermaid
sequenceDiagram
    participant User
    participant Graph as LangGraph
    participant Retriever
    participant LLM as Ollama/OpenAI
    participant Critic

    User->>Graph: Ask question
    Graph->>Graph: Rewrite query
    Graph->>Retriever: Retrieve relevant chunks
    Retriever-->>Graph: Candidate context
    Graph->>Graph: Grade context
    Graph->>LLM: Generate grounded answer
    LLM-->>Graph: Answer with sources
    Graph->>Critic: Validate grounding and citations
    alt Approved
        Critic-->>Graph: accept
        Graph-->>User: Final answer
    else Needs repair
        Critic-->>Graph: retry route + feedback
        Graph->>Graph: Retry retrieval or generation
    else Max attempts reached
        Graph-->>User: Safe fallback
    end
```

## Demo Strategy

The project can be showcased without paid API usage:

- Run the backend and model locally with Ollama.
- Use Qwen or Llama for generation and critique.
- Record a short demo video showing ingestion, question answering, critic verdict,
  retry trace, and source citations.
- Deploy the static/project page or repo publicly, while keeping model execution
  local in the recorded demo.
