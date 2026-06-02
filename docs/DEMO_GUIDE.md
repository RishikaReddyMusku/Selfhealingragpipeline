# Demo Guide

Use this guide to record a short resume-ready walkthrough without paid API usage.
The model runs locally through Ollama.

## Setup

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama serve
```

In another terminal:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
uvicorn self_healing_rag.api:app --reload
```

Use these `.env` values:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434
USE_LLM=true
LLM_FALLBACK_ENABLED=false
```

Then open `http://127.0.0.1:8000`.

## Recording Structure

Aim for a 2-3 minute video.

1. Show the README and architecture diagram.
2. Start the FastAPI app.
3. Open the dashboard.
4. Ingest `data/sample_docs`.
5. Ask a question from `data/demo_questions.json`.
6. Point out:
   - retrieved sources
   - generated answer
   - critic verdict
   - attempts count
   - workflow trace
7. Ask one broader or weaker question to show the retry/fallback path.

## Suggested Talking Points

- "This is not a basic one-shot RAG demo. The workflow validates retrieved
  context and critiques the generated answer before returning it."
- "LangGraph models the pipeline as a state machine, so failed answers can route
  back to retrieval or generation."
- "The demo uses Ollama locally with Qwen/Llama, so it does not require paid API
  credits."
- "The API response includes trace events, which makes the system easier to debug
  and evaluate."

## Demo Prompts

Good first prompt:

```text
What makes this RAG pipeline self-healing?
```

Good trace-focused prompt:

```text
How does the critic decide whether to retry retrieval or generation?
```

Good fallback/recovery prompt:

```text
What production monitoring tools does this project integrate with?
```

The final prompt should have weak evidence in the sample docs, which helps show
why grounded systems need safe fallback behavior.
