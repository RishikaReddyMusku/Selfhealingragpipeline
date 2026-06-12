# Free-Tier Showcase Plan

The project is designed to be shown publicly without paying for hosted LLM
inference, and now includes **reranking** and **Ragas evaluation** that both
run entirely free using local models.

---

## Reranking (Free Tiers)

Reranking re-scores retrieved chunks so the most relevant ones appear first,
improving answer quality without changing the retrieval architecture.

Set `RERANK_BACKEND` in your `.env` file:

| Backend | Cost | How |
|---|---|---|
| `none` | Free | No reranking (default) |
| `llm` | Free (local) | Uses your Ollama LLM to score each chunk |
| `cross-encoder` | Free (local) | `BAAI/bge-reranker-base` via `sentence-transformers` |
| `cohere` | Free trial | Cohere Rerank API — set `COHERE_API_KEY` |

### Quickstart

```bash
# Install rerank extras
pip install -e ".[rerank]"
```

```env
# .env — fully local cross-encoder (no API key)
RERANK_BACKEND=cross-encoder
RERANK_TOP_K=4
RERANK_CANDIDATES_FACTOR=3
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
```

```env
# .env — LLM-based scoring via Ollama
RERANK_BACKEND=llm
USE_LLM=true
LLM_PROVIDER=ollama
```

---

## Ragas Evaluation (Free Tier)

[Ragas](https://docs.ragas.io) measures RAG quality with LLM-as-a-judge metrics:
faithfulness, answer relevance, context precision, and context recall.

When `OPENAI_API_KEY` is **not** set, evaluation automatically uses your local
Ollama models — fully free, no cloud API needed.

### Quickstart

```bash
# Install evaluation extras
pip install -e ".[ragas]"

# Pull Ollama models if not already pulled
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

```bash
# Run evaluation via CLI
self-healing-rag eval --questions data/demo_questions.json

# Or via API
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"questions_path": "data/demo_questions.json"}'
```

The evaluation suite reports:
- **Faithfulness** — Is the answer supported by retrieved context?
- **Answer Relevance** — Does the answer address the question?
- **Context Precision** — Is the retrieved context relevant?
- **Context Recall** — Does retrieved context cover the answer?
- **Approval rate** — Fraction of answers approved by the critic
- **Citation rate** — Fraction of answers with valid source citations
- **Fallback rate** — Fraction of answers that hit the retry budget
- **Average attempts** — Mean number of self-healing loop iterations

---

## Recommended Free-Tier Stack

```text
Ollama (qwen2.5:7b-instruct + nomic-embed-text)
  + Cross-Encoder reranking (BAAI/bge-reranker-base, runs locally)
  + Ragas evaluation with Ollama judge + embeddings
  = Zero cost, zero cloud dependency
```

## Why Not Host The Full LLM App On A Free Tier?

Free web tiers usually do not provide enough CPU/RAM/GPU capacity to run Qwen or
Llama reliably. They also tend to sleep between requests, which makes local-model
startup time poor for demos.

Instead, the strongest portfolio story is:

```text
Public repo + architecture diagrams + local Ollama demo video
```

## Resume Bullet

Built a self-healing RAG pipeline with LangGraph, FastAPI, and Ollama/Qwen that
retrieves local documents, reranks candidates with a local cross-encoder, generates
grounded answers, critiques responses for source support, automatically retries
retrieval or generation when quality checks fail, and evaluates pipeline quality
end-to-end with Ragas metrics — all running free on local hardware.

## GitHub Checklist

- README explains the problem and architecture.
- `docs/ARCHITECTURE.md` contains Mermaid diagrams.
- `docs/DEMO_GUIDE.md` explains how the video was recorded.
- `docs/FREE_TIER_SHOWCASE.md` documents free-tier reranking and evaluation.
- `data/demo_questions.json` contains reproducible demo prompts.
- Setup uses Ollama by default, with OpenAI as an optional provider.
- `.env.example` documents all reranking and evaluation config knobs.
