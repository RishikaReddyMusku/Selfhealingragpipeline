# Free-Tier Showcase Plan

The project is designed to be shown publicly without paying for hosted LLM
inference.

## Recommended Approach

- Keep the GitHub repository public with clear architecture diagrams.
- Record the working RAG pipeline locally with Ollama.
- Add the demo video link to the README.
- Deploy only lightweight static materials if needed, such as docs or a frontend
  preview.

## Why Not Host The Full LLM App On A Free Tier?

Free web tiers usually do not provide enough CPU/RAM/GPU capacity to run Qwen or
Llama reliably. They also tend to sleep between requests, which makes local-model
startup time poor for demos.

Instead, the strongest portfolio story is:

```text
Public repo + architecture diagrams + local Ollama demo video
```

This shows the real engineering work while avoiding paid model APIs.

## Resume Bullet

Built a self-healing RAG pipeline with LangGraph, FastAPI, and Ollama/Qwen that
retrieves local documents, generates grounded answers, critiques responses for
source support, and automatically retries retrieval or generation when quality
checks fail.

## GitHub Checklist

- README explains the problem and architecture.
- `docs/ARCHITECTURE.md` contains Mermaid diagrams.
- `docs/DEMO_GUIDE.md` explains how the video was recorded.
- `data/demo_questions.json` contains reproducible demo prompts.
- A demo video link is added near the top of the README.
- Setup uses Ollama by default, with OpenAI as an optional provider.
