# Project Plan

## Goal

Build a resume-quality self-healing RAG pipeline that can retrieve documents, generate grounded answers, critique its own output, and retry when the answer does not meet quality standards.

## MVP Milestones

### 1. Repo Foundation

- Create a clean Git repository.
- Add README, roadmap, project config, and source layout.
- Define the LangGraph state and workflow nodes.

### 2. Basic RAG

- Load local markdown, text, and PDF files.
- Split documents into chunks.
- Embed chunks into a vector store.
- Retrieve relevant chunks for a user question.
- Generate an answer from retrieved context.

### 3. Self-Healing Loop

- Add context grading before generation.
- Add critic node after generation.
- Return structured critic output:
  - `approved`
  - `reason`
  - `retry_type`
  - `feedback`
- Route rejected answers back to retrieval or generation.
- Stop safely after `max_attempts`.

### 4. API And Demo

- Add FastAPI endpoint.
- Add a simple CLI command.
- Include sample documents and demo questions.

### 5. Evaluation

- Add test questions.
- Track answer approval rate.
- Track retry count.
- Track whether final answers cite retrieved context.
- Optional: add RAGAS or DeepEval.

## Suggested GitHub Issues

1. Scaffold LangGraph workflow.
2. Add document ingestion pipeline.
3. Add Chroma vector store.
4. Implement answer generation prompt.
5. Implement critic prompt with structured JSON output.
6. Add retry routing.
7. Add CLI command.
8. Add FastAPI endpoint.
9. Add evaluation dataset.
10. Add demo video or screenshots.

## Definition Of Done

The project is resume-ready when:

- A user can ingest documents.
- A user can ask a question.
- The graph can retrieve, answer, critique, and retry.
- The final answer includes source references.
- The README includes setup instructions and architecture.
- The repo has clear commits showing steady development.

