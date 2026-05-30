# Self-Healing RAG Notes

A self-healing RAG pipeline extends normal retrieval-augmented generation with feedback loops.

The basic RAG flow is:

1. Retrieve relevant context.
2. Generate an answer from that context.
3. Return the answer to the user.

The self-healing flow adds quality checks:

1. Rewrite the query if retrieval is weak.
2. Grade whether the retrieved context is relevant.
3. Generate an answer grounded only in retrieved context.
4. Critique the answer for grounding, completeness, and source support.
5. Retry retrieval or generation when the critic rejects the answer.

LangGraph is useful for this because it can represent stateful workflows with conditional routing and cycles.

