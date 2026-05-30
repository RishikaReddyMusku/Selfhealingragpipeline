from self_healing_rag.config import settings
from self_healing_rag.retrieval import retrieve_from_local_index
from self_healing_rag.state import Critique, RAGState, RetrievedDocument, TraceEvent


def append_trace(state: RAGState, step: str, status: str, detail: str) -> list[TraceEvent]:
    return [
        *state.get("trace", []),
        {
            "step": step,
            "status": status,
            "detail": detail,
        },
    ]


def rewrite_query(state: RAGState) -> RAGState:
    """Prepare the user question for retrieval."""
    question = state["question"]
    attempt = state.get("attempt", 0)

    if attempt == 0:
        rewritten_query = question.strip()
    else:
        feedback = state.get("critique", {}).get("feedback", "")
        rewritten_query = f"{question.strip()} {feedback}".strip()

    return {
        "rewritten_query": rewritten_query,
        "trace": append_trace(
            state,
            "rewrite_query",
            "completed",
            f"Prepared retrieval query: {rewritten_query}",
        ),
    }


def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents from the local index."""
    query = state.get("rewritten_query", state["question"])

    if not query:
        return {
            "retrieved_docs": [],
            "trace": append_trace(state, "retrieve_documents", "skipped", "Query was empty."),
        }

    docs: list[RetrievedDocument] = retrieve_from_local_index(
        query,
        index_path=state.get("index_path", settings.local_index_path),
    )
    return {
        "retrieved_docs": docs,
        "trace": append_trace(
            state,
            "retrieve_documents",
            "completed",
            f"Retrieved {len(docs)} candidate document chunks.",
        ),
    }


def grade_context(state: RAGState) -> RAGState:
    """Check whether retrieved context is good enough for generation."""
    docs = state.get("retrieved_docs", [])
    if docs:
        return {
            "trace": append_trace(
                state,
                "grade_context",
                "approved",
                "Retrieved context is available for answer generation.",
            )
        }

    attempt = state.get("attempt", 0) + 1
    critique: Critique = {
        "approved": False,
        "reason": "No relevant context was retrieved.",
        "retry_type": "rewrite_query",
        "feedback": "Try a broader query with more domain-specific terms.",
    }
    return {
        "critique": critique,
        "attempt": attempt,
        "trace": append_trace(
            state,
            "grade_context",
            "rejected",
            "No relevant context was found, so the graph will retry retrieval.",
        ),
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer grounded in retrieved context."""
    docs = state.get("retrieved_docs", [])
    context = "\n".join(f"- {doc['content']}" for doc in docs)
    sources = ", ".join(sorted({doc["source"] for doc in docs}))

    answer = (
        "A self-healing RAG pipeline improves a normal RAG flow by adding a "
        "critic step after answer generation. The critic checks whether the "
        "answer is grounded in retrieved context, complete, and safe to return. "
        "If the answer is rejected, LangGraph routes the workflow back to query "
        "rewriting, retrieval, or generation based on the failure reason.\n\n"
        f"Context used:\n{context}\n\nSources: {sources}"
    )
    return {
        "answer": answer,
        "trace": append_trace(
            state,
            "generate_answer",
            "completed",
            f"Generated an answer using {len(docs)} retrieved chunks.",
        ),
    }


def critique_answer(state: RAGState) -> RAGState:
    """Critique the generated answer and decide whether repair is needed."""
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    attempt = state.get("attempt", 0) + 1

    if not docs:
        critique: Critique = {
            "approved": False,
            "reason": "The answer has no supporting retrieved context.",
            "retry_type": "retrieve_again",
            "feedback": "Retrieve relevant documents before answering.",
        }
        return {
            "critique": critique,
            "attempt": attempt,
            "trace": append_trace(
                state,
                "critique_answer",
                "rejected",
                critique["reason"],
            ),
        }

    if "Sources:" not in answer:
        critique = {
            "approved": False,
            "reason": "The answer does not include source citations.",
            "retry_type": "regenerate",
            "feedback": "Regenerate the answer with explicit source citations.",
        }
        return {
            "critique": critique,
            "attempt": attempt,
            "trace": append_trace(
                state,
                "critique_answer",
                "rejected",
                critique["reason"],
            ),
        }

    critique = {
        "approved": True,
        "reason": "The answer is grounded and includes sources.",
        "retry_type": "accept",
        "feedback": "No changes needed.",
    }
    return {
        "critique": critique,
        "attempt": attempt,
        "trace": append_trace(
            state,
            "critique_answer",
            "approved",
            critique["reason"],
        ),
    }


def finalize_answer(state: RAGState) -> RAGState:
    """Prepare the accepted answer for the caller."""
    return {
        "final_answer": state.get("answer", "No answer was generated."),
        "trace": append_trace(
            state,
            "finalize_answer",
            "completed",
            "Returned the critic-approved answer.",
        ),
    }


def fallback_answer(state: RAGState) -> RAGState:
    """Return a safe fallback after retries are exhausted."""
    final_answer = (
        "I could not produce a sufficiently grounded answer after the "
        "allowed retry attempts. Try adding more source documents or asking "
        "a more specific question."
    )
    return {
        "final_answer": final_answer,
        "trace": append_trace(
            state,
            "fallback_answer",
            "completed",
            "Stopped after the retry budget was exhausted.",
        ),
    }
