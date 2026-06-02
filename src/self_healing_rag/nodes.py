from datetime import datetime, timezone
from time import perf_counter

from self_healing_rag.config import settings
from self_healing_rag.llm import critique_grounded_answer, generate_grounded_answer
from self_healing_rag.retrieval import retrieve_documents as retrieve_from_configured_backend
from self_healing_rag.state import Critique, RAGState, RetrievedDocument, TraceEvent


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "this",
    "to",
    "what",
    "why",
}


def append_trace(state: RAGState, step: str, status: str, detail: str) -> list[TraceEvent]:
    started = state.get("run_started_at")
    event: TraceEvent = {
        "step": step,
        "status": status,
        "detail": detail,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(started, (int, float)):
        event["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)

    return [
        *state.get("trace", []),
        event,
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

    docs: list[RetrievedDocument] = retrieve_from_configured_backend(
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
    if _context_is_useful(state.get("rewritten_query", state["question"]), docs):
        return {
            "trace": append_trace(
                state,
                "grade_context",
                "approved",
                "Retrieved context passed score and query-overlap checks.",
            )
        }

    attempt = state.get("attempt", 0) + 1
    retrieval_attempts = state.get("retrieval_attempts", 0) + 1
    critique: Critique = {
        "approved": False,
        "reason": "Retrieved context was missing or too weak.",
        "retry_type": "rewrite_query",
        "feedback": "Try a broader query with more domain-specific terms.",
    }
    return {
        "critique": critique,
        "attempt": attempt,
        "retrieval_attempts": retrieval_attempts,
        "trace": append_trace(
            state,
            "grade_context",
            "rejected",
            "Context quality was too weak, so the graph will retry retrieval.",
        ),
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer grounded in retrieved context."""
    docs = state.get("retrieved_docs", [])
    answer, mode = generate_grounded_answer(state["question"], docs)
    return {
        "answer": answer,
        "trace": append_trace(
            state,
            "generate_answer",
            "completed",
            f"Generated an answer using {len(docs)} retrieved chunks via {mode} mode.",
        ),
    }


def critique_answer(state: RAGState) -> RAGState:
    """Critique the generated answer and decide whether repair is needed."""
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    attempt = state.get("attempt", 0) + 1
    critique, mode = critique_grounded_answer(state["question"], answer, docs)

    citation_issue = _find_citation_issue(answer, docs)
    if citation_issue:
        critique = {
            "approved": False,
            "reason": citation_issue,
            "retry_type": "regenerate",
            "feedback": "Regenerate the answer with exact source names from retrieved context.",
        }

    generation_attempts = state.get("generation_attempts", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)
    if not critique["approved"]:
        if critique["retry_type"] == "regenerate":
            generation_attempts += 1
        elif critique["retry_type"] in {"rewrite_query", "retrieve_again"}:
            retrieval_attempts += 1

    return {
        "critique": critique,
        "attempt": attempt,
        "generation_attempts": generation_attempts,
        "retrieval_attempts": retrieval_attempts,
        "trace": append_trace(
            state,
            "critique_answer",
            "approved" if critique["approved"] else "rejected",
            f"{critique['reason']} ({mode} mode)",
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


def clarify_question(state: RAGState) -> RAGState:
    """Ask the caller for missing information when the critic requests clarification."""
    critique = state.get("critique") or {}
    feedback = str(critique.get("feedback") or "").strip()
    if feedback:
        final_answer = (
            "I need a bit more detail to answer accurately.\n\n"
            f"Clarifying question: {feedback}\n\n"
            "Sources: none"
        )
    else:
        final_answer = (
            "I need a bit more detail to answer accurately. "
            "Can you clarify your question?\n\n"
            "Sources: none"
        )

    return {
        "needs_clarification": True,
        "final_answer": final_answer,
        "trace": append_trace(
            state,
            "clarify_question",
            "completed",
            "The critic requested clarification instead of another retry.",
        ),
    }


def _context_is_useful(query: str, docs: list[RetrievedDocument]) -> bool:
    if not docs:
        return False

    best_score = max(doc["score"] for doc in docs)
    if best_score < settings.min_context_score:
        return False

    query_terms = _important_terms(query)
    if not query_terms:
        return True

    context_terms = _important_terms(" ".join(doc["content"] for doc in docs))
    overlap = len(query_terms & context_terms) / len(query_terms)
    return overlap >= settings.min_context_term_overlap


def _important_terms(text: str) -> set[str]:
    import re

    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9]+", text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def _find_citation_issue(answer: str, docs: list[RetrievedDocument]) -> str | None:
    if not docs:
        return None

    cited_sources = _extract_cited_sources(answer)
    if not cited_sources:
        return "Answer did not include usable source citations."

    valid_sources = {doc["source"] for doc in docs}
    unknown_sources = sorted(source for source in cited_sources if source not in valid_sources)
    if unknown_sources:
        return f"Answer cited unknown sources: {', '.join(unknown_sources)}."

    return None


def _extract_cited_sources(answer: str) -> set[str]:
    for line in answer.splitlines():
        if line.lower().startswith("sources:"):
            raw_sources = line.split(":", 1)[1]
            return {
                source.strip()
                for source in raw_sources.split(",")
                if source.strip() and source.strip().lower() != "none"
            }

    return set()
