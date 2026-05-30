from langgraph.graph import END, StateGraph

from self_healing_rag.config import settings
from self_healing_rag.nodes import (
    critique_answer,
    fallback_answer,
    finalize_answer,
    generate_answer,
    grade_context,
    retrieve_documents,
    rewrite_query,
)
from self_healing_rag.state import RAGState


def route_after_context_grade(state: RAGState) -> str:
    if state.get("retrieved_docs"):
        return "generate_answer"

    critique = state.get("critique")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", settings.max_attempts)

    if critique and attempt >= max_attempts:
        return "fallback_answer"

    if critique and critique["retry_type"] == "rewrite_query":
        return "rewrite_query"

    return "generate_answer"


def route_after_critique(state: RAGState) -> str:
    critique = state["critique"]
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", settings.max_attempts)

    if critique["approved"]:
        return "finalize_answer"

    if attempt >= max_attempts:
        return "fallback_answer"

    if critique["retry_type"] in {"rewrite_query", "retrieve_again"}:
        return "rewrite_query"

    if critique["retry_type"] == "regenerate":
        return "generate_answer"

    return "fallback_answer"


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_context", grade_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("critique_answer", critique_answer)
    graph.add_node("finalize_answer", finalize_answer)
    graph.add_node("fallback_answer", fallback_answer)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    graph.add_edge("generate_answer", "critique_answer")
    graph.add_conditional_edges(
        "critique_answer",
        route_after_critique,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
            "finalize_answer": "finalize_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    graph.add_edge("finalize_answer", END)
    graph.add_edge("fallback_answer", END)

    return graph.compile()


def ask(question: str) -> str:
    result = ask_with_trace(question)
    return result["final_answer"]


def ask_with_trace(question: str) -> RAGState:
    app = build_graph()
    return app.invoke(
        {
            "question": question,
            "attempt": 0,
            "max_attempts": settings.max_attempts,
            "trace": [],
        }
    )
