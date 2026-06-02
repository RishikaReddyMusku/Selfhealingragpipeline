from typing import Literal, TypedDict


RetryType = Literal[
    "accept",
    "rewrite_query",
    "retrieve_again",
    "regenerate",
    "clarify",
    "fallback",
]


class RetrievedDocument(TypedDict):
    content: str
    source: str
    score: float


class Critique(TypedDict):
    approved: bool
    reason: str
    retry_type: RetryType
    feedback: str


class TraceEvent(TypedDict, total=False):
    step: str
    status: str
    detail: str
    at: str
    elapsed_ms: float


class ObservabilityMetrics(TypedDict):
    total_elapsed_ms: float
    total_steps: int
    retrieval_attempts: int
    generation_attempts: int


class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    index_path: str
    retrieved_docs: list[RetrievedDocument]
    answer: str
    critique: Critique
    attempt: int
    max_attempts: int
    retrieval_attempts: int
    max_retrieval_attempts: int
    generation_attempts: int
    max_generation_attempts: int
    run_started_at: float
    run_started_iso: str
    metrics: ObservabilityMetrics
    final_answer: str
    needs_clarification: bool
    trace: list[TraceEvent]
