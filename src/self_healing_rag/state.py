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


class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retrieved_docs: list[RetrievedDocument]
    answer: str
    critique: Critique
    attempt: int
    max_attempts: int
    final_answer: str
    needs_clarification: bool

