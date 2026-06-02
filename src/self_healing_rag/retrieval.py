import math
import re
from collections import Counter

from self_healing_rag.index import load_local_index
from self_healing_rag.state import RetrievedDocument


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def retrieve_documents(
    query: str,
    index_path: str = "data/index/chunks.jsonl",
    top_k: int = 4,
) -> list[RetrievedDocument]:
    """Retrieve chunks from the configured backend."""
    from self_healing_rag.config import settings

    backend = settings.retrieval_backend.lower().strip()
    if backend == "vector":
        from self_healing_rag.vector_store import retrieve_from_vector_index

        return retrieve_from_vector_index(query, top_k=top_k)

    if backend == "local":
        return retrieve_from_local_index(query, index_path=index_path, top_k=top_k)

    raise ValueError("RETRIEVAL_BACKEND must be either 'vector' or 'local'.")


def retrieve_from_local_index(
    query: str,
    index_path: str = "data/index/chunks.jsonl",
    top_k: int = 4,
) -> list[RetrievedDocument]:
    """Retrieve chunks using a simple lexical score.

    This is intentionally small and dependency-free for the first working MVP.
    The project can later upgrade this module to Chroma + embeddings without
    changing the rest of the graph.
    """
    chunks = load_local_index(index_path)
    query_terms = _token_counts(query)

    if not chunks or not query_terms:
        return []

    scored_docs: list[RetrievedDocument] = []
    for chunk in chunks:
        chunk_terms = _token_counts(chunk.content)
        score = _cosine_similarity(query_terms, chunk_terms)
        if score > 0:
            scored_docs.append(
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "score": score,
                }
            )

    return sorted(scored_docs, key=lambda item: item["score"], reverse=True)[:top_k]


def _token_counts(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(text))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    common_terms = set(left) & set(right)
    dot_product = sum(left[term] * right[term] for term in common_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)

