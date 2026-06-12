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
    """Retrieve chunks from the configured backend, with optional reranking."""
    from self_healing_rag.config import settings

    backend = settings.retrieval_backend.lower().strip()
    rerank_backend = settings.rerank_backend.lower().strip()

    retrieve_k = top_k
    if rerank_backend != "none":
        retrieve_k = top_k * settings.rerank_candidates_factor

    docs: list[RetrievedDocument] = []
    if backend == "vector":
        from self_healing_rag.vector_store import retrieve_from_vector_index
        try:
            docs = retrieve_from_vector_index(query, top_k=retrieve_k)
        except Exception:
            # Prefer graceful degradation to lexical retrieval over hard failure.
            docs = retrieve_from_local_index(query, index_path=index_path, top_k=retrieve_k)
    elif backend == "hybrid":
        docs = retrieve_hybrid(query, index_path=index_path, top_k=retrieve_k)
    elif backend == "local":
        docs = retrieve_from_local_index(query, index_path=index_path, top_k=retrieve_k)
    else:
        raise ValueError("RETRIEVAL_BACKEND must be one of 'vector', 'hybrid', or 'local'.")

    if rerank_backend != "none" and docs:
        docs = rerank_documents(query, docs, top_k=top_k)
    else:
        docs = docs[:top_k]

    return docs


def rerank_documents(
    query: str,
    docs: list[RetrievedDocument],
    top_k: int = 4,
) -> list[RetrievedDocument]:
    """Sort and prune documents using the configured rerank backend."""
    from self_healing_rag.config import settings
    backend = settings.rerank_backend.lower().strip()

    if backend == "llm":
        from self_healing_rag.llm import rerank_with_llm
        ranked = rerank_with_llm(query, docs)
    elif backend == "cohere":
        if not settings.cohere_api_key:
            print("Cohere Rerank key is missing. Set COHERE_API_KEY. Falling back.")
            ranked = docs
        else:
            ranked = rerank_with_cohere(
                query, docs, settings.cohere_api_key, settings.cohere_model
            )
    elif backend == "cross-encoder":
        ranked = rerank_with_cross_encoder(query, docs, settings.cross_encoder_model)
    else:
        ranked = docs

    return ranked[:top_k]


def rerank_with_cohere(
    query: str,
    docs: list[RetrievedDocument],
    api_key: str,
    model: str,
) -> list[RetrievedDocument]:
    import json
    from urllib import request

    try:
        payload = {
            "model": model,
            "query": query,
            "documents": [doc["content"] for doc in docs],
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            "https://api.cohere.com/v1/rerank",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        scored_docs = []
        for result in data.get("results", []):
            idx = result["index"]
            score = result["relevance_score"]
            doc = docs[idx].copy()
            doc["score"] = score
            scored_docs.append(doc)

        returned_indices = {res["index"] for res in data.get("results", [])}
        for idx, doc in enumerate(docs):
            if idx not in returned_indices:
                doc_copy = doc.copy()
                doc_copy["score"] = 0.0
                scored_docs.append(doc_copy)

        return sorted(scored_docs, key=lambda x: x["score"], reverse=True)
    except Exception as exc:
        print(f"Cohere Rerank failed, returning original order: {exc}")
        return docs


def rerank_with_cross_encoder(
    query: str,
    docs: list[RetrievedDocument],
    model_name: str,
) -> list[RetrievedDocument]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("sentence-transformers not installed. Install with `pip install sentence-transformers` to use local cross-encoders.")
        return docs

    try:
        model = CrossEncoder(model_name)
        pairs = [[query, doc["content"]] for doc in docs]
        scores = model.predict(pairs)

        scored_docs = []
        for doc, score in zip(docs, scores):
            doc_copy = doc.copy()
            doc_copy["score"] = float(score)
            scored_docs.append(doc_copy)

        return sorted(scored_docs, key=lambda x: x["score"], reverse=True)
    except Exception as exc:
        print(f"Local CrossEncoder Rerank failed, returning original order: {exc}")
        return docs



def retrieve_hybrid(
    query: str,
    index_path: str = "data/index/chunks.jsonl",
    top_k: int = 4,
) -> list[RetrievedDocument]:
    """Combine lexical and vector retrieval, then rank and deduplicate."""
    from self_healing_rag.vector_store import retrieve_from_vector_index

    local_docs = retrieve_from_local_index(query, index_path=index_path, top_k=top_k * 2)
    try:
        vector_docs = retrieve_from_vector_index(query, top_k=top_k * 2)
    except Exception:
        vector_docs = []

    merged: dict[tuple[str, str], RetrievedDocument] = {}
    for doc in [*local_docs, *vector_docs]:
        key = (doc["source"], doc["content"])
        existing = merged.get(key)
        if existing is None or doc["score"] > existing["score"]:
            merged[key] = doc

    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


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

