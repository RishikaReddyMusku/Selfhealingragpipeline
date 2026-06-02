import json
from pathlib import Path
from urllib import request

from self_healing_rag.config import settings
from self_healing_rag.documents import DocumentChunk
from self_healing_rag.state import RetrievedDocument


class OllamaEmbeddingFunction:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "prompt": text,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        return data["embedding"]


def build_vector_index(
    chunks: list[DocumentChunk],
    persist_directory: str | Path = settings.vector_store_path,
    collection_name: str = settings.vector_collection_name,
) -> int:
    """Persist chunks in Chroma with local Ollama embeddings."""
    if not chunks:
        return 0

    collection = _get_collection(persist_directory, collection_name)
    ids = [chunk.chunk_id for chunk in chunks]
    collection.upsert(
        ids=ids,
        documents=[chunk.content for chunk in chunks],
        metadatas=[
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ],
    )
    return len(chunks)


def retrieve_from_vector_index(
    query: str,
    persist_directory: str | Path = settings.vector_store_path,
    collection_name: str = settings.vector_collection_name,
    top_k: int = 4,
) -> list[RetrievedDocument]:
    if not query.strip():
        return []

    collection = _get_collection(persist_directory, collection_name)
    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedDocument] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        score = 1 / (1 + float(distance))
        retrieved.append(
            {
                "content": content,
                "source": str(metadata.get("source", "unknown")),
                "score": score,
            }
        )

    return retrieved


def _get_collection(persist_directory: str | Path, collection_name: str):
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Vector retrieval requires ChromaDB. Install project dependencies with "
            "`pip install -e .` or set RETRIEVAL_BACKEND=local."
        ) from exc

    client = chromadb.PersistentClient(path=str(persist_directory))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=OllamaEmbeddingFunction(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        ),
    )
