import json
from pathlib import Path

from self_healing_rag.documents import DocumentChunk, chunk_documents, load_documents


def build_local_index(
    input_path: str | Path,
    output_path: str | Path = "data/index/chunks.jsonl",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Create a simple local chunk index.

    This keeps the first version runnable and inspectable. A later milestone can
    replace this JSONL index with Chroma embeddings while preserving the same
    chunk shape.
    """
    documents = load_documents(input_path)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "content": chunk.content,
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                    }
                )
                + "\n"
            )

    return chunks


def load_local_index(index_path: str | Path = "data/index/chunks.jsonl") -> list[DocumentChunk]:
    path = Path(index_path)
    if not path.exists():
        return []

    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            chunks.append(
                DocumentChunk(
                    content=record["content"],
                    source=record["source"],
                    chunk_id=record["chunk_id"],
                )
            )

    return chunks

