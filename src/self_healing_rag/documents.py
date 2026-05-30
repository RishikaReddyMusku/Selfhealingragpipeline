from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class SourceDocument:
    content: str
    source: str


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    source: str
    chunk_id: str


def load_documents(input_path: str | Path) -> list[SourceDocument]:
    """Load text-like documents from a file or directory."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")

    files = [path] if path.is_file() else sorted(_iter_supported_files(path))
    documents: list[SourceDocument] = []

    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(SourceDocument(content=content, source=str(file_path)))

    return documents


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Split documents into overlapping chunks for retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []

    for document in documents:
        start = 0
        chunk_number = 0
        content = document.content

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        source=document.source,
                        chunk_id=f"{Path(document.source).name}:{chunk_number}",
                    )
                )

            if end == len(content):
                break

            start = end - chunk_overlap
            chunk_number += 1

    return chunks


def _iter_supported_files(directory: Path):
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path

