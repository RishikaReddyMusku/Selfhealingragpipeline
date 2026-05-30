import tempfile
import unittest
from pathlib import Path

from self_healing_rag.documents import SourceDocument, chunk_documents, load_documents


class DocumentTests(unittest.TestCase):
    def test_load_documents_reads_supported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)
            notes = docs_dir / "notes.md"
            notes.write_text("# Notes\n\nSelf-healing RAG uses retry loops.", encoding="utf-8")
            ignored = docs_dir / "image.png"
            ignored.write_text("not a real image", encoding="utf-8")

            documents = load_documents(docs_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, str(notes))
        self.assertIn("retry loops", documents[0].content)

    def test_chunk_documents_creates_overlapping_chunks(self):
        documents = [
            SourceDocument(
                content="abcdefghijklmnopqrstuvwxyz",
                source="alphabet.txt",
            )
        ]

        chunks = chunk_documents(documents, chunk_size=10, chunk_overlap=2)

        self.assertEqual(
            [chunk.content for chunk in chunks],
            [
                "abcdefghij",
                "ijklmnopqr",
                "qrstuvwxyz",
            ],
        )
        self.assertEqual(chunks[0].chunk_id, "alphabet.txt:0")


if __name__ == "__main__":
    unittest.main()
