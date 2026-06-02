import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from self_healing_rag.index import build_local_index
from self_healing_rag.retrieval import retrieve_from_local_index, retrieve_hybrid


class RetrievalTests(unittest.TestCase):
    def test_retrieve_from_local_index_returns_best_matching_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            docs_dir = workspace / "docs"
            docs_dir.mkdir()
            index_path = workspace / "chunks.jsonl"

            (docs_dir / "rag.md").write_text(
                "Self-healing RAG systems critique answers and retry generation.",
                encoding="utf-8",
            )
            (docs_dir / "other.md").write_text(
                "This file is about cooking recipes and grocery lists.",
                encoding="utf-8",
            )

            build_local_index(docs_dir, index_path, chunk_size=200, chunk_overlap=0)
            results = retrieve_from_local_index(
                "How does a RAG critic retry bad answers?",
                str(index_path),
                top_k=1,
            )

        self.assertEqual(len(results), 1)
        self.assertIn("Self-healing RAG", results[0]["content"])

    def test_retrieve_hybrid_merges_and_deduplicates_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            docs_dir = workspace / "docs"
            docs_dir.mkdir()
            index_path = workspace / "chunks.jsonl"

            (docs_dir / "rag.md").write_text(
                "Self-healing RAG systems critique answers and retry generation.",
                encoding="utf-8",
            )

            build_local_index(docs_dir, index_path, chunk_size=200, chunk_overlap=0)

            vector_docs = [
                {
                    "content": "Self-healing RAG systems critique answers and retry generation.",
                    "source": str(docs_dir / "rag.md"),
                    "score": 0.99,
                }
            ]
            with patch(
                "self_healing_rag.vector_store.retrieve_from_vector_index",
                return_value=vector_docs,
            ):
                results = retrieve_hybrid(
                    "How does a RAG critic retry bad answers?",
                    str(index_path),
                    top_k=3,
                )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(len(results), 1)
        self.assertIn("Self-healing RAG", results[0]["content"])


if __name__ == "__main__":
    unittest.main()
