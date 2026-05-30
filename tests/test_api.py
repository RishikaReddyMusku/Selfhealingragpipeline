import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from self_healing_rag.api import app


class APITests(unittest.TestCase):
    def test_ingest_and_ask_endpoints_return_traceable_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            docs_dir = workspace / "docs"
            docs_dir.mkdir()
            index_path = workspace / "chunks.jsonl"

            (docs_dir / "rag.md").write_text(
                "Self-healing RAG critiques generated answers and retries weak responses.",
                encoding="utf-8",
            )

            client = TestClient(app)
            ingest_response = client.post(
                "/ingest",
                json={
                    "input_path": str(docs_dir),
                    "output_path": str(index_path),
                },
            )
            ask_response = client.post(
                "/ask",
                json={
                    "question": "How does self-healing RAG handle weak answers?",
                    "index_path": str(index_path),
                },
            )

        self.assertEqual(ingest_response.status_code, 200)
        self.assertEqual(ingest_response.json()["chunks_indexed"], 1)
        self.assertEqual(ask_response.status_code, 200)

        payload = ask_response.json()
        self.assertTrue(payload["critique"]["approved"])
        self.assertGreaterEqual(payload["attempts"], 1)
        self.assertGreaterEqual(len(payload["trace"]), 5)
        self.assertIn("Sources:", payload["answer"])

    def test_dashboard_serves_static_html(self):
        client = TestClient(app)

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Self-Healing RAG Pipeline", response.text)


if __name__ == "__main__":
    unittest.main()
