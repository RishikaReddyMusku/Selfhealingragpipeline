import unittest
from unittest.mock import patch

from self_healing_rag.state import RetrievedDocument
from self_healing_rag.retrieval import rerank_documents


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.mock_docs: list[RetrievedDocument] = [
            {"content": "Vanilla CSS is highly flexible.", "source": "css.md", "score": 0.5},
            {"content": "TailwindCSS is utility-first.", "source": "tailwind.md", "score": 0.4},
            {"content": "Next.js is a React framework.", "source": "next.md", "score": 0.3},
        ]

    def test_rerank_none_returns_original_top_k(self):
        with patch("self_healing_rag.config.settings.rerank_backend", "none"):
            result = rerank_documents("CSS flexbox styling", self.mock_docs, top_k=2)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["source"], "css.md")
            self.assertEqual(result[1]["source"], "tailwind.md")

    def test_rerank_cohere_mock_scores_and_sorts(self):
        mock_cohere_response = {
            "results": [
                {"index": 2, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.45},
                {"index": 1, "relevance_score": 0.12},
            ]
        }
        
        with patch("self_healing_rag.config.settings.rerank_backend", "cohere"), \
             patch("self_healing_rag.config.settings.cohere_api_key", "mock-key"), \
             patch("urllib.request.urlopen") as mock_urlopen:
            
            # Mock the urllib response read
            mock_response = mock_urlopen.return_value.__enter__.return_value
            import json
            mock_response.read.return_value = json.dumps(mock_cohere_response).encode("utf-8")
            
            result = rerank_documents("Next.js routing", self.mock_docs, top_k=2)
            
            self.assertEqual(len(result), 2)
            # Index 2 (Next.js) should have score 0.99 and be first
            self.assertEqual(result[0]["source"], "next.md")
            self.assertAlmostEqual(result[0]["score"], 0.99)
            # Index 0 (CSS) should be second with 0.45
            self.assertEqual(result[1]["source"], "css.md")
            self.assertAlmostEqual(result[1]["score"], 0.45)

    def test_rerank_llm_mock_scores_and_sorts(self):
        # LLM returns JSON scoring list
        mock_llm_json = '[{"index": 1, "score": 9.5}, {"index": 0, "score": 3.2}, {"index": 2, "score": 1.0}]'
        
        with patch("self_healing_rag.config.settings.rerank_backend", "llm"), \
             patch("self_healing_rag.llm._chat_with_configured_provider", return_value=mock_llm_json):
            
            result = rerank_documents("Tailwind utility classes", self.mock_docs, top_k=2)
            
            self.assertEqual(len(result), 2)
            # Index 1 (Tailwind) should be first with normalized score 0.95 (9.5/10)
            self.assertEqual(result[0]["source"], "tailwind.md")
            self.assertAlmostEqual(result[0]["score"], 0.95)
            # Index 0 (CSS) should be second with normalized score 0.32 (3.2/10)
            self.assertEqual(result[1]["source"], "css.md")
            self.assertAlmostEqual(result[1]["score"], 0.32)


if __name__ == "__main__":
    unittest.main()
