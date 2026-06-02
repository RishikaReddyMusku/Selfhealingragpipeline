import unittest
from unittest.mock import patch

from self_healing_rag.llm import (
    _parse_critique,
    critique_grounded_answer,
    generate_grounded_answer,
)
from self_healing_rag.state import RetrievedDocument


class LLMFallbackTests(unittest.TestCase):
    def test_fallback_generation_uses_question_context_and_sources(self):
        docs: list[RetrievedDocument] = [
            {
                "content": "Self-healing RAG critiques generated answers.",
                "source": "rag.md",
                "score": 0.8,
            }
        ]

        with patch("self_healing_rag.llm._use_llm", return_value=False):
            answer, mode = generate_grounded_answer("What does it do?", docs)

        self.assertEqual(mode, "fallback")
        self.assertIn("What does it do?", answer)
        self.assertIn("Self-healing RAG critiques", answer)
        self.assertIn("Sources: rag.md", answer)

    def test_fallback_critic_rejects_missing_sources(self):
        docs: list[RetrievedDocument] = [
            {
                "content": "Self-healing RAG critiques generated answers.",
                "source": "rag.md",
                "score": 0.8,
            }
        ]

        with patch("self_healing_rag.llm._use_llm", return_value=False):
            critique, mode = critique_grounded_answer("Question?", "Answer without citations.", docs)

        self.assertEqual(mode, "fallback")
        self.assertFalse(critique["approved"])
        self.assertEqual(critique["retry_type"], "regenerate")

    def test_parse_critique_validates_structured_json(self):
        critique = _parse_critique(
            """
            {
              "approved": true,
              "reason": "Supported by context.",
              "retry_type": "accept",
              "feedback": "No changes needed."
            }
            """
        )

        self.assertTrue(critique["approved"])
        self.assertEqual(critique["retry_type"], "accept")

    def test_parse_critique_rejects_invalid_retry_type(self):
        with self.assertRaises(ValueError):
            _parse_critique(
                """
                {
                  "approved": false,
                  "reason": "Bad route.",
                  "retry_type": "start_over",
                  "feedback": "Try again."
                }
                """
            )

    def test_parse_critique_accepts_fenced_json(self):
        critique = _parse_critique(
            """
            ```json
            {
              "approved": false,
              "reason": "Missing sources.",
              "retry_type": "regenerate",
              "feedback": "Add citations."
            }
            ```
            """
        )

        self.assertFalse(critique["approved"])
        self.assertEqual(critique["retry_type"], "regenerate")


if __name__ == "__main__":
    unittest.main()
