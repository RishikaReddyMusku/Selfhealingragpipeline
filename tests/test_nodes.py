import unittest
from unittest.mock import patch

from self_healing_rag.nodes import critique_answer
from self_healing_rag.state import Critique, RAGState, RetrievedDocument


class NodesTests(unittest.TestCase):
    def test_critique_answer_rejects_missing_sources(self):
        docs: list[RetrievedDocument] = [
            {"content": "RAG can retry.", "source": "rag.md", "score": 0.9}
        ]
        llm_critique: Critique = {
            "approved": True,
            "reason": "Looks good.",
            "retry_type": "accept",
            "feedback": "None",
        }
        state: RAGState = {
            "question": "How does it heal?",
            "answer": "It retries weak answers.",
            "retrieved_docs": docs,
            "attempt": 0,
            "generation_attempts": 0,
            "retrieval_attempts": 0,
            "trace": [],
        }

        with patch(
            "self_healing_rag.nodes.critique_grounded_answer",
            return_value=(llm_critique, "llm"),
        ):
            updated = critique_answer(state)

        self.assertFalse(updated["critique"]["approved"])
        self.assertEqual(updated["critique"]["retry_type"], "regenerate")
        self.assertEqual(updated["generation_attempts"], 1)

    def test_critique_answer_rejects_unknown_sources(self):
        docs: list[RetrievedDocument] = [
            {"content": "RAG can retry.", "source": "rag.md", "score": 0.9}
        ]
        llm_critique: Critique = {
            "approved": True,
            "reason": "Looks good.",
            "retry_type": "accept",
            "feedback": "None",
        }
        state: RAGState = {
            "question": "How does it heal?",
            "answer": "It retries weak answers.\n\nSources: other.md",
            "retrieved_docs": docs,
            "attempt": 0,
            "generation_attempts": 0,
            "retrieval_attempts": 0,
            "trace": [],
        }

        with patch(
            "self_healing_rag.nodes.critique_grounded_answer",
            return_value=(llm_critique, "llm"),
        ):
            updated = critique_answer(state)

        self.assertFalse(updated["critique"]["approved"])
        self.assertEqual(updated["critique"]["retry_type"], "regenerate")
        self.assertEqual(updated["generation_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
