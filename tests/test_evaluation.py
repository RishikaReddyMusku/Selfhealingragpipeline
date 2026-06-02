import unittest

from self_healing_rag.evaluation import summarize_results


class EvaluationTests(unittest.TestCase):
    def test_summarize_results_calculates_basic_rag_metrics(self):
        summary = summarize_results(
            [
                {
                    "question": "Question one?",
                    "expected_behavior": "Should be approved.",
                    "result": {
                        "final_answer": "Answer.\n\nSources: notes.md",
                        "attempt": 1,
                        "critique": {
                            "approved": True,
                            "retry_type": "accept",
                            "reason": "Grounded.",
                            "feedback": "None.",
                        },
                        "trace": [{"step": "finalize_answer", "status": "completed", "detail": "ok"}],
                    },
                },
                {
                    "question": "Question two?",
                    "expected_behavior": "Should fallback.",
                    "result": {
                        "final_answer": "Could not answer.",
                        "attempt": 3,
                        "critique": {
                            "approved": False,
                            "retry_type": "rewrite_query",
                            "reason": "Weak context.",
                            "feedback": "Broaden query.",
                        },
                        "trace": [{"step": "fallback_answer", "status": "completed", "detail": "stop"}],
                    },
                },
            ]
        )

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["approval_rate"], 0.5)
        self.assertEqual(summary["citation_rate"], 0.5)
        self.assertEqual(summary["fallback_rate"], 0.5)
        self.assertEqual(summary["average_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
