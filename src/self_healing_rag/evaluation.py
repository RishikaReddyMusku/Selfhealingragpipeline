import json
from pathlib import Path
from statistics import mean
from typing import Any

from self_healing_rag.state import RAGState


def load_eval_questions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation file must contain a JSON list.")

    questions: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError("Each evaluation item must contain a question.")
        questions.append(
            {
                "question": str(item["question"]),
                "expected_behavior": str(item.get("expected_behavior", "")),
            }
        )

    return questions


def run_evaluation(
    questions_path: str | Path = "data/demo_questions.json",
    index_path: str | None = None,
) -> dict[str, Any]:
    from self_healing_rag.graph import ask_with_trace

    questions = load_eval_questions(questions_path)
    results = [
        {
            "question": item["question"],
            "expected_behavior": item["expected_behavior"],
            "result": ask_with_trace(item["question"], index_path=index_path),
        }
        for item in questions
    ]

    summary = summarize_results(results)
    ragas_scores = run_ragas_evaluation(results)
    if ragas_scores:
        summary["ragas"] = ragas_scores
    return summary


def run_ragas_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run Ragas evaluation on the records if the package is installed."""
    import sys
    import types
    # Mock langchain_community.chat_models.vertexai so ragas doesn't crash on import
    try:
        import langchain_community.chat_models.vertexai
    except ModuleNotFoundError:
        mock_module = types.ModuleType("langchain_community.chat_models.vertexai")
        mock_module.ChatVertexAI = None
        sys.modules["langchain_community.chat_models.vertexai"] = mock_module

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError:
        # Ragas not installed, return empty
        print("Ragas or datasets packages not installed. Skipping Ragas metrics.")
        return {}

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for r in records:
        result = r["result"]
        questions.append(r["question"])
        answers.append(result.get("final_answer", ""))
        
        docs = result.get("retrieved_docs", [])
        contexts.append([doc["content"] for doc in docs])
        
        gt = r.get("expected_behavior", "")
        ground_truths.append([gt] if gt else [""])

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
    }
    
    try:
        dataset = Dataset.from_dict(data)
    except Exception as exc:
        print(f"Failed to create datasets Dataset: {exc}")
        return {}

    try:
        import os
        from self_healing_rag.config import settings
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        use_ollama = (
            settings.llm_provider.lower().strip() == "ollama"
            or not os.environ.get("OPENAI_API_KEY")
        )
        
        if use_ollama:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
            
            ollama_chat = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            ollama_embed = OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )
            
            # Use LangChain wrappers if compatible with Ragas v0.2+
            try:
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                evaluator_llm = LangchainLLMWrapper(ollama_chat)
                evaluator_embed = LangchainEmbeddingsWrapper(ollama_embed)
            except ImportError:
                evaluator_llm = ollama_chat
                evaluator_embed = ollama_embed
                
            try:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=evaluator_llm,
                    embeddings=evaluator_embed,
                )
            except Exception as e:
                print(f"Ragas evaluation with Ollama failed: {e}. Falling back to default evaluate.")
                result = evaluate(dataset, metrics=metrics)
        else:
            result = evaluate(dataset, metrics=metrics)
            
        scores = {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevance": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
        }
        return scores
    except Exception as exc:
        print(f"Ragas evaluation execution failed: {exc}")
        return {
            "error": str(exc),
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }



def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "approval_rate": 0.0,
            "citation_rate": 0.0,
            "fallback_rate": 0.0,
            "average_attempts": 0.0,
            "results": [],
        }

    normalized = [_normalize_record(record) for record in records]
    return {
        "total": total,
        "approval_rate": _rate(item["approved"] for item in normalized),
        "citation_rate": _rate(item["has_sources"] for item in normalized),
        "fallback_rate": _rate(item["used_fallback"] for item in normalized),
        "average_attempts": mean(item["attempts"] for item in normalized),
        "results": normalized,
    }


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    result: RAGState = record["result"]
    critique = result.get("critique") or {}
    answer = result.get("final_answer", "")
    trace = result.get("trace", [])

    return {
        "question": record["question"],
        "expected_behavior": record.get("expected_behavior", ""),
        "approved": bool(critique.get("approved", False)),
        "retry_type": critique.get("retry_type", "unknown"),
        "attempts": result.get("attempt", 0),
        "has_sources": "Sources:" in answer,
        "used_fallback": any(event["step"] == "fallback_answer" for event in trace),
        "answer_preview": answer[:240],
    }


def _rate(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)
