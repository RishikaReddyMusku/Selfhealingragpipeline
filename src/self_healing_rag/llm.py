import json
from urllib import request

from self_healing_rag.state import Critique, RetrievedDocument


VALID_RETRY_TYPES = {
    "accept",
    "rewrite_query",
    "retrieve_again",
    "regenerate",
    "clarify",
    "fallback",
}


def generate_grounded_answer(question: str, docs: list[RetrievedDocument]) -> tuple[str, str]:
    """Generate an answer with either an LLM or the deterministic local fallback."""
    if _use_llm():
        try:
            return _generate_with_llm(question, docs), "llm"
        except Exception:
            if not _llm_fallback_enabled():
                raise
            return _generate_fallback_answer(question, docs), "fallback"

    return _generate_fallback_answer(question, docs), "fallback"


def critique_grounded_answer(
    question: str,
    answer: str,
    docs: list[RetrievedDocument],
) -> tuple[Critique, str]:
    """Critique an answer with either an LLM or deterministic validation rules."""
    if _use_llm():
        try:
            return _critique_with_llm(question, answer, docs), "llm"
        except Exception:
            if not _llm_fallback_enabled():
                raise
            return _critique_fallback_answer(answer, docs), "fallback"

    return _critique_fallback_answer(answer, docs), "fallback"


def _generate_with_llm(question: str, docs: list[RetrievedDocument]) -> str:
    from self_healing_rag.config import settings

    context = _format_context(docs)
    prompt = (
        "You are a careful RAG assistant. Answer the question using only the provided "
        "context. If the context is insufficient, say so. End with a 'Sources:' line "
        "listing the source names you used.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )
    return _chat_with_configured_provider(prompt)


def _critique_with_llm(question: str, answer: str, docs: list[RetrievedDocument]) -> Critique:
    context = _format_context(docs)
    prompt = (
        "You are a strict RAG quality critic. Return only valid JSON with these keys: "
        "approved, reason, retry_type, feedback. retry_type must be one of accept, "
        "rewrite_query, retrieve_again, regenerate, clarify, fallback. Approve only if "
        "the answer is supported by the context and includes source citations.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{context}"
    )
    return _parse_critique(_chat_with_configured_provider(prompt))


def _chat_with_configured_provider(prompt: str) -> str:
    from self_healing_rag.config import settings

    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return _chat_with_ollama(
            prompt=prompt,
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "openai":
        return _chat_with_openai(prompt=prompt, model=settings.chat_model)

    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'openai'.")


def _chat_with_openai(prompt: str, model: str) -> str:
    from langchain_openai import ChatOpenAI

    response = ChatOpenAI(model=model, temperature=0).invoke(prompt)
    return str(response.content).strip()


def _chat_with_ollama(prompt: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    return str(data["message"]["content"]).strip()


def _generate_fallback_answer(question: str, docs: list[RetrievedDocument]) -> str:
    if not docs:
        return "I do not have retrieved context to answer this question.\n\nSources: none"

    context = "\n".join(f"- {doc['content']}" for doc in docs)
    sources = ", ".join(sorted({doc["source"] for doc in docs}))
    return (
        f"Based on the retrieved context, the answer to '{question}' is supported by "
        "the following evidence. The source material says:\n\n"
        f"{context}\n\n"
        f"Sources: {sources}"
    )


def _critique_fallback_answer(answer: str, docs: list[RetrievedDocument]) -> Critique:
    if not docs:
        return {
            "approved": False,
            "reason": "The answer has no supporting retrieved context.",
            "retry_type": "retrieve_again",
            "feedback": "Retrieve relevant documents before answering.",
        }

    if "Sources:" not in answer:
        return {
            "approved": False,
            "reason": "The answer does not include source citations.",
            "retry_type": "regenerate",
            "feedback": "Regenerate the answer with explicit source citations.",
        }

    return {
        "approved": True,
        "reason": "The answer is grounded and includes sources.",
        "retry_type": "accept",
        "feedback": "No changes needed.",
    }


def _parse_critique(raw_response: str) -> Critique:
    try:
        payload = json.loads(_extract_json_object(raw_response))
    except json.JSONDecodeError as exc:
        raise ValueError("Critic did not return valid structured JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Critic JSON must be an object.")

    required_keys = {"approved", "reason", "retry_type", "feedback"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Critic JSON is missing required keys: {sorted(missing_keys)}")

    if not isinstance(payload["approved"], bool):
        raise ValueError("Critic field 'approved' must be a boolean.")

    if payload["retry_type"] not in VALID_RETRY_TYPES:
        raise ValueError("Critic field 'retry_type' is not a valid route.")

    for key in ("reason", "feedback"):
        if not isinstance(payload[key], str):
            raise ValueError(f"Critic field '{key}' must be a string.")

    return {
        "approved": payload["approved"],
        "reason": payload["reason"],
        "retry_type": payload["retry_type"],
        "feedback": payload["feedback"],
    }


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text

    return text[start : end + 1]


def _format_context(docs: list[RetrievedDocument]) -> str:
    if not docs:
        return "No retrieved context."

    return "\n\n".join(
        f"Source: {doc['source']}\nScore: {doc['score']:.3f}\nContent: {doc['content']}"
        for doc in docs
    )


def _use_llm() -> bool:
    try:
        from self_healing_rag.config import settings
    except ModuleNotFoundError:
        return False

    return settings.use_llm


def _llm_fallback_enabled() -> bool:
    try:
        from self_healing_rag.config import settings
    except ModuleNotFoundError:
        return True

    return settings.llm_fallback_enabled
