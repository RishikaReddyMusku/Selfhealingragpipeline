from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from self_healing_rag.config import settings
from self_healing_rag.graph import ask_with_trace
from self_healing_rag.index import build_indexes


WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="Self-Healing RAG API",
    description="API for document ingestion and critic-driven RAG answers.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


class IngestRequest(BaseModel):
    input_path: str = Field(..., description="File or directory containing source documents.")
    output_path: str = Field(
        settings.local_index_path,
        description="Path where the local chunk index should be written.",
    )
    build_vector: bool = Field(
        settings.retrieval_backend == "vector",
        description="Whether to also build/update the Chroma vector index.",
    )


class IngestResponse(BaseModel):
    chunks_indexed: int
    output_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    index_path: str = Field(
        settings.local_index_path,
        description="Path to the local chunk index used for retrieval.",
    )


class RetrievedDocumentResponse(BaseModel):
    content: str
    source: str
    score: float


class CritiqueResponse(BaseModel):
    approved: bool
    reason: str
    retry_type: str
    feedback: str


class TraceEventResponse(BaseModel):
    step: str
    status: str
    detail: str
    at: str | None = None
    elapsed_ms: float | None = None


class ObservabilityMetricsResponse(BaseModel):
    total_elapsed_ms: float
    total_steps: int
    retrieval_attempts: int
    generation_attempts: int


class AskResponse(BaseModel):
    answer: str
    attempts: int
    sources: list[str]
    needs_clarification: bool = False
    retrieved_docs: list[RetrievedDocumentResponse]
    critique: CritiqueResponse | None
    trace: list[TraceEventResponse]
    metrics: ObservabilityMetricsResponse


class ConfigUpdateRequest(BaseModel):
    use_llm: bool | None = None
    llm_fallback_enabled: bool | None = None
    llm_provider: str | None = None
    retrieval_backend: str | None = None
    rerank_backend: str | None = None
    chat_model: str | None = None
    ollama_model: str | None = None
    max_attempts: int | None = None
    max_retrieval_retries: int | None = None
    max_generation_retries: int | None = None
    min_context_score: float | None = None
    cohere_api_key: str | None = None


class EvaluateRequest(BaseModel):
    questions_path: str = Field("data/demo_questions.json")
    index_path: str | None = None


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        chunks = build_indexes(
            input_path=request.input_path,
            output_path=request.output_path,
            build_vector=request.build_vector,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestResponse(
        chunks_indexed=len(chunks),
        output_path=request.output_path,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    result = ask_with_trace(request.question, index_path=request.index_path)
    retrieved_docs = result.get("retrieved_docs", [])
    sources = sorted({doc["source"] for doc in retrieved_docs})

    return AskResponse(
        answer=result["final_answer"],
        attempts=result.get("attempt", 0),
        sources=sources,
        needs_clarification=bool(result.get("needs_clarification", False)),
        retrieved_docs=retrieved_docs,
        critique=result.get("critique"),
        trace=result.get("trace", []),
        metrics=result["metrics"],
    )


@app.get("/config")
def get_config() -> dict:
    return {
        "use_llm": settings.use_llm,
        "llm_fallback_enabled": settings.llm_fallback_enabled,
        "llm_provider": settings.llm_provider,
        "retrieval_backend": settings.retrieval_backend,
        "rerank_backend": settings.rerank_backend,
        "chat_model": settings.chat_model,
        "ollama_model": settings.ollama_model,
        "max_attempts": settings.max_attempts,
        "max_retrieval_retries": settings.max_retrieval_retries,
        "max_generation_retries": settings.max_generation_retries,
        "min_context_score": settings.min_context_score,
        "cohere_api_key": settings.cohere_api_key,
    }


@app.post("/config")
def update_config(request: ConfigUpdateRequest) -> dict[str, str]:
    for key, value in request.dict(exclude_unset=True).items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return {"status": "success"}


@app.post("/evaluate")
def trigger_evaluation(request: EvaluateRequest):
    from self_healing_rag.evaluation import run_evaluation
    try:
        summary = run_evaluation(
            questions_path=request.questions_path,
            index_path=request.index_path,
        )
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
