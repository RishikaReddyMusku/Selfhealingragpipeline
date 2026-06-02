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


class AskResponse(BaseModel):
    answer: str
    attempts: int
    sources: list[str]
    retrieved_docs: list[RetrievedDocumentResponse]
    critique: CritiqueResponse | None
    trace: list[TraceEventResponse]


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
        retrieved_docs=retrieved_docs,
        critique=result.get("critique"),
        trace=result.get("trace", []),
    )
