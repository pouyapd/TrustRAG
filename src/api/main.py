"""FastAPI application entrypoint."""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import __version__
from src.api.dependencies import get_pipeline, get_vector_store
from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceRef,
)
from src.config import settings
from src.evaluation.runner import run_evaluation_inline
from src.logging_setup import get_logger, setup_logging
from src.monitoring.metrics import (
    FAILURE_MODES,
    FAITHFULNESS_SCORE,
    INGEST_TOTAL,
    QUERIES_TOTAL,
    QUERY_LATENCY,
    VECTORS_IN_STORE,
)
from src.rag.chunking import Chunk, DocumentChunker
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    setup_logging()
    log = get_logger("api")
    log.info("api_starting", version=__version__, llm_provider=settings.llm_provider)
    # Warm up the vector store so the first request is fast
    store = get_vector_store()
    VECTORS_IN_STORE.set(store.count())
    log.info("api_ready", vectors_in_store=store.count())
    yield
    log.info("api_shutting_down")


app = FastAPI(
    title="TrustRAG",
    description="Production RAG with systematic evaluation and failure mode analysis.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    """Attach a request_id and log timing for every request."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    log = get_logger("api.middleware")

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:
        log.error("request_failed", error=str(e))
        raise
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info("request_complete", duration_ms=round(duration_ms, 1))
        structlog.contextvars.clear_contextvars()

    response.headers["x-request-id"] = request_id
    return response


# ---------- Endpoints ----------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(store: VectorStore = Depends(get_vector_store)) -> HealthResponse:
    """Liveness + basic state."""
    return HealthResponse(
        status="ok",
        vectors_in_store=store.count(),
        llm_provider=settings.llm_provider,
    )


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    """Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(
    req: IngestRequest,
    store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    """Chunk and embed inline documents."""
    log = get_logger("api.ingest")

    if req.reset:
        store.reset()
        log.info("collection_reset")

    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    all_chunks: list[Chunk] = []
    for d in req.documents:
        if not all(k in d for k in ("doc_id", "source", "text")):
            raise HTTPException(400, "each document needs doc_id, source, text")
        all_chunks.extend(
            chunker.chunk_text(d["text"], doc_id=d["doc_id"], source=d["source"])
        )

    added = store.add(all_chunks)
    INGEST_TOTAL.inc(added)
    VECTORS_IN_STORE.set(store.count())

    return IngestResponse(chunks_added=added, total_in_store=store.count())


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(
    req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Run a RAG query."""
    log = get_logger("api.query")

    try:
        with QUERY_LATENCY.time():
            response = pipeline.query(
                req.question,
                top_k=req.top_k,
                score_faithfulness=req.score_faithfulness,
            )
    except Exception as e:
        QUERIES_TOTAL.labels(status="error").inc()
        log.exception("query_failed")
        raise HTTPException(500, f"query failed: {e}") from e

    QUERIES_TOTAL.labels(status="ok").inc()
    if response.faithfulness_score is not None:
        FAITHFULNESS_SCORE.observe(response.faithfulness_score)

    sources = [
        SourceRef(
            doc_id=s.doc_id,
            source=s.source,
            score=round(s.score, 4),
            snippet=s.text[:200],
        )
        for s in response.sources
    ]
    return QueryResponse(
        answer=response.answer,
        sources=sources,
        faithfulness_score=response.faithfulness_score,
        latency_ms=round(response.latency_ms, 1),
    )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
def evaluate(
    req: EvaluateRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvaluateResponse:
    """Run an inline evaluation against a Q/A dataset."""
    log = get_logger("api.evaluate")
    log.info("inline_eval_starting", dataset_size=len(req.dataset))

    summary, rows = run_evaluation_inline(req.dataset, pipeline=pipeline, top_k=req.top_k)

    for mode, count in summary.get("failure_modes", {}).items():
        FAILURE_MODES.labels(mode=mode).inc(count)

    return EvaluateResponse(summary=summary, rows=rows)
