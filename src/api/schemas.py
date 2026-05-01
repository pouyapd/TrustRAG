"""Request/response schemas for the API."""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Inline document ingestion."""
    documents: list[dict] = Field(
        ...,
        description="List of {doc_id, source, text} objects",
        examples=[[{"doc_id": "faq_001", "source": "faq.md", "text": "Our refund policy..."}]],
    )
    reset: bool = Field(default=False, description="Reset collection before ingest")


class IngestResponse(BaseModel):
    chunks_added: int
    total_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    score_faithfulness: bool = True


class SourceRef(BaseModel):
    doc_id: str
    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    faithfulness_score: float | None
    latency_ms: float


class EvaluateRequest(BaseModel):
    """Inline evaluation request."""
    dataset: list[dict] = Field(
        ...,
        description="List of {question, answer, relevant_doc_ids}",
    )
    top_k: int = Field(default=4, ge=1, le=20)


class EvaluateResponse(BaseModel):
    summary: dict
    rows: list[dict]


class HealthResponse(BaseModel):
    status: str
    vectors_in_store: int
    llm_provider: str
