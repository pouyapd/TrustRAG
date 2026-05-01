"""API endpoint tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_chroma_dir, monkeypatch):
    """TestClient with mock LLM and deterministic embeddings injected."""
    from src.api import main as api_main
    from src.api.dependencies import get_pipeline, get_vector_store
    from src.rag.mock_llm import MockExtractiveLLM
    from src.rag.pipeline import RAGPipeline
    from src.rag.providers import HashEmbeddings
    from src.rag.vector_store import VectorStore

    store = VectorStore(HashEmbeddings())
    pipeline = RAGPipeline(vector_store=store, llm=MockExtractiveLLM())

    api_main.app.dependency_overrides[get_vector_store] = lambda: store
    api_main.app.dependency_overrides[get_pipeline] = lambda: pipeline

    with TestClient(api_main.app) as c:
        yield c

    api_main.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "vectors_in_store" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"trustrag_queries_total" in r.content


def test_ingest_and_query(client):
    ingest_payload = {
        "documents": [
            {
                "doc_id": "policy",
                "source": "policy.md",
                "text": "Refunds are available within 30 days of purchase. Contact billing@example.com.",
            }
        ],
        "reset": True,
    }
    r = client.post("/ingest", json=ingest_payload)
    assert r.status_code == 200, r.text
    assert r.json()["chunks_added"] >= 1

    r = client.post(
        "/query",
        json={"question": "How long is the refund window?", "top_k": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert len(body["sources"]) > 0
    assert body["latency_ms"] > 0


def test_query_validation(client):
    r = client.post("/query", json={"question": "", "top_k": 4})
    assert r.status_code == 422  # min_length violation


def test_evaluate_inline(client):
    # First ingest some docs
    client.post(
        "/ingest",
        json={
            "documents": [
                {"doc_id": "d1", "source": "a.md", "text": "Paris is the capital of France."},
                {"doc_id": "d2", "source": "b.md", "text": "Rome is the capital of Italy."},
            ],
            "reset": True,
        },
    )

    eval_payload = {
        "dataset": [
            {
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "relevant_doc_ids": ["d1"],
            },
            {
                "question": "What is the capital of Italy?",
                "answer": "Rome.",
                "relevant_doc_ids": ["d2"],
            },
        ],
        "top_k": 2,
    }
    r = client.post("/evaluate", json=eval_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["total"] == 2
    assert len(body["rows"]) == 2
