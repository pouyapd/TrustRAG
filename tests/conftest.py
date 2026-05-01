"""Shared pytest fixtures."""
import gc
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Set test config BEFORE importing the app
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def tmp_chroma_dir(monkeypatch) -> Path:
    """Isolated ChromaDB persist directory per test.

    ChromaDB caches connections at the process level, so we also need to clear
    its internal client registry between tests to avoid stale handles pointing
    at deleted directories.
    """
    tmpdir = tempfile.mkdtemp(prefix="trustrag_test_")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tmpdir)

    # Reload settings to pick up the override
    from src import config
    config.settings = config.Settings()

    # Reset cached singletons in our app
    from src.api.dependencies import get_pipeline, get_vector_store
    get_pipeline.cache_clear()
    get_vector_store.cache_clear()

    # Clear ChromaDB's internal client cache
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system.clear()
    except Exception:
        pass

    yield Path(tmpdir)

    # Force GC so SQLite handles release before we rmtree
    gc.collect()
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system.clear()
    except Exception:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_pipeline(tmp_chroma_dir):
    """RAG pipeline wired with a mock LLM and deterministic hash embedder.

    Hash embedder is used (not sentence-transformers) so tests are:
    - fast (no model download)
    - offline (no network needed)
    - deterministic (reproducible across machines)
    """
    from src.rag.mock_llm import MockExtractiveLLM
    from src.rag.pipeline import RAGPipeline
    from src.rag.providers import HashEmbeddings
    from src.rag.vector_store import VectorStore

    store = VectorStore(HashEmbeddings())
    return RAGPipeline(vector_store=store, llm=MockExtractiveLLM())


@pytest.fixture
def populated_pipeline(mock_pipeline):
    """Pipeline pre-loaded with a small document set."""
    from src.rag.chunking import DocumentChunker

    docs = [
        ("doc_a", "faq_a.md", "The capital of France is Paris. France is in Europe."),
        ("doc_b", "faq_b.md", "Python is a programming language created by Guido van Rossum."),
        ("doc_c", "faq_c.md", "The refund window for annual plans is 30 days."),
    ]
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=10)
    for doc_id, source, text in docs:
        chunks = chunker.chunk_text(text, doc_id=doc_id, source=source)
        mock_pipeline.vector_store.add(chunks)
    return mock_pipeline
