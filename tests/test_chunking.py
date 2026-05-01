"""Tests for document chunking."""
from src.rag.chunking import DocumentChunker


def test_chunks_overlap():
    chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
    text = "word " * 200
    chunks = chunker.chunk_text(text, doc_id="doc_a", source="test.txt")
    assert len(chunks) > 1
    # Overlap should produce shared tokens between consecutive chunks
    assert all(c.doc_id == "doc_a" for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)


def test_unique_chunk_ids():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_text("hello world. " * 100, doc_id="d1", source="s.md")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_short_text_single_chunk():
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_text("short text", doc_id="d", source="s")
    assert len(chunks) == 1
    assert "short text" in chunks[0].text


def test_empty_text():
    chunker = DocumentChunker()
    chunks = chunker.chunk_text("", doc_id="d", source="s")
    assert chunks == []
