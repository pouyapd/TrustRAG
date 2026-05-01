"""ChromaDB-backed vector store."""
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.logging_setup import get_logger
from src.rag.chunking import Chunk
from src.rag.providers import EmbeddingProvider

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk_id: str
    doc_id: str
    text: str
    source: str
    score: float


class VectorStore:
    """Persistent vector store over ChromaDB."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Returns count added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"doc_id": c.doc_id, "source": c.source, **c.metadata}
                for c in chunks
            ],
        )
        log.info("vectors_added", count=len(chunks))
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """Top-k cosine retrieval."""
        query_emb = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        results: list[RetrievalResult] = []
        if not result.get("ids") or not result["ids"][0]:
            return results

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        # ChromaDB returns cosine distance; similarity = 1 - distance
        distances = result["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", "unknown"),
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=float(1.0 - dist),
                )
            )
        return results

    def count(self) -> int:
        """Number of vectors stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete the collection — useful for tests."""
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
