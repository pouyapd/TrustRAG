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


def _optional_int(value: object) -> int | None:
    """Coerce a stored metadata value to int, tolerating absence.

    Older collections were written without offsets, so a missing key must read
    as None rather than raising or defaulting to a misleading 0.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score.

    `start_char`/`end_char` are the chunk's half-open character range in its
    source document, carried through from ingestion. They are None only for
    vectors written before offsets were recorded.
    """
    chunk_id: str
    doc_id: str
    text: str
    source: str
    score: float
    start_char: int | None = None
    end_char: int | None = None


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
                    start_char=_optional_int(meta.get("start_char")),
                    end_char=_optional_int(meta.get("end_char")),
                )
            )
        return results

    def count(self) -> int:
        """Number of vectors stored."""
        return self.collection.count()

    def doc_chunk_counts(self) -> dict[str, int]:
        """How many chunks each document contributed to the collection.

        Evaluation needs this to build a correct denominator: chunk-level
        recall@k and the ideal ranking for nDCG both depend on how many chunks
        the relevant documents actually have. Returns an empty dict if the
        store cannot report metadata, so callers degrade to the weaker
        "among retrieved" variants rather than failing.
        """
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as e:  # pragma: no cover - store-specific failure
            log.warning("doc_chunk_counts_unavailable", error=str(e)[:80])
            return {}

        counts: dict[str, int] = {}
        for meta in result.get("metadatas") or []:
            doc_id = (meta or {}).get("doc_id", "unknown")
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def reset(self) -> None:
        """Delete the collection — useful for tests."""
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
