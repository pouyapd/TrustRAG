"""FastAPI dependency providers — singleton pipeline."""
from functools import lru_cache

from src.rag.pipeline import RAGPipeline
from src.rag.providers import get_embeddings
from src.rag.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Singleton vector store."""
    return VectorStore(get_embeddings())


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Singleton pipeline reusing the shared store."""
    return RAGPipeline(vector_store=get_vector_store())
