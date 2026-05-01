"""Application configuration loaded from environment variables."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized config. Override via environment variables or .env file."""

    # LLM
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "trustrag_docs"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 4

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
