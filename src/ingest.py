"""CLI to ingest a directory of documents into the vector store.

Usage:
    python -m src.ingest --path data/documents
"""
import argparse
from pathlib import Path

from src.config import settings
from src.logging_setup import get_logger, setup_logging
from src.rag.chunking import DocumentChunker, load_directory
from src.rag.providers import get_embeddings
from src.rag.vector_store import VectorStore


def main() -> None:
    setup_logging()
    log = get_logger("ingest")

    parser = argparse.ArgumentParser(description="Ingest documents into TrustRAG")
    parser.add_argument("--path", required=True, help="Directory of documents")
    parser.add_argument("--reset", action="store_true", help="Reset collection first")
    args = parser.parse_args()

    chunker = DocumentChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    store = VectorStore(get_embeddings())

    if args.reset:
        store.reset()
        log.info("collection_reset")

    docs = load_directory(Path(args.path))
    if not docs:
        log.warning("no_documents_found", path=args.path)
        return

    total = 0
    for text, doc_id, source in docs:
        chunks = chunker.chunk_text(text, doc_id=doc_id, source=source)
        total += store.add(chunks)

    log.info("ingest_complete", documents=len(docs), chunks=total, total_in_store=store.count())


if __name__ == "__main__":
    main()
