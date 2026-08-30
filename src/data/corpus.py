"""The bridge from dataset documents into the running RAG pipeline (W5).

Before this module the dataset layer and the pipeline could not meet. Ingestion
went through `load_directory()`, which reads files off disk and derives a
document id from the filename, so a `Document` carrying a namespaced id like
`nq:page123` had no way into the vector store. Every `relevant_doc_ids`
comparison would then fail and every question would score as a retrieval
failure for purely plumbing reasons.

`build_corpus` closes that gap: it chunks `Document` objects directly,
preserving both the namespaced document id and the character offsets W1
attaches, and reports what it indexed so an experiment can record it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.data.schema import Document
from src.logging_setup import get_logger
from src.rag.chunking import Chunk, DocumentChunker

log = get_logger(__name__)


@dataclass
class CorpusStats:
    """What was actually indexed, for the experiment record."""

    n_documents: int
    n_chunks: int
    chunk_size: int
    chunk_overlap: int
    total_characters: int
    chunks_per_document: dict[str, int] = field(default_factory=dict)
    #: Chunks whose offsets do not slice back to their source text. Must be 0;
    #: a non-zero value means evidence alignment cannot be trusted.
    offset_mismatches: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def chunk_documents(
    documents: list[Document],
    chunker: DocumentChunker,
    verify_offsets: bool = True,
) -> tuple[list[Chunk], CorpusStats]:
    """Chunk dataset documents, preserving ids and character offsets.

    With `verify_offsets` every chunk is checked against its source document:
    `text[start_char:end_char]` must equal the chunk text. That check is cheap
    and it is the guarantee the entire evidence-alignment layer rests on, so it
    runs by default rather than being left to a test.
    """
    all_chunks: list[Chunk] = []
    per_document: dict[str, int] = {}
    mismatches = 0

    for document in documents:
        chunks = chunker.chunk_text(document.text, doc_id=document.doc_id, source=document.source)
        if verify_offsets:
            for chunk in chunks:
                start, end = chunk.start_char, chunk.end_char
                if start is None or end is None or document.text[start:end] != chunk.text:
                    mismatches += 1
        per_document[document.doc_id] = len(chunks)
        all_chunks.extend(chunks)

    stats = CorpusStats(
        n_documents=len(documents),
        n_chunks=len(all_chunks),
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.chunk_overlap,
        total_characters=sum(len(d.text) for d in documents),
        chunks_per_document=per_document,
        offset_mismatches=mismatches,
    )
    log.info(
        "corpus_chunked",
        documents=stats.n_documents,
        chunks=stats.n_chunks,
        offset_mismatches=mismatches,
    )
    return all_chunks, stats


def build_corpus(
    documents: list[Document],
    store,
    chunker: DocumentChunker,
    reset: bool = True,
    batch_size: int = 256,
) -> CorpusStats:
    """Chunk and index dataset documents into a vector store.

    Raises when any chunk's offsets fail to reproduce its text: indexing a
    corpus whose offsets are wrong would silently invalidate every
    evidence-level metric computed from it, and failing loudly at build time is
    far cheaper than discovering it in the results.
    """
    chunks, stats = chunk_documents(documents, chunker)
    if stats.offset_mismatches:
        raise ValueError(
            f"{stats.offset_mismatches} chunk(s) do not slice back to their source text; "
            "evidence alignment would be unreliable, refusing to index this corpus"
        )

    if reset:
        store.reset()

    # Batched so a large corpus does not build one enormous embedding call.
    for start in range(0, len(chunks), batch_size):
        store.add(chunks[start : start + batch_size])

    log.info("corpus_indexed", chunks=stats.n_chunks, in_store=store.count())
    return stats
