"""Document loading and chunking.

Uses tiktoken for token-accurate chunking when available, with a graceful
word-based fallback for offline environments and CI without network access.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """A chunk of text from a document with metadata."""
    text: str
    doc_id: str
    chunk_id: str
    source: str
    metadata: dict


class _WordTokenizer:
    """Whitespace tokenizer used as a fallback when tiktoken is unavailable.

    Token counts approximate but consistent — sufficient for chunk sizing.
    """

    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def _load_encoder():
    """Try tiktoken first, fall back to word tokenizer if network blocks BPE download."""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        encoder.encode("test")  # verify it works (may need network for BPE files)
        return encoder
    except Exception as e:
        log.info("tiktoken_unavailable_using_word_fallback", reason=str(e)[:80])
        return _WordTokenizer()


class DocumentChunker:
    """Token-aware document chunker with word-based fallback."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = _load_encoder()

    def chunk_text(self, text: str, doc_id: str, source: str) -> list[Chunk]:
        """Split text into overlapping token-bounded chunks."""
        if not text.strip():
            return []

        tokens = self.encoder.encode(text)
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{idx}",
                    source=source,
                    metadata={"chunk_index": idx, "token_count": len(chunk_tokens)},
                )
            )

            idx += 1
            if end == len(tokens):
                break
            start = end - self.chunk_overlap

        log.info("chunked_document", doc_id=doc_id, num_chunks=len(chunks))
        return chunks


def load_document(path: Path) -> tuple[str, str]:
    """Load a document from disk. Returns (text, source_name)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return text, path.name


def load_directory(directory: Path) -> list[tuple[str, str, str]]:
    """Load all supported docs from a directory. Returns list of (text, doc_id, source)."""
    if not directory.exists():
        log.warning("directory_not_found", path=str(directory))
        return []

    results: list[tuple[str, str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            text, source = load_document(path)
            doc_id = path.stem
            results.append((text, doc_id, source))
        except Exception as e:
            log.error("doc_load_failed", path=str(path), error=str(e))
    return results
