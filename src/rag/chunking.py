"""Document loading and chunking.

Uses tiktoken for token-accurate chunking when available, with a graceful
word-based fallback for offline environments and CI without network access.

**Every chunk carries exact character offsets into its source document.**
Chunk text is produced by slicing the original string over a half-open range
`[start_char, end_char)`, never by decoding tokens back into text. That
distinction matters: `_WordTokenizer.decode` is `" ".join(tokens)`, which
collapses the original whitespace, so a decoded chunk is not a substring of the
document and no offset could ever be recovered for it. Offsets are therefore
computed from token spans measured against the source at encode time.

Offsets are what lets evaluation ask whether a retrieved chunk overlaps a
labelled supporting span, without re-deriving positions by searching for the
chunk text — a search that is ambiguous whenever a document repeats itself.
"""
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.logging_setup import get_logger

log = get_logger(__name__)

#: Matches the token boundaries of `str.split()`: maximal runs of non-whitespace.
_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    """A chunk of text from a document with metadata.

    `metadata` always carries `start_char` and `end_char`, the half-open
    character range this chunk occupies in the source document, so that
    `document_text[start_char:end_char] == chunk.text`.
    """
    text: str
    doc_id: str
    chunk_id: str
    source: str
    metadata: dict

    @property
    def start_char(self) -> int | None:
        """Inclusive start offset in the source document."""
        return self.metadata.get("start_char")

    @property
    def end_char(self) -> int | None:
        """Exclusive end offset in the source document."""
        return self.metadata.get("end_char")


class _WordTokenizer:
    """Whitespace tokenizer used as a fallback when tiktoken is unavailable.

    Token counts approximate but consistent — sufficient for chunk sizing.

    `decode` is lossy by construction, so it is never used to build chunk text.
    `char_spans` is the offset-preserving path, and it reproduces exactly the
    token boundaries `encode` would produce.
    """

    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)

    @staticmethod
    def char_spans(text: str) -> list[tuple[int, int]]:
        """Character range of every token, in the original text."""
        return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _tiktoken_char_spans(text: str, encoder) -> list[tuple[int, int]]:
    """Character range of every tiktoken token, in the original text.

    tiktoken is a byte-level BPE: decoding is `b"".join(token_bytes)`, so
    cumulative token byte lengths give exact byte offsets into the UTF-8
    encoding of the source. Those are converted to character offsets through a
    table of per-character byte positions.

    A token boundary can fall inside a multi-byte character (a single character
    split across two tokens). Such a boundary has no exact character offset, so
    it is snapped outwards — start down, end up — to the enclosing character.
    That keeps every span a valid slice of the source and can only widen a
    chunk by the character it straddles, never misalign it.
    """
    tokens = encoder.encode(text)

    # Byte offset at which each character starts, plus a total-length sentinel.
    char_starts: list[int] = []
    position = 0
    for character in text:
        char_starts.append(position)
        position += len(character.encode("utf-8"))
    char_starts.append(position)

    spans: list[tuple[int, int]] = []
    byte_cursor = 0
    for token in tokens:
        token_bytes = encoder.decode_single_token_bytes(token)
        start_byte = byte_cursor
        byte_cursor += len(token_bytes)
        # Snap start down and end up to enclosing character boundaries.
        start_char = max(0, bisect_right(char_starts, start_byte) - 1)
        end_char = bisect_left(char_starts, byte_cursor)
        if end_char <= start_char:
            end_char = start_char + 1
        spans.append((start_char, min(end_char, len(text))))
    return spans


def token_char_spans(text: str, encoder) -> list[tuple[int, int]]:
    """Character range of every token produced by `encoder` for `text`.

    Dispatches on capability rather than type, so an alternative tokenizer can
    participate by exposing `char_spans`. A tokenizer that can offer neither
    exact spans nor token bytes raises: an approximate mapping would be worse
    than no mapping, because it would silently misalign gold evidence.
    """
    char_spans = getattr(encoder, "char_spans", None)
    if callable(char_spans):
        return char_spans(text)
    if callable(getattr(encoder, "decode_single_token_bytes", None)):
        return _tiktoken_char_spans(text, encoder)
    raise TypeError(
        f"{type(encoder).__name__} cannot provide exact character offsets: it "
        "exposes neither char_spans() nor decode_single_token_bytes()"
    )


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
        """Split text into overlapping token-bounded chunks.

        Chunk boundaries are chosen on token indices exactly as before; what
        changed is how the chunk's text is materialised. Rather than decoding
        the token slice back into a string, the token spans give a character
        range and the chunk text is that slice of the original document. The
        invariant `text[chunk.start_char:chunk.end_char] == chunk.text` holds
        for every chunk and every tokenizer.
        """
        if not text.strip():
            return []

        spans = token_char_spans(text, self.encoder)
        if not spans:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(spans):
            end = min(start + self.chunk_size, len(spans))
            # Half-open range: first token's start through last token's end.
            start_char = spans[start][0]
            end_char = spans[end - 1][1]
            chunk_text = text[start_char:end_char]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{idx}",
                    source=source,
                    metadata={
                        "chunk_index": idx,
                        "token_count": end - start,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )

            idx += 1
            if end == len(spans):
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
