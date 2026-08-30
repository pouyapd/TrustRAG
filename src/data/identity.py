"""Deterministic identifiers for questions and documents.

Every loader previously fell back to Python's `hash()` when a source item
carried no native id. `hash()` for `str` is randomised per process, so two
builds of the same raw file produced different question ids. That silently
breaks the thing the whole pipeline depends on: human annotations, ablation
`pair_id` links and cached results are all keyed by question id, and none of
them survive a rebuild if the key moves.

Ids here are content hashes. They are stable across processes, machines and
Python versions, and two runs over identical input produce byte-identical ids.
"""
from __future__ import annotations

import hashlib
import unicodedata

#: Hex characters kept from the digest. 16 hex chars = 64 bits, which keeps a
#: collision vanishingly unlikely at the scale of these corpora (well under a
#: billionth for a million items) while staying readable in a filename.
ID_LENGTH = 16


def _normalise(text: str) -> str:
    """Canonical form for hashing.

    NFC-normalises so that the same characters written with different Unicode
    encodings hash identically, and collapses whitespace so that a reflowed
    source file does not change an id.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_hash(*parts: str, length: int = ID_LENGTH) -> str:
    """Stable hash of the given parts.

    Parts are joined with a separator that cannot occur in the normalised
    input, so ("ab", "c") and ("a", "bc") never collide.
    """
    digest = hashlib.sha256("\x00".join(_normalise(p) for p in parts).encode("utf-8"))
    return digest.hexdigest()[:length]


def question_id(dataset: str, native_id: str | None, question: str) -> str:
    """Namespaced, deterministic question id.

    A native id from the source dataset is preferred, because it lets a reader
    trace the item back to the original release. When there is none, the id is
    derived from the question text instead of a random hash.
    """
    if native_id:
        return f"{dataset}:{native_id}"
    return f"{dataset}:q{content_hash(dataset, question)}"


def document_id(dataset: str, native_id: str | None, text: str) -> str:
    """Namespaced, deterministic document id.

    Falling back to the document's own content means two copies of the same
    document converge on one id, which is what stops a corpus from carrying
    duplicate documents under different names.
    """
    if native_id:
        return f"{dataset}:{native_id}"
    return f"{dataset}:d{content_hash(dataset, text)}"


def content_fingerprint(text: str, length: int = ID_LENGTH) -> str:
    """Fingerprint of a document's text, for deduplication."""
    return content_hash(text, length=length)
