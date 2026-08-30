"""Loader interface and registry.

Every loader takes a path to raw data and returns `(questions, documents)` in
the unified schema. Two rules hold for all of them:

**Raw data is opened read-only and never written.** The pipeline treats the
downloaded corpus as immutable input; everything derived goes to a separate
output directory.

**Formats are parsed strictly.** A loader raises `DatasetFormatError` when the
input does not look like what it expects, rather than silently producing zero
records — a malformed parse that yields an empty dataset is far more expensive
to discover later than an immediate failure.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.data.schema import Document, QuestionRecord


class DatasetFormatError(ValueError):
    """Raised when raw data does not match the format a loader expects."""


@dataclass
class LoadResult:
    """What a loader produced."""

    questions: list[QuestionRecord]
    documents: list[Document]
    dataset: str
    #: Items skipped because they could not be represented, with reasons.
    skipped: dict[str, int]

    @property
    def counts(self) -> dict:
        answerable = sum(1 for q in self.questions if q.is_answerable)
        return {
            "questions": len(self.questions),
            "answerable": answerable,
            "unanswerable": len(self.questions) - answerable,
            "documents": len(self.documents),
            "skipped": dict(self.skipped),
        }


class DatasetLoader(ABC):
    """Converts one dataset's native format into the unified schema."""

    #: Short identifier used in question ids and the registry.
    name: str = ""
    #: SPDX id of the source licence, resolved through `data.licensing`.
    license_spdx: str = "UNKNOWN"
    #: Human-readable pointer to where the raw data comes from.
    source_url: str = ""

    def __init__(self, corpus_id: str | None = None) -> None:
        self.corpus_id = corpus_id or self.name

    @abstractmethod
    def load(self, path: Path, limit: int | None = None, split: str = "test") -> LoadResult:
        """Load raw data from `path`. Never writes to it."""

    # ---- helpers shared by loaders ----

    def make_question_id(self, item_id: str) -> str:
        """Namespaced question id, stable across runs."""
        return f"{self.name}:{item_id}"

    def make_doc_id(self, raw_id: str) -> str:
        """Namespaced document id, stable across runs."""
        return f"{self.name}:{raw_id}"

    @staticmethod
    def read_jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
        """Stream a JSONL file, transparently handling gzip.

        Large QA dumps ship gzipped; reading them without decompressing first
        keeps the raw directory untouched.
        """
        opener = _open_maybe_gzip(path)
        count = 0
        with opener as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetFormatError(
                        f"{path.name} line {line_number} is not valid JSON: {e}"
                    ) from e
                count += 1
                if limit is not None and count >= limit:
                    return

    @staticmethod
    def read_json(path: Path) -> object:
        """Read a whole JSON document, transparently handling gzip."""
        with _open_maybe_gzip(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise DatasetFormatError(f"{path.name} is not valid JSON: {e}") from e

    @staticmethod
    def require(condition: bool, message: str) -> None:
        """Assert a format expectation, raising DatasetFormatError when violated."""
        if not condition:
            raise DatasetFormatError(message)


def _open_maybe_gzip(path: Path):
    """Open a text file, decompressing transparently when it is gzipped."""
    if not path.exists():
        raise FileNotFoundError(f"raw dataset file not found: {path}")
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------

_REGISTRY: dict[str, type[DatasetLoader]] = {}


def register_loader(cls: type[DatasetLoader]) -> type[DatasetLoader]:
    """Class decorator adding a loader to the registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a name")
    _REGISTRY[cls.name] = cls
    return cls


def get_loader(name: str, corpus_id: str | None = None) -> DatasetLoader:
    """Instantiate a registered loader by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"unknown dataset loader {name!r}; available: {available}")
    return _REGISTRY[name](corpus_id=corpus_id)


def available_loaders() -> list[str]:
    """Names of all registered loaders."""
    return sorted(_REGISTRY)
