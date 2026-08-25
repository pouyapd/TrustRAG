"""Inference records — the boundary between running a system and judging it.

Before this module, `runner._run_rows` generated an answer and classified it in
the same loop, so changing a threshold meant re-running every query through the
LLM. Threshold sensitivity analysis was therefore as expensive as a full
evaluation, which in practice meant it never happened.

An `InferenceRecord` captures everything a scorer needs — the question, the
reference, the retrieved chunks with their ranks and scores, the generated
answer, the faithfulness score — so scoring and classification become pure
functions over stored data. `scripts/reclassify.py` re-scores a completed run
with different thresholds and makes zero model calls.

Records are persisted as JSONL next to the report, one object per line.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECORDS_FILENAME = "inference.jsonl"
RECORD_SCHEMA_VERSION = "1.0"


@dataclass
class RetrievedChunk:
    """One retrieved chunk, with its position in the ranking."""

    rank: int
    chunk_id: str
    doc_id: str
    source: str
    score: float
    text: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RetrievedChunk:
        return cls(
            rank=int(data["rank"]),
            chunk_id=str(data.get("chunk_id", "")),
            doc_id=str(data.get("doc_id", "unknown")),
            source=str(data.get("source", "unknown")),
            score=float(data.get("score", 0.0)),
            text=str(data.get("text", "")),
        )


@dataclass
class InferenceRecord:
    """Everything produced by running one question through the pipeline.

    Deliberately contains no metric and no failure label: those are derived,
    and deriving them again from this record must always be possible.
    """

    index: int
    question: str
    reference_answer: str
    relevant_doc_ids: list[str]
    predicted_answer: str
    retrieved: list[RetrievedChunk]
    faithfulness: float | None
    latency_ms: float
    top_k: int
    #: Total number of chunks in the corpus belonging to the relevant documents.
    #: Needed for a correct chunk-level recall denominator and for nDCG's ideal
    #: ranking. None when the vector store could not report it.
    n_relevant_chunks: int | None = None
    #: Total chunks in the collection, for context.
    corpus_chunk_count: int | None = None
    schema_version: str = RECORD_SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """Chunk-level doc ids in rank order — repeats when a doc wins twice."""
        return [c.doc_id for c in self.retrieved]

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.retrieved]

    @property
    def is_answerable(self) -> bool:
        return bool(self.relevant_doc_ids)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["retrieved"] = [c.as_dict() for c in self.retrieved]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> InferenceRecord:
        return cls(
            index=int(data.get("index", 0)),
            question=str(data["question"]),
            reference_answer=str(data.get("reference_answer", "")),
            relevant_doc_ids=list(data.get("relevant_doc_ids", [])),
            predicted_answer=str(data.get("predicted_answer", "")),
            retrieved=[RetrievedChunk.from_dict(c) for c in data.get("retrieved", [])],
            faithfulness=(
                None if data.get("faithfulness") is None else float(data["faithfulness"])
            ),
            latency_ms=float(data.get("latency_ms", 0.0)),
            top_k=int(data.get("top_k", 0)),
            n_relevant_chunks=(
                None
                if data.get("n_relevant_chunks") is None
                else int(data["n_relevant_chunks"])
            ),
            corpus_chunk_count=(
                None
                if data.get("corpus_chunk_count") is None
                else int(data["corpus_chunk_count"])
            ),
            schema_version=str(data.get("schema_version", RECORD_SCHEMA_VERSION)),
            metadata=dict(data.get("metadata", {})),
        )


def write_records(records: list[InferenceRecord], path: Path) -> Path:
    """Persist inference records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_records(path: Path) -> list[InferenceRecord]:
    """Load inference records from JSONL."""
    records: list[InferenceRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(InferenceRecord.from_dict(json.loads(line)))
    return records
