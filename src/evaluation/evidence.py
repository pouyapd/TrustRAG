"""Evidence-aware retrieval measurement (W2).

Document-level retrieval metrics answer "did we retrieve the right document?".
For a corpus of whole scientific papers that question is almost free: a paper
is tens of thousands of characters, so any chunk of it counts as a hit and the
metric saturates. What matters for a RAG system is narrower — was the specific
passage supporting the answer put in front of the generator?

This module answers that using the exact character offsets W1 attached to
chunks. A gold span and a retrieved chunk are both half-open character ranges
in the same document, so overlap is arithmetic. Nothing is located by
searching for text, so a document that repeats itself cannot fool alignment.

The output feeds two things: evidence-level retrieval metrics, and a failure
attribution hierarchy that refuses to blame the generator for evidence it was
never given.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

#: A gold span must share at least this many characters with a chunk before the
#: chunk counts as carrying it. One character is deliberately permissive; the
#: value is a declared experimental parameter, reported with any metric derived
#: from it.
DEFAULT_MIN_OVERLAP_CHARS = 1


class EvidenceMode(StrEnum):
    """How the gold spans combine to justify an answer.

    Mirrors `data.schema.EvidenceMode` by value, but is defined here so the
    evaluation layer does not depend on the dataset layer.
    """

    ANY_SUFFICIENT = "any_sufficient"
    ALL_REQUIRED = "all_required"


class EvidenceStatus(StrEnum):
    """How much of the required evidence reached the generator."""

    #: Everything the question needs was retrieved.
    COMPLETE = "complete"
    #: Some required evidence was retrieved, but not all. Only reachable under
    #: ALL_REQUIRED: for a multi-hop question this is a genuine retrieval
    #: failure even though the retriever looks partly right.
    PARTIAL = "partial"
    #: No gold evidence was retrieved at all.
    NONE = "none"
    #: The question has no gold evidence, so there is nothing to retrieve and
    #: no evidence judgement to make.
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GoldSpan:
    """A labelled supporting span, as a half-open character range."""

    doc_id: str
    start_char: int
    end_char: int

    def overlap(self, doc_id: str, start: int, end: int) -> int:
        """Characters shared with another range in the same document."""
        if doc_id != self.doc_id:
            return 0
        return max(0, min(self.end_char, end) - max(self.start_char, start))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedSpan:
    """A retrieved chunk reduced to what alignment needs."""

    rank: int
    doc_id: str
    start_char: int | None
    end_char: int | None

    @property
    def has_offsets(self) -> bool:
        return self.start_char is not None and self.end_char is not None


@dataclass
class EvidenceAlignment:
    """Result of aligning gold spans against a retrieval ranking."""

    status: EvidenceStatus
    n_gold_spans: int
    n_covered_spans: int
    n_gold_docs: int
    n_covered_docs: int
    #: Rank of the highest-ranked chunk carrying any gold evidence.
    first_evidence_rank: int | None = None
    #: Fraction of gold spans covered by some retrieved chunk.
    evidence_recall: float | None = None
    #: Fraction of retrieved chunks carrying some gold evidence. Low values
    #: mean the generator's context was mostly padding.
    evidence_precision: float | None = None
    #: Gold documents for which no span was covered. Names what to fix.
    missing_doc_ids: list[str] = field(default_factory=list)
    #: True when some chunk lacked offsets, so alignment could not be exact.
    degraded_to_document_level: bool = False
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS
    evidence_mode: str = EvidenceMode.ANY_SUFFICIENT.value

    @property
    def is_complete(self) -> bool:
        return self.status is EvidenceStatus.COMPLETE

    def as_dict(self) -> dict:
        data = asdict(self)
        data["status"] = str(self.status)
        return data


def align_evidence(
    gold_spans: Sequence[GoldSpan],
    retrieved: Sequence[RetrievedSpan],
    evidence_mode: EvidenceMode | str = EvidenceMode.ANY_SUFFICIENT,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> EvidenceAlignment:
    """Align gold evidence spans against a retrieval ranking.

    `evidence_mode` decides what "complete" means. Under ANY_SUFFICIENT one
    covered span suffices. Under ALL_REQUIRED every gold *document* must
    contribute a covered span, which is what makes a half-retrieved multi-hop
    question a retrieval failure rather than a generation failure.

    A retrieved chunk with no offsets cannot be credited with any span, and the
    result is flagged `degraded_to_document_level`, so a corpus indexed before
    offsets existed produces a visibly degraded measurement rather than a
    quietly wrong one.
    """
    mode = EvidenceMode(evidence_mode)

    if not gold_spans:
        return EvidenceAlignment(
            status=EvidenceStatus.NOT_APPLICABLE,
            n_gold_spans=0,
            n_covered_spans=0,
            n_gold_docs=0,
            n_covered_docs=0,
            min_overlap_chars=min_overlap_chars,
            evidence_mode=str(mode),
        )

    gold_docs = list(dict.fromkeys(s.doc_id for s in gold_spans))
    degraded = any(not c.has_offsets for c in retrieved)

    covered_span_ranks: dict[int, int] = {}
    chunks_with_evidence: set[int] = set()
    covered_docs: set[str] = set()

    for chunk_index, chunk in enumerate(retrieved):
        if not chunk.has_offsets:
            # No exact position: this chunk cannot be credited with any span.
            continue
        for span_index, span in enumerate(gold_spans):
            shared = span.overlap(chunk.doc_id, chunk.start_char, chunk.end_char)
            if shared >= min_overlap_chars:
                covered_span_ranks.setdefault(span_index, chunk.rank)
                chunks_with_evidence.add(chunk_index)
                covered_docs.add(span.doc_id)

    n_covered = len(covered_span_ranks)
    first_rank = min(covered_span_ranks.values()) if covered_span_ranks else None

    if mode is EvidenceMode.ALL_REQUIRED:
        complete = len(covered_docs) == len(gold_docs)
    else:
        complete = n_covered > 0

    if complete:
        status = EvidenceStatus.COMPLETE
    elif n_covered > 0:
        status = EvidenceStatus.PARTIAL
    else:
        status = EvidenceStatus.NONE

    return EvidenceAlignment(
        status=status,
        n_gold_spans=len(gold_spans),
        n_covered_spans=n_covered,
        n_gold_docs=len(gold_docs),
        n_covered_docs=len(covered_docs),
        first_evidence_rank=first_rank,
        evidence_recall=n_covered / len(gold_spans),
        evidence_precision=(len(chunks_with_evidence) / len(retrieved)) if retrieved else 0.0,
        missing_doc_ids=[d for d in gold_docs if d not in covered_docs],
        degraded_to_document_level=degraded,
        min_overlap_chars=min_overlap_chars,
        evidence_mode=str(mode),
    )


# ---------------------------------------------------------------
# Attribution hierarchy
# ---------------------------------------------------------------

class AttributionStage(StrEnum):
    """Which stage a failure is charged to."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    ABSTENTION = "abstention"
    #: The corpus itself cannot support the question. Charging this to
    #: retrieval would blame the retriever for something no retriever could
    #: have found.
    CORPUS = "corpus"
    NONE = "none"


def attribute_stage(
    *,
    alignment: EvidenceAlignment,
    answer_is_correct: bool,
    is_answerable: bool,
    abstained: bool,
    n_retrieved: int,
) -> tuple[AttributionStage, str]:
    """Charge an outcome to a pipeline stage, evidence first.

    The ordering is the point of the module:

    1. An unanswerable question is judged only on whether the system abstained.
    2. A question whose required evidence never reached the generator is a
       retrieval failure, whatever the answer looks like. Crediting a correct
       answer here would credit memorisation, not retrieval.
    3. Only once the evidence was present does the answer decide the verdict.

    Returns the stage and a human-readable reason.
    """
    if not is_answerable:
        if abstained:
            return AttributionStage.NONE, "correctly abstained on an unanswerable question"
        return AttributionStage.ABSTENTION, "answered a question the corpus cannot support"

    if n_retrieved == 0:
        return AttributionStage.RETRIEVAL, "nothing was retrieved"

    if alignment.status is EvidenceStatus.NOT_APPLICABLE:
        if answer_is_correct:
            return AttributionStage.NONE, "answer correct; no gold evidence recorded"
        return AttributionStage.GENERATION, "answer incorrect; no gold evidence recorded"

    if alignment.status is EvidenceStatus.NONE:
        suffix = (
            " (answer nonetheless correct: not supported by retrieved context)"
            if answer_is_correct
            else ""
        )
        return AttributionStage.RETRIEVAL, "no gold evidence was retrieved" + suffix

    if alignment.status is EvidenceStatus.PARTIAL:
        missing = ", ".join(alignment.missing_doc_ids) or "some required evidence"
        return AttributionStage.RETRIEVAL, f"multi-hop evidence incomplete; missing {missing}"

    if answer_is_correct:
        return AttributionStage.NONE, "required evidence retrieved and answer correct"
    return AttributionStage.GENERATION, "required evidence was retrieved but the answer is wrong"


def answer_supported_by_evidence(alignment: EvidenceAlignment, answer_is_correct: bool) -> bool:
    """Whether a correct answer was actually grounded in retrieved evidence.

    A correct answer produced without the gold evidence in context indicates
    parametric knowledge, not working retrieval. Separating the two is what
    stops a memorisation-prone corpus from flattering the system.
    """
    return answer_is_correct and alignment.status is EvidenceStatus.COMPLETE


def spans_from_records(spans: Sequence[dict]) -> list[GoldSpan]:
    """Build gold spans from their serialised form."""
    return [
        GoldSpan(
            doc_id=str(s["doc_id"]),
            start_char=int(s["start_char"]),
            end_char=int(s["end_char"]),
        )
        for s in spans
    ]


def retrieved_from_chunks(chunks: Sequence[object]) -> list[RetrievedSpan]:
    """Build retrieval spans from `RetrievedChunk`-shaped objects."""
    return [
        RetrievedSpan(
            rank=int(getattr(c, "rank", index + 1)),
            doc_id=str(getattr(c, "doc_id", "unknown")),
            start_char=getattr(c, "start_char", None),
            end_char=getattr(c, "end_char", None),
        )
        for index, c in enumerate(chunks)
    ]
