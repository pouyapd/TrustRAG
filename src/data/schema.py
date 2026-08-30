"""The unified question schema.

Every dataset loader produces `QuestionRecord`s, and everything downstream reads
them. Three decisions in here matter more than the rest:

**Evidence is anchored to character spans, never to chunk ids.** Chunking is a
swept experimental variable, so a chunk-level label is only valid for the one
chunking configuration that produced it. Store the span; derive chunk relevance
per configuration.

**Answerability is a property of the corpus, not of a passage.** A question is
unanswerable when the corpus in use cannot support an answer. That is why
ablated questions carry `removed_doc_ids`: the label is only meaningful
alongside the corpus view it was computed against.

**Ablated questions keep a link to their original** via `pair_id`, so abstention
can be measured as a paired comparison on identical question wording.

The serialised form is a superset of the legacy evaluation format: `answer` and
`relevant_doc_ids` are emitted as derived fields, so a built dataset file feeds
`evaluation.runner.load_dataset` unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

SCHEMA_VERSION = "1.0"

#: Reference answer written for an unanswerable question. Matches the wording
#: used by the bundled fixture and by the pipeline's refusal markers.
DEFAULT_REFUSAL = "I cannot answer this from the provided context."


class Answerability(StrEnum):
    """Whether the corpus can support an answer, and if not, why not.

    The unanswerable variants are the typology from the dataset design: they
    exist so abstention failures can be broken down by kind rather than
    collapsed into a single rate.
    """

    ANSWERABLE = "answerable"
    #: Supporting documents deliberately removed from the corpus.
    UNANSWERABLE_EVIDENCE_REMOVED = "unanswerable_evidence_removed"
    #: The source dataset itself marks the question unanswerable (e.g. QASPER).
    UNANSWERABLE_NATIVE = "unanswerable_native"
    #: Presupposes an entity or attribute the corpus never supports.
    UNANSWERABLE_FALSE_PREMISE = "unanswerable_false_premise"
    #: Multi-part question the corpus can only partly support.
    UNANSWERABLE_PARTIAL = "unanswerable_partially_answerable"
    #: Requires a time period the corpus does not cover.
    UNANSWERABLE_TEMPORAL = "unanswerable_temporal"
    #: Off topic entirely. The easy control condition.
    UNANSWERABLE_OUT_OF_DOMAIN = "unanswerable_out_of_domain"

    @property
    def is_answerable(self) -> bool:
        return self is Answerability.ANSWERABLE


class EvidenceMode(StrEnum):
    """How the supporting spans combine to justify the answer."""

    #: Any one span is enough (single-hop).
    ANY_SUFFICIENT = "any_sufficient"
    #: Every span is needed (multi-hop). Removing one makes it unanswerable.
    ALL_REQUIRED = "all_required"


class QuestionType(StrEnum):
    """Coarse question shape, used for stratified sampling and breakdowns."""

    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    YES_NO = "yes_no"
    ABSTRACTIVE = "abstractive"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SupportingSpan:
    """A character range in a document that supports the answer."""

    doc_id: str
    start_char: int
    end_char: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError(f"start_char must be >= 0, got {self.start_char}")
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be greater than "
                f"start_char ({self.start_char})"
            )
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")

    @property
    def length(self) -> int:
        return self.end_char - self.start_char

    def overlaps(self, start: int, end: int) -> int:
        """Number of characters shared with the range [start, end)."""
        return max(0, min(self.end_char, end) - max(self.start_char, start))

    def resolve(self, document: Document) -> str:
        """The text this span actually points at, read from the document."""
        return document.text[self.start_char : self.end_char]

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SupportingSpan:
        return cls(
            doc_id=str(data["doc_id"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            text=str(data.get("text", "")),
        )


@dataclass
class Document:
    """One corpus document. The unit that ablation removes."""

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        return cls(
            doc_id=str(data["doc_id"]),
            text=str(data["text"]),
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SourceInfo:
    """Where a question came from. Carried so provenance survives every step."""

    dataset: str
    split: str = ""
    item_id: str = ""
    license: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SourceInfo:
        return cls(
            dataset=str(data.get("dataset", "unknown")),
            split=str(data.get("split", "")),
            item_id=str(data.get("item_id", "")),
            license=str(data.get("license", "")),
        )


@dataclass
class QuestionRecord:
    """One evaluation question with its evidence and answerability label."""

    question_id: str
    corpus_id: str
    question: str
    answerability: Answerability = Answerability.ANSWERABLE
    answers: list[str] = field(default_factory=list)
    supporting_spans: list[SupportingSpan] = field(default_factory=list)
    evidence_mode: EvidenceMode = EvidenceMode.ANY_SUFFICIENT
    question_type: QuestionType = QuestionType.UNKNOWN
    hops: int = 1
    #: Documents removed from the corpus to make this question unanswerable.
    removed_doc_ids: list[str] = field(default_factory=list)
    #: question_id of the answerable original, for ablated items.
    pair_id: str = ""
    source: SourceInfo = field(default_factory=lambda: SourceInfo(dataset="unknown"))
    split: str = "test"
    schema_version: str = SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    # ---- derived views ----

    @property
    def is_answerable(self) -> bool:
        return self.answerability.is_answerable

    @property
    def relevant_doc_ids(self) -> list[str]:
        """Distinct supporting documents, in first-appearance order.

        This is the legacy relevance signal. It is empty for unanswerable
        questions, which is exactly what the v2 taxonomy keys abstention off.
        """
        seen: list[str] = []
        for span in self.supporting_spans:
            if span.doc_id not in seen:
                seen.append(span.doc_id)
        return seen

    @property
    def reference_answer(self) -> str:
        """Single reference string for the legacy evaluation format."""
        if self.answers:
            return self.answers[0]
        return "" if self.is_answerable else DEFAULT_REFUSAL

    def to_eval_item(self) -> dict:
        """The dict shape `evaluation.runner.load_dataset` already consumes."""
        return {
            "question": self.question,
            "answer": self.reference_answer,
            "relevant_doc_ids": list(self.relevant_doc_ids),
        }

    def to_experiment_item(self) -> dict:
        """The full item shape the evidence-aware runner consumes.

        A superset of `to_eval_item()`: the legacy three keys are still present
        and unchanged, so the same dict also feeds the legacy runner, while the
        extra keys carry the evidence spans, answerability typology and every
        acceptable answer that evidence-aware scoring needs.
        """
        item = self.to_eval_item()
        item.update(
            {
                "question_id": self.question_id,
                "answers": list(self.answers),
                "supporting_spans": [s.as_dict() for s in self.supporting_spans],
                "evidence_mode": str(self.evidence_mode),
                "answerability": str(self.answerability),
                "hops": self.hops,
                "question_type": str(self.question_type),
            }
        )
        return item

    def chunk_is_relevant(
        self,
        doc_id: str,
        start_char: int,
        end_char: int,
        min_overlap_chars: int = 1,
    ) -> bool:
        """Whether a chunk counts as relevant under a given overlap rule.

        This is how chunk-level gold is derived per chunking configuration
        rather than stored. `min_overlap_chars` is a declared parameter of the
        experiment and should be reported alongside any retrieval metric.
        """
        for span in self.supporting_spans:
            if span.doc_id != doc_id:
                continue
            if span.overlaps(start_char, end_char) >= min_overlap_chars:
                return True
        return False

    # ---- serialisation ----

    def as_dict(self) -> dict:
        """Full record, including the derived legacy fields."""
        return {
            "question_id": self.question_id,
            "corpus_id": self.corpus_id,
            "question": self.question,
            "answerability": str(self.answerability),
            "answers": list(self.answers),
            "supporting_spans": [s.as_dict() for s in self.supporting_spans],
            "evidence_mode": str(self.evidence_mode),
            "question_type": str(self.question_type),
            "hops": self.hops,
            "removed_doc_ids": list(self.removed_doc_ids),
            "pair_id": self.pair_id,
            "source": self.source.as_dict(),
            "split": self.split,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            # Derived, for backward compatibility with the legacy runner.
            # Never author these by hand; they are regenerated on every write.
            "answer": self.reference_answer,
            "relevant_doc_ids": list(self.relevant_doc_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuestionRecord:
        return cls(
            question_id=str(data["question_id"]),
            corpus_id=str(data.get("corpus_id", "")),
            question=str(data["question"]),
            answerability=Answerability(data.get("answerability", "answerable")),
            answers=list(data.get("answers", [])),
            supporting_spans=[
                SupportingSpan.from_dict(s) for s in data.get("supporting_spans", [])
            ],
            evidence_mode=EvidenceMode(data.get("evidence_mode", "any_sufficient")),
            question_type=QuestionType(data.get("question_type", "unknown")),
            hops=int(data.get("hops", 1)),
            removed_doc_ids=list(data.get("removed_doc_ids", [])),
            pair_id=str(data.get("pair_id", "")),
            source=SourceInfo.from_dict(data.get("source", {})),
            split=str(data.get("split", "test")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

def validate_record(
    record: QuestionRecord,
    documents: dict[str, Document] | None = None,
) -> list[str]:
    """Structural problems with one record. Empty list means valid.

    When `documents` is supplied the spans are also checked against the real
    text: a span pointing past the end of its document, or whose stored text no
    longer matches what the offsets resolve to, is a silent corruption that
    would otherwise surface as an inexplicable retrieval failure much later.
    """
    issues: list[str] = []

    if not record.question_id:
        issues.append("question_id is empty")
    if not record.question.strip():
        issues.append("question is empty")

    if record.is_answerable:
        if not record.supporting_spans:
            issues.append("answerable question has no supporting spans")
        if not record.answers:
            issues.append("answerable question has no reference answer")
        if record.removed_doc_ids:
            issues.append("answerable question must not declare removed_doc_ids")
    else:
        if record.supporting_spans:
            issues.append("unanswerable question must have no supporting spans")
        evidence_removed = record.answerability is Answerability.UNANSWERABLE_EVIDENCE_REMOVED
        if evidence_removed and not record.removed_doc_ids:
            issues.append("evidence-removed question does not record removed_doc_ids")

    if (
        record.evidence_mode is EvidenceMode.ALL_REQUIRED
        and record.is_answerable
        and len(record.supporting_spans) < 2
    ):
        issues.append("all_required evidence mode needs at least two spans")

    if record.hops < 1:
        issues.append(f"hops must be >= 1, got {record.hops}")

    if documents is not None:
        for span in record.supporting_spans:
            document = documents.get(span.doc_id)
            if document is None:
                issues.append(f"span references unknown document {span.doc_id!r}")
                continue
            if span.end_char > len(document.text):
                issues.append(
                    f"span [{span.start_char}:{span.end_char}] exceeds document "
                    f"{span.doc_id!r} of length {len(document.text)}"
                )
            elif span.text and span.resolve(document) != span.text:
                issues.append(
                    f"span text does not match document {span.doc_id!r} at "
                    f"[{span.start_char}:{span.end_char}]"
                )
    return issues


def validate_dataset(
    records: list[QuestionRecord],
    documents: dict[str, Document] | None = None,
) -> dict[str, list[str]]:
    """Validate a whole dataset. Returns {question_id: issues} for bad records."""
    problems: dict[str, list[str]] = {}
    seen_ids: set[str] = set()

    for record in records:
        issues = validate_record(record, documents)
        if record.question_id in seen_ids:
            issues.append("duplicate question_id")
        seen_ids.add(record.question_id)
        if issues:
            problems[record.question_id] = issues
    return problems


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

def write_questions(records: list[QuestionRecord], path: Path) -> Path:
    """Write questions as JSONL. Deterministic: stable key order, no timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_questions(path: Path) -> list[QuestionRecord]:
    """Read questions from JSONL."""
    records: list[QuestionRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(QuestionRecord.from_dict(json.loads(line)))
    return records


def write_documents(documents: list[Document], path: Path) -> Path:
    """Write the corpus as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_documents(path: Path) -> list[Document]:
    """Read the corpus from JSONL."""
    documents: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(Document.from_dict(json.loads(line)))
    return documents


def index_documents(documents: list[Document]) -> dict[str, Document]:
    """Map doc_id -> Document."""
    return {d.doc_id: d for d in documents}
