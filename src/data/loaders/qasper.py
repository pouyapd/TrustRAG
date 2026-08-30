"""QASPER loader — QA over NLP papers.

Licence: CC BY 4.0. 5,049 questions over 1,585 papers, including 810 natively
unanswerable ones, which is why this dataset needs no ablation to contribute
abstention items.

Its methodological value: each question was written by an NLP practitioner who
had read **only the title and abstract**. The question author had not seen the
body text the answer comes from, so questions cannot inherit the gold
paragraph's vocabulary. That makes it the strongest anti-lexical-anchoring
source in the selected set.

Native format: a JSON object keyed by paper id::

    {"1234.5678": {
        "title": ..., "abstract": ...,
        "full_text": [{"section_name": ..., "paragraphs": [...]}, ...],
        "qas": [{"question": ..., "question_id": ...,
                 "answers": [{"answer": {"unanswerable": bool,
                                         "extractive_spans": [...],
                                         "yes_no": bool|null,
                                         "free_form_answer": str,
                                         "evidence": [...paragraph strings...]}}]}]}}

Evidence arrives as paragraph *strings*. This loader reconstructs the paper
body as one document and locates each evidence paragraph within it to recover
character offsets, because the schema anchors gold to spans rather than to
paragraph indices.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.identity import question_id
from src.data.loaders.base import DatasetLoader, LoadResult, register_loader
from src.data.schema import (
    Answerability,
    Document,
    EvidenceMode,
    QuestionRecord,
    QuestionType,
    SourceInfo,
    SupportingSpan,
)

#: Separator used when joining paragraphs into one document body. Fixed, because
#: character offsets are computed against the joined text and must be stable.
PARAGRAPH_SEPARATOR = "\n\n"


@register_loader
class QasperLoader(DatasetLoader):
    """Loads QASPER into the unified schema."""

    name = "qasper"
    license_spdx = "CC-BY-4.0"
    source_url = "https://allenai.org/data/qasper"

    def load(self, path: Path, limit: int | None = None, split: str = "test") -> LoadResult:
        raw = self.read_json(path)
        self.require(
            isinstance(raw, dict),
            f"QASPER expects a JSON object keyed by paper id, got {type(raw).__name__}",
        )

        questions: list[QuestionRecord] = []
        documents: list[Document] = []
        skipped: Counter[str] = Counter()

        for paper_id, paper in raw.items():
            if not isinstance(paper, dict):
                skipped["paper_not_an_object"] += 1
                continue

            body, paragraph_offsets = _build_body(paper)
            if not body:
                skipped["paper_without_body"] += 1
                continue

            doc_id = self.make_doc_id(str(paper_id))
            documents.append(
                Document(
                    doc_id=doc_id,
                    text=body,
                    title=str(paper.get("title", "")),
                    source=f"qasper/{paper_id}",
                    metadata={
                        "paper_id": str(paper_id),
                        "abstract": str(paper.get("abstract", "")),
                        "n_paragraphs": len(paragraph_offsets),
                    },
                )
            )

            for qa in paper.get("qas", []) or []:
                record = self._build_question(
                    qa, paper_id, doc_id, body, paragraph_offsets, split, skipped
                )
                if record is not None:
                    questions.append(record)
                    if limit is not None and len(questions) >= limit:
                        return LoadResult(
                            questions, _used_documents(questions, documents),
                            self.name, dict(skipped),
                        )

        return LoadResult(
            questions, _used_documents(questions, documents), self.name, dict(skipped)
        )

    def _build_question(
        self,
        qa: dict,
        paper_id: str,
        doc_id: str,
        body: str,
        paragraph_offsets: dict[str, tuple[int, int]],
        split: str,
        skipped: Counter,
    ) -> QuestionRecord | None:
        question_text = str(qa.get("question", "")).strip()
        if not question_text:
            skipped["question_empty"] += 1
            return None

        native_id = str(qa.get("question_id", "")) or ""
        # Deterministic content hash when the release carries no question id;
        # `hash()` would change between processes and break annotation keys.
        item_id = native_id or question_id("qasper", None, f"{paper_id} {question_text}")[
            len("qasper:") :
        ]
        answer_blocks = [a.get("answer", {}) for a in qa.get("answers", []) or []]
        if not answer_blocks:
            skipped["question_without_answers"] += 1
            return None

        # A question is unanswerable only when every annotator said so; if any
        # annotator found an answer, the corpus does support one.
        unanswerable = all(bool(a.get("unanswerable")) for a in answer_blocks)

        source = SourceInfo(
            dataset=self.name,
            split=split,
            item_id=item_id,
            license=self.license_spdx,
        )

        if unanswerable:
            return QuestionRecord(
                question_id=self.make_question_id(item_id),
                corpus_id=self.corpus_id,
                question=question_text,
                answerability=Answerability.UNANSWERABLE_NATIVE,
                answers=[],
                supporting_spans=[],
                question_type=QuestionType.UNKNOWN,
                source=source,
                split=split,
                metadata={"paper_id": paper_id, "n_annotations": len(answer_blocks)},
            )

        answers, answer_type = _collect_answers(answer_blocks)
        if not answers:
            skipped["answerable_without_answer_text"] += 1
            return None

        spans, unresolved = _evidence_spans(answer_blocks, doc_id, body, paragraph_offsets)
        if not spans:
            # Every piece of evidence was a figure/table caption or unmatched:
            # the text corpus cannot support this question at all.
            reason = (
                "evidence_only_in_figures_tables"
                if unresolved["float"] and not unresolved["unmatched"]
                else "evidence_not_locatable"
            )
            skipped[reason] += 1
            return None

        return QuestionRecord(
            question_id=self.make_question_id(item_id),
            corpus_id=self.corpus_id,
            question=question_text,
            answerability=Answerability.ANSWERABLE,
            answers=answers,
            supporting_spans=spans,
            # Evidence paragraphs are alternative supports from different
            # annotators, so any one is sufficient.
            evidence_mode=EvidenceMode.ANY_SUFFICIENT,
            question_type=answer_type,
            hops=1,
            source=source,
            split=split,
            metadata={
                "paper_id": paper_id,
                "n_annotations": len(answer_blocks),
                "n_evidence_spans": len(spans),
                "unresolved_float_evidence": unresolved["float"],
                "unresolved_unmatched_evidence": unresolved["unmatched"],
            },
        )


def _used_documents(questions: list, documents: list) -> list:
    """Only the documents some emitted question actually points at.

    Reaching a `limit` mid-paper, or skipping every question in a paper, would
    otherwise leave orphan documents in the corpus and change retrieval
    difficulty in a way the caller never asked for.
    """
    referenced = {span.doc_id for q in questions for span in q.supporting_spans}
    referenced |= {str(q.metadata.get("doc_id", "")) for q in questions}
    return [d for d in documents if d.doc_id in referenced]


def _build_body(paper: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    """Join the paper's paragraphs into one document, tracking offsets.

    Returns the body text and a map from paragraph text to its (start, end)
    character range, used to resolve evidence strings to spans.

    The abstract is included as the first paragraph. QASPER questions were
    written by people who had read only the title and abstract, so abstract
    text is a common evidence target; excluding it silently discarded every
    abstract-grounded question as "evidence not locatable" and biased the
    surviving sample towards body-grounded questions.
    """
    pieces: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0

    abstract = str(paper.get("abstract", "")).strip()
    if abstract:
        offsets.setdefault(abstract, (0, len(abstract)))
        pieces.append(abstract)
        cursor = len(abstract) + len(PARAGRAPH_SEPARATOR)

    for section in paper.get("full_text", []) or []:
        if not isinstance(section, dict):
            continue
        for paragraph in section.get("paragraphs", []) or []:
            text = str(paragraph).strip()
            if not text:
                continue
            start = cursor
            end = start + len(text)
            # First occurrence wins: duplicated paragraph text resolves to its
            # earliest position, which keeps offsets deterministic.
            offsets.setdefault(text, (start, end))
            pieces.append(text)
            cursor = end + len(PARAGRAPH_SEPARATOR)

    return PARAGRAPH_SEPARATOR.join(pieces), offsets


def _collect_answers(answer_blocks: list[dict]) -> tuple[list[str], QuestionType]:
    """Gather acceptable answer strings and infer the question type."""
    answers: list[str] = []
    question_type = QuestionType.UNKNOWN

    for block in answer_blocks:
        if block.get("yes_no") is not None:
            answers.append("Yes" if block["yes_no"] else "No")
            question_type = QuestionType.YES_NO
            continue

        spans = [str(s).strip() for s in block.get("extractive_spans", []) or [] if str(s).strip()]
        if spans:
            answers.extend(spans)
            if question_type is QuestionType.UNKNOWN:
                question_type = QuestionType.LIST if len(spans) > 1 else QuestionType.FACTOID
            continue

        free_form = str(block.get("free_form_answer", "")).strip()
        if free_form:
            answers.append(free_form)
            if question_type is QuestionType.UNKNOWN:
                question_type = QuestionType.ABSTRACTIVE

    # Preserve order, drop duplicates.
    return list(dict.fromkeys(answers)), question_type


#: Prefix QASPER uses for figure and table captions. Those live outside
#: `full_text`, so such evidence has no character span in the paper body. On
#: the dev split this accounts for 253 of 384 unresolvable evidence strings.
FLOAT_PREFIX = "FLOAT SELECTED"


def _evidence_spans(
    answer_blocks: list[dict],
    doc_id: str,
    body: str,
    paragraph_offsets: dict[str, tuple[int, int]],
) -> tuple[list[SupportingSpan], dict[str, int]]:
    """Resolve evidence paragraph strings to character spans in the body.

    Returns the resolved spans and a census of what could not be resolved.
    Nothing is guessed at: evidence with no exact position in the body is
    excluded from the spans and counted instead, because an approximate
    position would put gold evidence in the wrong place and silently corrupt
    every span-level retrieval metric computed from it.

    Two kinds are distinguished, because they mean different things:

    - `float`: a figure or table caption. QASPER stores these outside
      `full_text`, so the paper body genuinely does not contain them. A
      question supported only by float evidence is not answerable from the
      text corpus this pipeline builds, and reporting it as an ordinary
      retrieval failure would be wrong.
    - `unmatched`: evidence that should be in the body but does not match any
      paragraph exactly, usually because of normalisation differences.
    """
    spans: list[SupportingSpan] = []
    seen: set[tuple[int, int]] = set()
    unresolved = {"float": 0, "unmatched": 0}

    for block in answer_blocks:
        for evidence in block.get("evidence", []) or []:
            text = str(evidence).strip()
            if not text:
                continue

            location = paragraph_offsets.get(text)
            if location is None:
                # Some releases normalise whitespace in the evidence field
                # relative to full_text. This is an exact containment check on
                # the paper body, not a fuzzy match.
                index = body.find(text)
                if index < 0:
                    key = "float" if text.startswith(FLOAT_PREFIX) else "unmatched"
                    unresolved[key] += 1
                    continue
                location = (index, index + len(text))

            if location in seen:
                continue
            seen.add(location)
            spans.append(
                SupportingSpan(
                    doc_id=doc_id,
                    start_char=location[0],
                    end_char=location[1],
                    text=body[location[0] : location[1]],
                )
            )
    return spans, unresolved
