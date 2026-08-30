"""Natural Questions loader — real Google Search queries over Wikipedia.

Licence: CC BY-SA 3.0. 307,373 train / 7,830 dev (5-way annotated) / 7,842 test.

Selected as the primary corpus because it is the only major dataset carrying
**both** passage-level (long answer) and span-level (short answer) human
annotations, and because its questions are real search queries written without
sight of any Wikipedia page — so they carry no lexical anchoring to the gold
passage.

Two properties of NQ shape this loader:

**The null label is page-scoped, not corpus-scoped.** An NQ item with no long
answer means "not on *this* page", which is not the same as "not in the
corpus". Those items are therefore **not** emitted as unanswerable; corpus-scoped
unanswerables come from `data.ablation` instead. Null items are skipped and
counted.

**Use the original release, not NQ-Open.** NQ-Open discards the passage
annotations, which are the reason this dataset was selected.

Native format (simplified release), one JSON object per line::

    {"example_id": int, "question_text": str,
     "document_text": str,                  # whitespace-joined tokens
     "long_answer_candidates": [{"start_token": int, "end_token": int, ...}],
     "annotations": [{"long_answer": {"start_token": int, "end_token": int},
                      "short_answers": [{"start_token": int, "end_token": int}],
                      "yes_no_answer": "YES"|"NO"|"NONE"}]}

Token offsets are converted to character offsets against `document_text`.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

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

#: Sentinel used by NQ for "no annotation".
NO_ANSWER_TOKEN = -1


@register_loader
class NaturalQuestionsLoader(DatasetLoader):
    """Loads the simplified Natural Questions release into the unified schema."""

    name = "nq_simplified"
    license_spdx = "CC-BY-SA-3.0"
    source_url = "https://ai.google.com/research/NaturalQuestions"

    def load(self, path: Path, limit: int | None = None, split: str = "test") -> LoadResult:
        questions: list[QuestionRecord] = []
        documents: list[Document] = []
        skipped: Counter[str] = Counter()

        for item in self.read_jsonl(path):
            record = self._build(item, documents, split, skipped)
            if record is not None:
                questions.append(record)
                if limit is not None and len(questions) >= limit:
                    break

        return LoadResult(questions, documents, self.name, dict(skipped))

    def _build(
        self,
        item: dict,
        documents: list[Document],
        split: str,
        skipped: Counter,
    ) -> QuestionRecord | None:
        question_text = str(item.get("question_text", "")).strip()
        document_text = item.get("document_text")

        if not question_text:
            skipped["question_empty"] += 1
            return None
        if not isinstance(document_text, str) or not document_text:
            # The non-simplified release stores tokens instead; fail loudly
            # rather than silently producing an empty dataset.
            self.require(
                "document_tokens" not in item,
                "this looks like the full Natural Questions release; the simplified "
                "release with a 'document_text' field is required",
            )
            skipped["document_missing"] += 1
            return None

        annotations = item.get("annotations") or []
        if not annotations:
            skipped["no_annotations"] += 1
            return None

        token_offsets = _token_char_offsets(document_text)
        example_id = str(item.get("example_id", "")) or str(abs(hash(question_text)))

        long_answer = _first_long_answer(annotations)
        if long_answer is None:
            # Page-scoped null. Not a corpus-scoped unanswerable, so it is not
            # emitted as one -- see the module docstring.
            skipped["no_long_answer_page_scoped_null"] += 1
            return None

        span = _tokens_to_span(long_answer, token_offsets)
        if span is None:
            skipped["long_answer_offsets_unresolvable"] += 1
            return None

        doc_id = self.make_doc_id(example_id)
        documents.append(
            Document(
                doc_id=doc_id,
                text=document_text,
                title=str(item.get("document_title", "")),
                source=str(item.get("document_url", "")) or f"nq/{example_id}",
                metadata={"example_id": example_id, "n_tokens": len(token_offsets)},
            )
        )

        answers, question_type = _collect_answers(annotations, document_text, token_offsets)
        if not answers:
            skipped["no_short_answer_text"] += 1
            return None

        start_char, end_char = span
        return QuestionRecord(
            question_id=self.make_question_id(example_id),
            corpus_id=self.corpus_id,
            question=question_text,
            answerability=Answerability.ANSWERABLE,
            answers=answers,
            supporting_spans=[
                SupportingSpan(
                    doc_id=doc_id,
                    start_char=start_char,
                    end_char=end_char,
                    text=document_text[start_char:end_char],
                )
            ],
            evidence_mode=EvidenceMode.ANY_SUFFICIENT,
            question_type=question_type,
            hops=1,
            source=SourceInfo(
                dataset=self.name,
                split=split,
                item_id=example_id,
                license=self.license_spdx,
            ),
            split=split,
            metadata={
                "example_id": example_id,
                "n_annotations": len(annotations),
            },
        )


def _token_char_offsets(document_text: str) -> list[tuple[int, int]]:
    """Character range of every whitespace-delimited token.

    The simplified release joins tokens with single spaces, so token index maps
    to a character range by walking the string once.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    length = len(document_text)

    while cursor < length:
        while cursor < length and document_text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        start = cursor
        while cursor < length and not document_text[cursor].isspace():
            cursor += 1
        offsets.append((start, cursor))
    return offsets


def _first_long_answer(annotations: list) -> dict | None:
    """The first annotation carrying a real long answer, if any."""
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        long_answer = annotation.get("long_answer")
        if not isinstance(long_answer, dict):
            continue
        if int(long_answer.get("start_token", NO_ANSWER_TOKEN)) != NO_ANSWER_TOKEN:
            return long_answer
    return None


def _tokens_to_span(
    answer: dict,
    token_offsets: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Convert an NQ token range to a character range."""
    try:
        start_token = int(answer.get("start_token", NO_ANSWER_TOKEN))
        end_token = int(answer.get("end_token", NO_ANSWER_TOKEN))
    except (TypeError, ValueError):
        return None

    if start_token == NO_ANSWER_TOKEN or end_token <= start_token:
        return None
    if start_token >= len(token_offsets):
        return None

    # NQ end_token is exclusive.
    last_token = min(end_token, len(token_offsets)) - 1
    if last_token < start_token:
        return None
    return token_offsets[start_token][0], token_offsets[last_token][1]


def _collect_answers(
    annotations: list,
    document_text: str,
    token_offsets: list[tuple[int, int]],
) -> tuple[list[str], QuestionType]:
    """Gather short answers across annotators, or a yes/no answer."""
    answers: list[str] = []
    question_type = QuestionType.FACTOID

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        yes_no = str(annotation.get("yes_no_answer", "NONE")).upper()
        if yes_no in {"YES", "NO"}:
            answers.append(yes_no.capitalize())
            question_type = QuestionType.YES_NO
            continue

        short_answers = annotation.get("short_answers") or []
        for short in short_answers:
            if not isinstance(short, dict):
                continue
            span = _tokens_to_span(short, token_offsets)
            if span is None:
                continue
            text = document_text[span[0] : span[1]].strip()
            if text:
                answers.append(text)
        if len(short_answers) > 1:
            question_type = QuestionType.LIST

    return list(dict.fromkeys(answers)), question_type
