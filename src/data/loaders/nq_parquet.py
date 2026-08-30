"""Natural Questions loader for the HuggingFace parquet distribution.

The original simplified release (`document_text` as one whitespace-joined
string) is handled by `natural_questions.py`. The distribution that is actually
obtainable today is the full release, published as parquet, where a document is
a token list with `is_html` flags and annotations index into that token list.

Three decisions shape this loader.

**The corpus is built from non-HTML tokens.** Keeping markup would put `<Table>`
and `<P>` into the retrieval corpus and into the generator's context, which is
not the document a RAG system should be reading. Dropping tokens shifts every
position, so the loader keeps an explicit map from original token index to the
character range that token occupies in the cleaned text; annotation offsets are
translated through it rather than recomputed.

**A page is a document, not a question.** Natural Questions ships one Wikipedia
page per item, and the same page recurs across items. Keying documents by the
page URL and deduplicating by content means two questions about one page share
one document. Keying by example id instead would put byte-identical copies in
the corpus under different ids, and retrieval of the "wrong" copy would then be
scored as a retrieval failure although the text answers the question.

**A page-scoped null is not a corpus-scoped unanswerable.** An NQ item with no
long answer means the annotator found no answer *on that page*. That is not the
same claim as "the corpus cannot answer this", so such items are skipped and
counted rather than emitted as abstention targets.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.identity import content_fingerprint
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

NO_ANSWER_TOKEN = -1

#: Tokens are joined with a single space when rebuilding the page text.
TOKEN_SEPARATOR = " "


@register_loader
class NaturalQuestionsParquetLoader(DatasetLoader):
    """Loads the HuggingFace parquet distribution of Natural Questions."""

    name = "nq"
    license_spdx = "CC-BY-SA-3.0"
    source_url = "https://huggingface.co/datasets/google-research-datasets/natural_questions"

    def load(self, path: Path, limit: int | None = None, split: str = "validation") -> LoadResult:
        import pyarrow.parquet as pq

        questions: list[QuestionRecord] = []
        documents: dict[str, Document] = {}
        # Page content fingerprint -> doc_id, so an identical page reached by a
        # second question resolves to the document already in the corpus.
        by_fingerprint: dict[str, str] = {}
        skipped: Counter[str] = Counter()

        parquet = pq.ParquetFile(str(path))
        for batch in parquet.iter_batches(batch_size=64):
            for item in batch.to_pylist():
                record = self._build(item, documents, by_fingerprint, split, skipped)
                if record is not None:
                    questions.append(record)
                    if limit is not None and len(questions) >= limit:
                        return self._result(questions, documents, skipped)
        return self._result(questions, documents, skipped)

    def _result(self, questions, documents, skipped) -> LoadResult:
        """Keep only documents some emitted question actually points at."""
        used = {span.doc_id for q in questions for span in q.supporting_spans}
        corpus = [d for doc_id, d in documents.items() if doc_id in used]
        return LoadResult(questions, corpus, self.name, dict(skipped))

    def _build(
        self,
        item: dict,
        documents: dict[str, Document],
        by_fingerprint: dict[str, str],
        split: str,
        skipped: Counter,
    ) -> QuestionRecord | None:
        question_text = str((item.get("question") or {}).get("text", "")).strip()
        if not question_text:
            skipped["question_empty"] += 1
            return None

        document = item.get("document") or {}
        tokens_block = document.get("tokens") or {}
        tokens = tokens_block.get("token") or []
        is_html = tokens_block.get("is_html") or []
        if not tokens:
            skipped["document_missing_tokens"] += 1
            return None

        text, token_spans = _clean_text_with_spans(tokens, is_html)
        if not text.strip():
            skipped["document_empty_after_html_removal"] += 1
            return None

        annotations = item.get("annotations") or {}
        long_answers = annotations.get("long_answer") or []
        short_answers = annotations.get("short_answers") or []
        yes_no = annotations.get("yes_no_answer") or []

        span_tokens = _first_long_answer(long_answers)
        if span_tokens is None:
            # Page-scoped null: not a corpus-scoped unanswerable.
            skipped["no_long_answer_page_scoped_null"] += 1
            return None

        char_span = _tokens_to_char_span(span_tokens, token_spans)
        if char_span is None:
            skipped["long_answer_not_representable_in_clean_text"] += 1
            return None

        answers, question_type = _collect_answers(short_answers, yes_no, text, token_spans)
        if not answers:
            skipped["no_short_answer_text"] += 1
            return None

        # One page, one document, regardless of how many questions reach it.
        url = str(document.get("url", "")).strip()
        title = str(document.get("title", "")).strip()
        fingerprint = content_fingerprint(text)
        doc_id = by_fingerprint.get(fingerprint)
        if doc_id is None:
            native = url or title or f"page{fingerprint}"
            doc_id = self.make_doc_id(native)
            by_fingerprint[fingerprint] = doc_id
            documents[doc_id] = Document(
                doc_id=doc_id,
                text=text,
                title=title,
                source=url or f"nq/{title}",
                metadata={
                    "url": url,
                    "n_tokens_original": len(tokens),
                    "n_tokens_kept": sum(1 for s in token_spans if s is not None),
                    "content_fingerprint": fingerprint,
                },
            )

        example_id = str(item.get("id", "")) or f"h{fingerprint}"
        start_char, end_char = char_span
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
                    text=text[start_char:end_char],
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
                "n_annotations": len(long_answers),
                "n_reference_answers": len(answers),
                "document_url": url,
            },
        )


def _clean_text_with_spans(
    tokens: list[str],
    is_html: list[bool],
) -> tuple[str, list[tuple[int, int] | None]]:
    """Join non-HTML tokens, recording where each original token landed.

    Returns the cleaned text and a list parallel to `tokens`: the character
    range of each kept token, or None for a token that was dropped. Annotation
    offsets index the original token list, so this map is what lets them be
    translated exactly instead of approximately.
    """
    pieces: list[str] = []
    spans: list[tuple[int, int] | None] = []
    cursor = 0

    for index, token in enumerate(tokens):
        html = bool(is_html[index]) if index < len(is_html) else False
        if html:
            spans.append(None)
            continue
        text = str(token)
        if not text:
            spans.append(None)
            continue
        start = cursor
        end = start + len(text)
        spans.append((start, end))
        pieces.append(text)
        cursor = end + len(TOKEN_SEPARATOR)

    return TOKEN_SEPARATOR.join(pieces), spans


def _first_long_answer(long_answers: list) -> tuple[int, int] | None:
    """Token range of the first annotator who marked a long answer."""
    for annotation in long_answers:
        if not isinstance(annotation, dict):
            continue
        start = int(annotation.get("start_token", NO_ANSWER_TOKEN))
        end = int(annotation.get("end_token", NO_ANSWER_TOKEN))
        if start != NO_ANSWER_TOKEN and end > start:
            return start, end
    return None


def _tokens_to_char_span(
    token_range: tuple[int, int],
    token_spans: list[tuple[int, int] | None],
) -> tuple[int, int] | None:
    """Translate an original token range to a character range in clean text.

    HTML tokens inside the range were dropped, so the span runs from the first
    kept token at or after the start to the last kept token before the end.
    Returns None when the range contained no kept token at all, which happens
    for long answers that are pure markup such as an empty table element.
    """
    start_token, end_token = token_range
    kept = [
        span
        for index, span in enumerate(token_spans)
        if span is not None and start_token <= index < end_token
    ]
    if not kept:
        return None
    return kept[0][0], kept[-1][1]


def _collect_answers(
    short_answers: list,
    yes_no: list,
    text: str,
    token_spans: list[tuple[int, int] | None],
) -> tuple[list[str], QuestionType]:
    """Gather every annotator's acceptable answer.

    NQ dev is five-way annotated, and the annotators do not always agree. All
    distinct answers are kept so scoring can take the maximum over references
    rather than arbitrarily privileging the first annotator.
    """
    answers: list[str] = []
    question_type = QuestionType.FACTOID

    for value in yes_no or []:
        label = str(value).upper()
        if label in {"YES", "NO"}:
            answers.append(label.capitalize())
            question_type = QuestionType.YES_NO

    for annotation in short_answers or []:
        if not isinstance(annotation, dict):
            continue
        starts = annotation.get("start_token") or []
        ends = annotation.get("end_token") or []
        if len(starts) > 1:
            question_type = QuestionType.LIST
        for start, end in zip(starts, ends, strict=False):
            span = _tokens_to_char_span((int(start), int(end)), token_spans)
            if span is None:
                continue
            candidate = text[span[0] : span[1]].strip()
            if candidate:
                answers.append(candidate)

    return list(dict.fromkeys(answers)), question_type
