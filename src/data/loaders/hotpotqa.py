"""HotpotQA loader — multi-hop QA over Wikipedia introductions.

Licence: CC BY-SA 4.0 (dataset and processed corpus). Selected as a
**supplementary subset only**, for two reasons recorded in the corpus review:

- Sentence-level supporting facts are the finest evidence granularity in the
  selected set, and they map cleanly onto character spans.
- Genuine multi-hop items give `evidence_mode = all_required`, which makes the
  hardest ablation case available: remove one of two required documents and the
  question becomes unanswerable while highly relevant context remains.

Its known weakness is that crowdworkers wrote the questions while looking at the
paragraphs, so lexical anchoring is present and retrieval looks easier than it
is. Question-to-gold overlap should be reported separately for this subset.

Native format (distractor setting): a JSON list of objects::

    {"_id": ..., "question": ..., "answer": ...,
     "supporting_facts": [[title, sent_id], ...],
     "context": [[title, [sentence, ...]], ...],
     "type": "comparison"|"bridge", "level": "easy"|"medium"|"hard"}

Each context paragraph becomes one document; supporting facts identify
sentences, which this loader converts to character offsets within them.
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


@register_loader
class HotpotQaLoader(DatasetLoader):
    """Loads HotpotQA into the unified schema."""

    name = "hotpotqa"
    license_spdx = "CC-BY-SA-4.0"
    source_url = "https://hotpotqa.github.io/"

    def __init__(self, corpus_id: str | None = None, multi_hop_only: bool = True) -> None:
        super().__init__(corpus_id=corpus_id)
        #: Keep only items whose evidence genuinely spans two or more documents.
        #: The subset exists to supply multi-hop difficulty, so single-document
        #: items would defeat its purpose.
        self.multi_hop_only = multi_hop_only

    def load(self, path: Path, limit: int | None = None, split: str = "test") -> LoadResult:
        raw = self.read_json(path)
        self.require(
            isinstance(raw, list),
            f"HotpotQA expects a JSON list of items, got {type(raw).__name__}",
        )

        questions: list[QuestionRecord] = []
        documents: dict[str, Document] = {}
        skipped: Counter[str] = Counter()

        for item in raw:
            if not isinstance(item, dict):
                skipped["item_not_an_object"] += 1
                continue

            record = self._build_question(item, documents, split, skipped)
            if record is not None:
                questions.append(record)
                if limit is not None and len(questions) >= limit:
                    break

        # Only keep documents actually reachable from the questions we kept, so
        # the corpus does not carry distractors for discarded items.
        used = {span.doc_id for q in questions for span in q.supporting_spans}
        used |= {d for q in questions for d in q.metadata.get("distractor_doc_ids", [])}
        corpus = [doc for doc_id, doc in documents.items() if doc_id in used]

        return LoadResult(questions, corpus, self.name, dict(skipped))

    def _build_question(
        self,
        item: dict,
        documents: dict[str, Document],
        split: str,
        skipped: Counter,
    ) -> QuestionRecord | None:
        question_text = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question_text:
            skipped["question_empty"] += 1
            return None
        if not answer:
            skipped["answer_empty"] += 1
            return None

        item_id = str(item.get("_id", "")) or str(abs(hash(question_text)))

        # Build one document per context paragraph, recording sentence offsets.
        sentence_offsets: dict[str, list[tuple[int, int]]] = {}
        titles_in_context: list[str] = []

        for entry in item.get("context", []) or []:
            if not (isinstance(entry, list | tuple) and len(entry) == 2):
                skipped["malformed_context_entry"] += 1
                continue
            title, sentences = entry
            title = str(title)
            text, offsets = _join_sentences(sentences)
            if not text:
                continue

            doc_id = self.make_doc_id(title)
            titles_in_context.append(title)
            sentence_offsets[title] = offsets
            documents.setdefault(
                doc_id,
                Document(
                    doc_id=doc_id,
                    text=text,
                    title=title,
                    source=f"hotpotqa/{title}",
                    metadata={"n_sentences": len(offsets)},
                ),
            )

        spans: list[SupportingSpan] = []
        gold_titles: list[str] = []
        for fact in item.get("supporting_facts", []) or []:
            if not (isinstance(fact, list | tuple) and len(fact) == 2):
                skipped["malformed_supporting_fact"] += 1
                continue
            title, sent_id = str(fact[0]), fact[1]
            offsets = sentence_offsets.get(title)
            if offsets is None:
                skipped["supporting_fact_without_context"] += 1
                continue
            try:
                start, end = offsets[int(sent_id)]
            except (ValueError, TypeError, IndexError):
                skipped["supporting_fact_out_of_range"] += 1
                continue

            doc_id = self.make_doc_id(title)
            spans.append(
                SupportingSpan(
                    doc_id=doc_id,
                    start_char=start,
                    end_char=end,
                    text=documents[doc_id].text[start:end],
                )
            )
            if title not in gold_titles:
                gold_titles.append(title)

        if not spans:
            skipped["no_resolvable_supporting_facts"] += 1
            return None

        if self.multi_hop_only and len(gold_titles) < 2:
            skipped["single_document_evidence"] += 1
            return None

        distractors = [
            self.make_doc_id(t) for t in titles_in_context if t not in gold_titles
        ]

        return QuestionRecord(
            question_id=self.make_question_id(item_id),
            corpus_id=self.corpus_id,
            question=question_text,
            answerability=Answerability.ANSWERABLE,
            answers=[answer],
            supporting_spans=spans,
            # Every gold document is needed: that is what makes it multi-hop,
            # and it is what makes single-document ablation a valid hard case.
            evidence_mode=(
                EvidenceMode.ALL_REQUIRED if len(gold_titles) > 1 else EvidenceMode.ANY_SUFFICIENT
            ),
            question_type=(
                QuestionType.MULTI_HOP if len(gold_titles) > 1 else QuestionType.FACTOID
            ),
            hops=len(gold_titles),
            source=SourceInfo(
                dataset=self.name,
                split=split,
                item_id=item_id,
                license=self.license_spdx,
            ),
            split=split,
            metadata={
                "hotpot_type": str(item.get("type", "")),
                "level": str(item.get("level", "")),
                "gold_doc_ids": [self.make_doc_id(t) for t in gold_titles],
                "distractor_doc_ids": distractors,
            },
        )


def _join_sentences(sentences: object) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate a paragraph's sentences, returning text and per-sentence spans.

    HotpotQA sentences usually carry their own leading space, so they are joined
    verbatim rather than with an inserted separator; that keeps the offsets
    aligned with the original text.
    """
    if not isinstance(sentences, list | tuple):
        return "", []

    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for sentence in sentences:
        text = str(sentence)
        start = cursor
        end = start + len(text)
        offsets.append((start, end))
        pieces.append(text)
        cursor = end

    joined = "".join(pieces)
    # Drop trailing empty sentences that would produce zero-length spans.
    offsets = [(s, e) for s, e in offsets if e > s]
    return joined, offsets
