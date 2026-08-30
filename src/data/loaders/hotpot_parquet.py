"""HotpotQA loader for the HuggingFace parquet distribution (distractor setting).

HotpotQA is the only selected dataset with genuine multi-hop questions, which
makes it the only one that exercises `evidence_mode = all_required` — the case
where retrieving one of two required documents is a *retrieval* failure even
though the retriever looks half right. Neither NQ nor QASPER produces such a
question, so without this loader that path stays unit-tested but never
empirically demonstrated.

Structure of the distractor setting: each item ships ten paragraphs, of which
two are gold and eight are distractors. Supporting facts identify individual
sentences by (title, sentence index), so evidence is finer-grained here than in
either primary dataset.

Two decisions:

**Sentences are joined verbatim.** HotpotQA sentences usually carry their own
leading space, so inserting a separator would shift every character offset
relative to the text a reader sees. Joining as-is keeps sentence offsets exact.

**Distractors stay in the corpus.** They are the point of the distractor
setting: removing them would make retrieval trivially easy and destroy the
difficulty the dataset exists to provide.

Known weakness of this dataset, recorded because it affects interpretation:
crowdworkers wrote the questions while looking at the paragraphs, so lexical
anchoring is present and retrieval looks easier than it would on naturally
occurring queries. Results from this subset should be reported separately
rather than pooled with NQ and QASPER.
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


@register_loader
class HotpotQaParquetLoader(DatasetLoader):
    """Loads the HuggingFace parquet distribution of HotpotQA (distractor)."""

    name = "hotpotqa"
    license_spdx = "CC-BY-SA-4.0"
    source_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"

    def __init__(self, corpus_id: str | None = None, multi_hop_only: bool = True) -> None:
        super().__init__(corpus_id=corpus_id)
        #: Keep only items whose evidence genuinely spans two or more documents.
        #: The subset exists to supply multi-hop difficulty.
        self.multi_hop_only = multi_hop_only

    def load(self, path: Path, limit: int | None = None, split: str = "validation") -> LoadResult:
        import pyarrow.parquet as pq

        questions: list[QuestionRecord] = []
        documents: dict[str, Document] = {}
        skipped: Counter[str] = Counter()

        parquet = pq.ParquetFile(str(path))
        for batch in parquet.iter_batches(batch_size=64):
            for item in batch.to_pylist():
                record = self._build(item, documents, split, skipped)
                if record is not None:
                    questions.append(record)
                    if limit is not None and len(questions) >= limit:
                        return self._result(questions, documents, skipped)
        return self._result(questions, documents, skipped)

    def _result(self, questions, documents, skipped) -> LoadResult:
        """Keep gold and distractor documents for the questions actually kept."""
        used = {span.doc_id for q in questions for span in q.supporting_spans}
        for question in questions:
            used.update(question.metadata.get("distractor_doc_ids", []))
        return LoadResult(
            questions, [d for i, d in documents.items() if i in used], self.name, dict(skipped)
        )

    def _build(
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

        context = item.get("context") or {}
        titles = context.get("title") or []
        paragraphs = context.get("sentences") or []
        if not titles:
            skipped["context_missing"] += 1
            return None

        # Build one document per context paragraph, recording sentence offsets.
        sentence_offsets: dict[str, list[tuple[int, int]]] = {}
        doc_ids_by_title: dict[str, str] = {}
        for title, sentences in zip(titles, paragraphs, strict=False):
            title = str(title)
            text, offsets = _join_sentences(sentences)
            if not text:
                continue
            doc_id = self.make_doc_id(title)
            doc_ids_by_title[title] = doc_id
            sentence_offsets[title] = offsets
            documents.setdefault(
                doc_id,
                Document(
                    doc_id=doc_id,
                    text=text,
                    title=title,
                    source=f"hotpotqa/{title}",
                    metadata={
                        "n_sentences": len(offsets),
                        "content_fingerprint": content_fingerprint(text),
                    },
                ),
            )

        facts = item.get("supporting_facts") or {}
        fact_titles = facts.get("title") or []
        fact_sent_ids = facts.get("sent_id") or []

        spans: list[SupportingSpan] = []
        gold_titles: list[str] = []
        for title, sent_id in zip(fact_titles, fact_sent_ids, strict=False):
            title = str(title)
            offsets = sentence_offsets.get(title)
            if offsets is None:
                skipped["supporting_fact_without_context"] += 1
                continue
            try:
                start, end = offsets[int(sent_id)]
            except (ValueError, TypeError, IndexError):
                skipped["supporting_fact_out_of_range"] += 1
                continue
            doc_id = doc_ids_by_title[title]
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

        multi_hop = len(gold_titles) > 1
        item_id = str(item.get("id", "")) or content_fingerprint(question_text)
        return QuestionRecord(
            question_id=self.make_question_id(item_id),
            corpus_id=self.corpus_id,
            question=question_text,
            answerability=Answerability.ANSWERABLE,
            answers=[answer],
            supporting_spans=spans,
            # Every gold document is needed: that is what makes it multi-hop.
            evidence_mode=(
                EvidenceMode.ALL_REQUIRED if multi_hop else EvidenceMode.ANY_SUFFICIENT
            ),
            question_type=QuestionType.MULTI_HOP if multi_hop else QuestionType.FACTOID,
            hops=len(gold_titles),
            source=SourceInfo(
                dataset=self.name, split=split, item_id=item_id, license=self.license_spdx
            ),
            split=split,
            metadata={
                "hotpot_type": str(item.get("type", "")),
                "level": str(item.get("level", "")),
                "gold_doc_ids": [doc_ids_by_title[t] for t in gold_titles],
                "distractor_doc_ids": [
                    doc_id for title, doc_id in doc_ids_by_title.items()
                    if title not in gold_titles
                ],
            },
        )


def _join_sentences(sentences: object) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate a paragraph's sentences, returning text and per-sentence spans.

    Joined verbatim: HotpotQA sentences usually carry their own leading space,
    so inserting a separator would shift every offset relative to the text a
    reader actually sees.
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
        # Zero-length sentences would produce an invalid span.
        if end > start:
            offsets.append((start, end))
        else:
            offsets.append((start, start + 1))
        pieces.append(text)
        cursor = end
    return "".join(pieces), offsets
