"""2WikiMultihopQA loader for the HuggingFace parquet distribution.

Why this dataset. The quantifier effect — a metric counting *a* relevant
document as success when the question needs *all* of them — was measured on
HotpotQA alone, which made it a property of one corpus rather than of multi-hop
questions. 2WikiMultihopQA is the natural replication target because it is
structurally comparable: ten context paragraphs per item, two or more of them
gold, and supporting facts identified as (title, sentence index). That means
the *same* evidence-alignment and A/B/C code runs over it unchanged, so a
difference in the result is a difference in the data rather than in the method.

It is not a duplicate of HotpotQA. The questions are generated from Wikidata
relation paths rather than written freehand by crowdworkers, and the released
`evidences` field states the (subject, relation, object) triples the answer
rests on. Question construction therefore has a different bias profile: less of
HotpotQA's lexical anchoring from workers copying phrasing out of the
paragraphs, more templated phrasing. Agreement across the two is worth more
than agreement across two crowdsourced sets would be.

Format note. This mirror stores the nested fields as JSON-encoded *strings*
rather than as native parquet structs, so every one of `context`,
`supporting_facts` and `evidences` has to be parsed. The native shape also
differs from HotpotQA's: context is a list of `[title, [sentences]]` pairs
instead of two parallel arrays. Both are handled here rather than by
pre-processing the file, so the committed pipeline reads the file exactly as
downloaded.

Sentences are joined verbatim, for the same reason as in the HotpotQA loader:
inserting a separator would shift every character offset relative to the text a
reader sees, and the offsets are what evidence alignment is computed from.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.identity import content_fingerprint
from src.data.loaders.base import DatasetLoader, LoadResult, register_loader
from src.data.loaders.hotpot_parquet import _join_sentences
from src.data.schema import (
    Answerability,
    Document,
    EvidenceMode,
    QuestionRecord,
    QuestionType,
    SourceInfo,
    SupportingSpan,
)


def _maybe_json(value: object) -> object:
    """Parse a field that this mirror stores as a JSON string.

    Returned unchanged when it is already a native list, so the loader works
    against both this distribution and any future one that keeps real structs.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


@register_loader
class TwoWikiMultihopParquetLoader(DatasetLoader):
    """Loads the HuggingFace parquet distribution of 2WikiMultihopQA."""

    name = "2wiki"
    license_spdx = "Apache-2.0"
    source_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA"

    def __init__(self, corpus_id: str | None = None, multi_hop_only: bool = True) -> None:
        super().__init__(corpus_id=corpus_id)
        #: Keep only items whose evidence genuinely spans two or more documents,
        #: matching the HotpotQA loader so the comparison stays like-for-like.
        self.multi_hop_only = multi_hop_only

    def load(self, path: Path, limit: int | None = None, split: str = "dev") -> LoadResult:
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

        context = _maybe_json(item.get("context"))
        if not isinstance(context, list) or not context:
            skipped["context_missing"] += 1
            return None

        # Context is [[title, [sentence, ...]], ...] rather than two parallel arrays.
        sentence_offsets: dict[str, list[tuple[int, int]]] = {}
        doc_ids_by_title: dict[str, str] = {}
        for entry in context:
            if not isinstance(entry, list | tuple) or len(entry) < 2:
                continue
            title = str(entry[0])
            text, offsets = _join_sentences(entry[1])
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
                    source=f"2wiki/{title}",
                    metadata={
                        "n_sentences": len(offsets),
                        "content_fingerprint": content_fingerprint(text),
                    },
                ),
            )

        if not doc_ids_by_title:
            skipped["context_unparseable"] += 1
            return None

        facts = _maybe_json(item.get("supporting_facts"))
        if not isinstance(facts, list):
            facts = []

        spans: list[SupportingSpan] = []
        gold_titles: list[str] = []
        for fact in facts:
            if not isinstance(fact, list | tuple) or len(fact) < 2:
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
        item_id = str(item.get("_id", "")) or content_fingerprint(question_text)
        evidences = _maybe_json(item.get("evidences"))
        return QuestionRecord(
            question_id=self.make_question_id(item_id),
            corpus_id=self.corpus_id,
            question=question_text,
            answerability=Answerability.ANSWERABLE,
            answers=[answer],
            supporting_spans=spans,
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
                # compositional / comparison / inference / bridge_comparison
                "twowiki_type": str(item.get("type", "")),
                "n_evidence_triples": len(evidences) if isinstance(evidences, list) else 0,
                "gold_doc_ids": [doc_ids_by_title[t] for t in gold_titles],
                "distractor_doc_ids": [
                    doc_id for title, doc_id in doc_ids_by_title.items()
                    if title not in gold_titles
                ],
            },
        )
