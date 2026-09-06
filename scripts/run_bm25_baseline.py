#!/usr/bin/env python
"""A lexical BM25 baseline, scored under the same evidence-level definitions.

The retrieval results in this project all come from one dense retriever, which
invites an obvious objection: is the document/span gap a property of embeddings
rather than of the metric definition? This runs Okapi BM25 over exactly the same
chunks, questions and depth, and scores it with the same A/B/C decomposition.

BM25 is implemented here rather than pulled in as a dependency -- it is forty lines,
and adding a package for it would make the baseline harder to reproduce, not easier.

    python scripts/run_bm25_baseline.py --dataset qasper \
        --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300 \
        --out results/bm25_qasper_dev_300.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.evaluation.statistics import mcnemar_exact, wilson_proportion_ci  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 with the usual defaults."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.lengths = [len(d) for d in corpus]
        self.avg_len = sum(self.lengths) / self.n if self.n else 0.0
        self.freqs = [Counter(d) for d in corpus]
        df: Counter[str] = Counter()
        for doc in self.freqs:
            df.update(doc.keys())
        # Robertson/Sparck-Jones idf with the +1 guard, so common terms stay positive.
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top_k(self, query: list[str], k: int) -> list[int]:
        scores = [0.0] * self.n
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                f = freq.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(range(self.n), key=lambda i: scores[i], reverse=True)[:k]


def covered(spans, chunks) -> bool:
    """Every gold span overlapped by some retrieved chunk from the same document."""
    for span in spans:
        hit = False
        for c in chunks:
            if c.doc_id == span.doc_id and min(span.end_char, c.end_char) - max(
                    span.start_char, c.start_char) > 0:
                hit = True
                break
        if not hit:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--dense-rows", default="",
                    help="rows.jsonl from the dense run, for a paired comparison")
    ap.add_argument("--dense-records", default="",
                    help="inference.jsonl matching --dense-rows; supplies the question_id "
                         "that rows.jsonl does not carry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loader = get_loader(args.dataset)
    loaded = loader.load(Path(args.raw), split=args.split, limit=args.limit)
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks, stats = chunk_documents(loaded.documents, chunker)
    if stats.offset_mismatches:
        print(f"offset mismatches: {stats.offset_mismatches} - refusing to continue")
        return 1

    index = BM25([tokenize(c.text) for c in chunks])
    print(f"indexed {len(chunks)} chunks from {stats.n_documents} documents")

    answerable = [q for q in loaded.questions if q.is_answerable]
    a_hits = b_hits = c_hits = 0
    per_question = {}
    for n, q in enumerate(answerable, 1):
        picked = [chunks[i] for i in index.top_k(tokenize(q.question), args.top_k)]
        relevant = set(q.relevant_doc_ids)
        retrieved_docs = {c.doc_id for c in picked}
        a = bool(relevant & retrieved_docs)
        b = relevant.issubset(retrieved_docs)
        c = covered(q.supporting_spans, picked)
        a_hits += a
        b_hits += b
        c_hits += c
        per_question[q.question_id] = {"A": a, "B": b, "C": c}
        if n % 50 == 0:
            print(f"  {n}/{len(answerable)}")

    n = len(answerable)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "retriever": "BM25 (Okapi, k1=1.5, b=0.75), implemented in this script",
        "n_answerable": n,
        "top_k": args.top_k,
        "chunking": {"size": args.chunk_size, "overlap": args.chunk_overlap},
        "corpus": {"documents": stats.n_documents, "chunks": stats.n_chunks,
                   "offset_mismatches": stats.offset_mismatches},
        "conditions": {
            "A_document_any": round(a_hits / n, 4),
            "B_document_quantified": round(b_hits / n, 4),
            "C_span_quantified": round(c_hits / n, 4),
        },
        "confidence_intervals": {
            "A_document_any": wilson_proportion_ci(a_hits, n).as_dict(),
            "B_document_quantified": wilson_proportion_ci(b_hits, n).as_dict(),
            "C_span_quantified": wilson_proportion_ci(c_hits, n).as_dict(),
        },
        "gaps_pp": {
            "quantifier_A_to_B": round(100 * (a_hits - b_hits) / n, 1),
            "granularity_B_to_C": round(100 * (b_hits - c_hits) / n, 1),
        },
    }

    if args.dense_rows:
        rows_list = [json.loads(line) for line
                     in Path(args.dense_rows).read_text(encoding="utf-8").splitlines() if line.strip()]
        dense = {}
        if args.dense_records:
            recs = [json.loads(line) for line
                    in Path(args.dense_records).read_text(encoding="utf-8").splitlines() if line.strip()]
            for rec, row in zip(recs, rows_list, strict=False):
                qid = (rec.get("metadata") or {}).get("question_id")
                if qid:
                    dense[qid] = row
        else:
            for row in rows_list:
                qid = row.get("question_id")
                if qid:
                    dense[qid] = row
        shared = [q for q in answerable if q.question_id in dense]
        if shared:
            only_bm25 = only_dense = 0
            for q in shared:
                d_c = dense[q.question_id].get("evidence_status") == "complete"
                b_c = per_question[q.question_id]["C"]
                only_bm25 += b_c and not d_c
                only_dense += d_c and not b_c
            report["paired_vs_dense"] = {
                "n": len(shared),
                "only_bm25_span_hit": only_bm25,
                "only_dense_span_hit": only_dense,
                "test": mcnemar_exact(only_bm25, only_dense).as_dict(),
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nA {report['conditions']['A_document_any']}  "
          f"B {report['conditions']['B_document_quantified']}  "
          f"C {report['conditions']['C_span_quantified']}  (n={n})")
    print(f"quantifier {report['gaps_pp']['quantifier_A_to_B']} pp · "
          f"granularity {report['gaps_pp']['granularity_B_to_C']} pp")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
