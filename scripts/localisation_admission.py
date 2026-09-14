#!/usr/bin/env python
"""Admission breadth at a fixed chunk budget: does span coverage rise when slots move
from more documents to more chunks per reached document?

Policies compared at the same budget k:
  flat@k        the usual top-k chunks
  docfirst(d,j) rank documents by their best chunk (MaxP), keep the top d, admit the
                top j chunks of each (d * j = k)

Reach A, span coverage C and C|A are reported per policy, so the reach/localisation
trade-off is visible directly. Uses stored chunk vectors where available.

    python scripts/localisation_admission.py --dataset qasper --raw data/raw/qasper-dev-v0.3.json \
        --split dev --limit 300 --embedders minilm,bge \
        --chroma-map "minilm=data/build/index_qasper_dev_300_emb_minilm:exp_qasper_dev,..." \
        --out results/localisation/admission_qasper.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from localisation_probe import EXTRA_EMBEDDERS, overlap  # noqa: E402
from run_bm25_baseline import BM25, tokenize  # noqa: E402

from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402
from src.rag.embedders import EMBEDDERS  # noqa: E402

BUDGETS = {6: [(6, 1), (3, 2), (2, 3), (1, 6)],
           12: [(12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12)],
           20: [(20, 1), (10, 2), (5, 4), (4, 5), (2, 10), (1, 20)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--embedders", default="minilm")
    ap.add_argument("--chroma-map", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    loaded = get_loader(args.dataset).load(Path(args.raw), split=args.split, limit=args.limit)
    chunks, stats = chunk_documents(loaded.documents, DocumentChunker(args.chunk_size, args.chunk_overlap))
    assert stats.offset_mismatches == 0
    doc_of = np.array([c.doc_id for c in chunks])
    docs = sorted(set(doc_of))
    doc_index = {d: i for i, d in enumerate(docs)}
    doc_id_arr = np.array([doc_index[d] for d in doc_of])

    questions = []
    for q in loaded.questions:
        if not q.is_answerable or not q.supporting_spans:
            continue
        spans = [s for s in q.supporting_spans if s.doc_id in doc_index]
        if spans:
            questions.append((q, spans, {s.doc_id for s in spans}))
    print(f"[{args.dataset}] {len(chunks)} chunks, {len(questions)} questions", flush=True)

    bm = BM25([tokenize(c.text) for c in chunks])

    def bm25_all(query: str) -> np.ndarray:
        qt = tokenize(query)
        out = np.zeros(len(chunks))
        for i, freq in enumerate(bm.freqs):
            s = 0.0
            for t in qt:
                idf, f = bm.idf.get(t), freq.get(t)
                if idf is None or not f:
                    continue
                s += idf * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * bm.lengths[i] / bm.avg_len))
            out[i] = s
        return out

    scorers = {"bm25": ("lexical", bm25_all)}
    chroma_map = {}
    for item in [x for x in args.chroma_map.split(",") if x]:
        key, rest = item.split("=", 1)
        chroma_map[key] = tuple(rest.split(":", 1))
    from sentence_transformers import SentenceTransformer
    for key in [k for k in args.embedders.split(",") if k]:
        spec = EMBEDDERS.get(key) or EXTRA_EMBEDDERS[key]
        model = SentenceTransformer(spec.repo_id, device="cpu")
        if key in chroma_map:
            import chromadb
            idx_dir, coll = chroma_map[key]
            got = chromadb.PersistentClient(path=str(REPO / idx_dir)).get_collection(coll).get(include=["embeddings"])
            by_id = dict(zip(got["ids"], got["embeddings"], strict=True))
            emb = np.asarray([by_id[c.chunk_id] for c in chunks], dtype=np.float32)
        else:
            emb = model.encode([spec.passage_prefix + c.text for c in chunks], batch_size=64,
                               convert_to_numpy=True, show_progress_bar=False)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        qemb = model.encode([spec.query_prefix + q.question for q, *_ in questions], batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        scorers[key] = ("dense", (emb, qemb))
        print(f"  {key} ready {time.time() - t0:.0f}s", flush=True)

    def evaluate(picked: list[int], spans, gold_docs) -> tuple[bool, bool]:
        reached = any(doc_of[i] in gold_docs for i in picked)
        covered = any(doc_of[i] == s.doc_id and overlap(chunks[i].start_char, chunks[i].end_char,
                                                       s.start_char, s.end_char) > 0
                      for i in picked for s in spans)
        return reached, covered

    results: dict[str, dict] = {}
    for key, (kind, obj) in scorers.items():
        tallies: dict[str, list[tuple[bool, bool]]] = {}
        for qi, (q, spans, gold_docs) in enumerate(questions):
            scores = obj(q.question) if kind == "lexical" else obj[0] @ obj[1][qi]
            order = np.argsort(-scores, kind="stable")
            # document ranking by best chunk (MaxP), and per-document chunk order
            doc_best: dict[int, float] = {}
            per_doc: dict[int, list[int]] = {}
            for i in order:
                d = int(doc_id_arr[i])
                if d not in doc_best:
                    doc_best[d] = float(scores[i])
                per_doc.setdefault(d, []).append(int(i))
            doc_order = sorted(doc_best, key=lambda d: -doc_best[d])
            for k, combos in BUDGETS.items():
                tallies.setdefault(f"flat@{k}", []).append(evaluate([int(i) for i in order[:k]], spans, gold_docs))
                for d_n, j in combos:
                    picked = [i for d in doc_order[:d_n] for i in per_doc[d][:j]]
                    tallies.setdefault(f"docfirst@{k}(d={d_n},j={j})", []).append(evaluate(picked, spans, gold_docs))
        results[key] = {}
        for name, tl in tallies.items():
            a = sum(x for x, _ in tl)
            c = sum(y for _, y in tl)
            results[key][name] = {"A": round(a / len(tl), 4), "C": round(c / len(tl), 4),
                                  "C_given_A": round(c / a, 4) if a else None}
        print(f"  {key} evaluated {time.time() - t0:.0f}s", flush=True)

    out = {"dataset": args.dataset, "n_questions": len(questions), "n_chunks": len(chunks),
           "chunking": {"size": args.chunk_size, "overlap": args.chunk_overlap}, "results": results}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, res in results.items():
        print(f"\n{key}")
        for name, v in res.items():
            print(f"  {name:26} A={v['A']:.3f} C={v['C']:.3f} C|A={v['C_given_A']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
