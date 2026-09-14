#!/usr/bin/env python
"""Extra within-document localisers: a cross-encoder, a document-local-IDF BM25, and
an RRF hybrid of BM25 with a dense model (from stored vectors).

    python localisation_extra.py --dataset qasper --raw ... --split dev --limit 300 \
        --dense-key bge --chroma data/build/index_qasper_dev_300_emb_bge:exp_qasper_dev \
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 --out extra_qasper.json
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
from localisation_probe import (  # noqa: E402
    EXTRA_EMBEDDERS,
    overlap,
    random_expectations,
    rank_stats,
)
from run_bm25_baseline import BM25, tokenize  # noqa: E402

from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402
from src.rag.embedders import EMBEDDERS  # noqa: E402


def bm25_subset(bm, qt, idxs):
    out = np.zeros(len(idxs))
    for pos, i in enumerate(idxs):
        s, freq = 0.0, bm.freqs[i]
        for t in qt:
            idf, f = bm.idf.get(t), freq.get(t)
            if idf is None or not f:
                continue
            denom = f + bm.k1 * (1 - bm.b + bm.b * bm.lengths[i] / bm.avg_len)
            s += idf * f * (bm.k1 + 1) / denom
        out[pos] = s
    return out


def rrf(orders: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for order in orders:
        for r, idx in enumerate(order, 1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda i: -score[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--dense-key", default="bge")
    ap.add_argument("--chroma", default="", help="index_dir:collection with stored chunk vectors")
    ap.add_argument("--dense-npy", default="", help="<key>.npy chunk vectors written by localisation_probe.py")
    ap.add_argument("--cross-encoder", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    loaded = get_loader(args.dataset).load(Path(args.raw), split=args.split, limit=args.limit)
    chunks, stats = chunk_documents(loaded.documents, DocumentChunker(args.chunk_size, args.chunk_overlap))
    assert stats.offset_mismatches == 0
    doc_idx: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_idx.setdefault(c.doc_id, []).append(i)

    questions = []
    for q in loaded.questions:
        if not q.is_answerable or not q.supporting_spans:
            continue
        spans = [s for s in q.supporting_spans if s.doc_id in doc_idx]
        if not spans:
            continue
        d = sorted({s.doc_id for s in spans})[0]
        idxs = doc_idx[d]
        gold = {j for j, ci in enumerate(idxs)
                if any(overlap(chunks[ci].start_char, chunks[ci].end_char, s.start_char, s.end_char) > 0
                       for s in spans if s.doc_id == d)}
        questions.append((q, d, idxs, gold))
    print(f"{len(questions)} questions", flush=True)

    bm = BM25([tokenize(c.text) for c in chunks])

    # dense from stored vectors
    import chromadb
    from sentence_transformers import SentenceTransformer
    spec = EMBEDDERS.get(args.dense_key) or EXTRA_EMBEDDERS[args.dense_key]
    if args.dense_npy:
        emb = np.load(args.dense_npy).astype(np.float32)
        assert emb.shape[0] == len(chunks)
    else:
        idx_dir, coll = args.chroma.split(":", 1)
        col = chromadb.PersistentClient(path=str(REPO / idx_dir)).get_collection(coll)
        got = col.get(include=["embeddings"])
        by_id = dict(zip(got["ids"], got["embeddings"], strict=True))
        emb = np.asarray([by_id[c.chunk_id] for c in chunks], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    model = SentenceTransformer(spec.repo_id, device="cpu")
    qemb = model.encode([spec.query_prefix + q.question for q, *_ in questions], batch_size=64,
                        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    print(f"dense ready {time.time() - t0:.0f}s", flush=True)

    ce = None
    if args.cross_encoder:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.cross_encoder, device="cpu", max_length=512)
        print("cross-encoder ready", flush=True)

    rows: dict[str, list[dict]] = {k: [] for k in
                                   ["bm25", "bm25_local_idf", args.dense_key, f"rrf_bm25+{args.dense_key}",
                                    "oracle_union_bm25+dense"] + (["cross_encoder"] if ce else [])}
    for n, (q, d, idxs, gold) in enumerate(questions, 1):
        D, m = len(idxs), len(gold)
        base = {"question_id": q.question_id, "doc_id": d}
        qt = tokenize(q.question)
        s_bm = bm25_subset(bm, qt, idxs)
        o_bm = list(np.argsort(-s_bm, kind="stable"))
        rows["bm25"].append({**base, **rank_stats(o_bm, gold, D, m)})

        # document-local IDF: idf computed over this document's chunks only
        local = BM25([tokenize(chunks[i].text) for i in idxs])
        s_loc = np.zeros(D)
        for pos in range(D):
            s, freq = 0.0, local.freqs[pos]
            for t in qt:
                idf, f = local.idf.get(t), freq.get(t)
                if idf is None or not f:
                    continue
                denom = f + local.k1 * (1 - local.b + local.b * local.lengths[pos] / local.avg_len)
                s += idf * f * (local.k1 + 1) / denom
            s_loc[pos] = s
        rows["bm25_local_idf"].append({**base, **rank_stats(list(np.argsort(-s_loc, kind="stable")), gold, D, m)})

        s_de = emb[idxs] @ qemb[n - 1]
        o_de = list(np.argsort(-s_de, kind="stable"))
        rows[args.dense_key].append({**base, **rank_stats(o_de, gold, D, m)})
        rows[f"rrf_bm25+{args.dense_key}"].append({**base, **rank_stats(rrf([o_bm, o_de]), gold, D, m)})
        # oracle union: hit if either ranks gold first (upper bound on a per-question selector)
        st_b, st_d = rows["bm25"][-1], rows[args.dense_key][-1]
        best_rank = min(x for x in (st_b["rank"], st_d["rank"]) if x) if (st_b["rank"] or st_d["rank"]) else None
        rows["oracle_union_bm25+dense"].append({**base, "D": D, "m": m, "rank": best_rank,
                                                "rr": 1 / best_rank if best_rank else 0.0,
                                                "hit@1": bool(best_rank and best_rank <= 1),
                                                "hit@3": bool(best_rank and best_rank <= 3),
                                                "hit@5": bool(best_rank and best_rank <= 5)})
        if ce:
            s_ce = np.asarray(ce.predict([(q.question, chunks[i].text) for i in idxs], batch_size=32,
                                         show_progress_bar=False))
            rows["cross_encoder"].append({**base, **rank_stats(list(np.argsort(-s_ce, kind="stable")), gold, D, m)})
        if n % 50 == 0:
            print(f"  {n}/{len(questions)} {time.time() - t0:.0f}s", flush=True)

    def summ(rs):
        return {"n": len(rs), "mrr": round(float(np.mean([r["rr"] for r in rs])), 4),
                "hit@1": round(float(np.mean([r["hit@1"] for r in rs])), 4),
                "hit@3": round(float(np.mean([r["hit@3"] for r in rs])), 4),
                "hit@5": round(float(np.mean([r["hit@5"] for r in rs])), 4)}
    rnd = [random_expectations(len(idxs), len(gold)) for _, _, idxs, gold in questions]
    summary = {"dataset": args.dataset, "n": len(questions),
               "random": {k: round(float(np.mean([r[k] for r in rnd])), 4) for k in ("rr", "hit@1", "hit@3", "hit@5")},
               **{k: summ(v) for k, v in rows.items()}}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
