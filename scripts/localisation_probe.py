#!/usr/bin/env python
"""Reach vs localisation probe.

For every answerable question whose gold document is in the corpus, rank *all* chunks
of that document against the question under each retriever, and record the
within-document rank of the first chunk that overlaps a gold span. This isolates
localisation (finding the passage inside the document) from reach (finding the
document among others), because the ranking is restricted to the gold document and
therefore cannot be affected by competition from other documents.

Also computes the global top-k picture (reach A, span coverage C, C|A, chunks admitted
from the gold document) for every retriever by brute-force ranking over the whole
corpus, so the two views can be compared on identical questions.

    python localisation_probe.py --dataset qasper --raw data/raw/qasper-dev-v0.3.json \
        --split dev --limit 300 --out probe_qasper.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_bm25_baseline import BM25, tokenize  # noqa: E402

from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402
from src.rag.embedders import EMBEDDERS, EmbedderSpec  # noqa: E402

EXTRA_EMBEDDERS = {
    "bge_base": EmbedderSpec(key="bge_base", repo_id="BAAI/bge-base-en-v1.5", dimension=768,
                             license_spdx="MIT",
                             query_prefix="Represent this sentence for searching relevant passages: ",
                             normalize=True, family="BAAI BGE"),
    "bge_large": EmbedderSpec(key="bge_large", repo_id="BAAI/bge-large-en-v1.5", dimension=1024,
                              license_spdx="MIT",
                              query_prefix="Represent this sentence for searching relevant passages: ",
                              normalize=True, family="BAAI BGE"),
}
CROSS_ENCODERS = {
    "ce_msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ce_bge_reranker_base": "BAAI/bge-reranker-base",
}


def overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def random_expectations(D: int, m: int, ks=(1, 3, 5)) -> dict:
    """Expected first-gold reciprocal rank and hit@k when chunk order is random."""
    if m <= 0 or D <= 0:
        return {"rr": 0.0, **{f"hit@{k}": 0.0 for k in ks}}
    total = comb(D, m)
    rr = 0.0
    for r in range(1, D - m + 2):
        p = comb(D - r, m - 1) / total
        rr += p / r
    hits = {f"hit@{k}": 1 - comb(D - m, min(k, D)) / comb(D, min(k, D)) for k in ks}
    return {"rr": rr, **hits}


def rank_stats(order: list[int], gold: set[int], D: int, m: int) -> dict:
    """order: chunk indices (within the doc) sorted best-first."""
    r = next((i + 1 for i, idx in enumerate(order) if idx in gold), None)
    return {
        "D": D, "m": m, "rank": r,
        "rr": (1.0 / r) if r else 0.0,
        "hit@1": bool(r and r <= 1), "hit@3": bool(r and r <= 3), "hit@5": bool(r and r <= 5),
        "norm_rank": ((r - 1) / max(1, D - 1)) if r else 1.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--embedders", default="minilm,mpnet,bge,e5,bge_base")
    ap.add_argument("--chroma-map", default="",
                    help="key=index_dir:collection,... : reuse stored chunk embeddings for these keys")
    ap.add_argument("--cross-encoders", default="ce_msmarco_minilm")
    ap.add_argument("--save-embeddings", default="",
                    help="directory to write <key>.npy chunk vectors (chunk order) for reuse")
    ap.add_argument("--load-embeddings", default="",
                    help="directory holding <key>.npy chunk vectors written by --save-embeddings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    loader = get_loader(args.dataset)
    loaded = loader.load(Path(args.raw), split=args.split, limit=args.limit)
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks, stats = chunk_documents(loaded.documents, chunker)
    assert stats.offset_mismatches == 0, stats.offset_mismatches
    print(f"[{args.dataset}] {stats.n_documents} docs, {len(chunks)} chunks, "
          f"{time.time() - t0:.0f}s", flush=True)

    doc_chunk_idx: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_chunk_idx.setdefault(c.doc_id, []).append(i)

    # questions with at least one gold span in an indexed document
    questions = []
    for q in loaded.questions:
        if not q.is_answerable or not q.supporting_spans:
            continue
        spans = [s for s in q.supporting_spans if s.doc_id in doc_chunk_idx]
        if not spans:
            continue
        gold_docs = sorted({s.doc_id for s in spans})
        # within-document gold chunk set, per gold document
        per_doc = {}
        for d in gold_docs:
            idxs = doc_chunk_idx[d]
            gold_local = {j for j, ci in enumerate(idxs)
                          if any(overlap(chunks[ci].start_char, chunks[ci].end_char,
                                         s.start_char, s.end_char) > 0
                                 for s in spans if s.doc_id == d)}
            per_doc[d] = gold_local
        questions.append((q, spans, gold_docs, per_doc))
    print(f"  {len(questions)} answerable questions with locatable gold spans", flush=True)

    # ---- retrievers: a scorer returns scores for a list of chunk indices ----
    scorers = {}

    bm25 = BM25([tokenize(c.text) for c in chunks])

    def bm25_scores(query: str, idxs: list[int]) -> np.ndarray:
        qt = tokenize(query)
        out = np.zeros(len(idxs))
        for pos, i in enumerate(idxs):
            s = 0.0
            freq = bm25.freqs[i]
            for term in qt:
                idf = bm25.idf.get(term)
                f = freq.get(term)
                if idf is None or not f:
                    continue
                denom = f + bm25.k1 * (1 - bm25.b + bm25.b * bm25.lengths[i] / bm25.avg_len)
                s += idf * f * (bm25.k1 + 1) / denom
            out[pos] = s
        return out

    scorers["bm25"] = ("lexical", bm25_scores)

    from sentence_transformers import SentenceTransformer

    chroma_map = {}
    for item in [x for x in args.chroma_map.split(",") if x]:
        key, rest = item.split("=", 1)
        idx_dir, coll = rest.split(":", 1)
        chroma_map[key] = (idx_dir, coll)

    for key in [k for k in args.embedders.split(",") if k]:
        spec = EMBEDDERS.get(key) or EXTRA_EMBEDDERS[key]
        t1 = time.time()
        model = SentenceTransformer(spec.repo_id, device="cpu")
        npy = Path(args.load_embeddings) / f"{key}.npy" if args.load_embeddings else None
        if npy is not None and npy.exists():
            emb = np.load(npy).astype(np.float32)
            assert emb.shape[0] == len(chunks), f"{npy}: {emb.shape[0]} rows for {len(chunks)} chunks"
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            print(f"  loaded {len(chunks)} chunk vectors for {key} from {npy.as_posix()}", flush=True)
        elif key in chroma_map:
            import chromadb
            idx_dir, coll = chroma_map[key]
            col = chromadb.PersistentClient(path=str(REPO / idx_dir)).get_collection(coll)
            got = col.get(include=["metadatas", "embeddings"])
            by_id = {i: (np.asarray(e, dtype=np.float32), m) for i, e, m in
                     zip(got["ids"], got["embeddings"], got["metadatas"], strict=True)}
            emb = np.zeros((len(chunks), len(next(iter(by_id.values()))[0])), dtype=np.float32)
            bad = 0
            for i, c in enumerate(chunks):
                e, m = by_id[c.chunk_id]
                if int(m["start_char"]) != c.start_char or int(m["end_char"]) != c.end_char:
                    bad += 1
                emb[i] = e
            assert bad == 0, f"{bad} offset mismatches against stored index {idx_dir}"
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            print(f"  loaded {len(chunks)} stored embeddings for {key} from {idx_dir}", flush=True)
        else:
            emb = model.encode([spec.passage_prefix + c.text for c in chunks], batch_size=64,
                               convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
        if args.save_embeddings:
            Path(args.save_embeddings).mkdir(parents=True, exist_ok=True)
            np.save(Path(args.save_embeddings) / f"{key}.npy", emb)
        qtexts = [spec.query_prefix + q.question for q, *_ in questions]
        qemb = model.encode(qtexts, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        qmap = {q.question_id: qemb[i] for i, (q, *_) in enumerate(questions)}
        print(f"  embedded {spec.repo_id} in {time.time() - t1:.0f}s", flush=True)
        scorers[key] = ("dense", (emb, qmap))

    ce_models = {}
    for key in [k for k in args.cross_encoders.split(",") if k]:
        from sentence_transformers import CrossEncoder
        ce_models[key] = CrossEncoder(CROSS_ENCODERS[key], device="cpu", max_length=512)
        print(f"  loaded cross-encoder {CROSS_ENCODERS[key]}", flush=True)

    # ---- within-document localisation ----
    within: dict[str, list[dict]] = {k: [] for k in list(scorers) + list(ce_models) + ["random"]}
    global_view: dict[str, list[dict]] = {k: [] for k in scorers}
    n_all = len(chunks)

    for n, (q, spans, gold_docs, per_doc) in enumerate(questions, 1):
        # within-document: evaluate on the first gold document (single-doc corpora here)
        d = gold_docs[0]
        idxs = doc_chunk_idx[d]
        D, gold_local = len(idxs), per_doc[d]
        m = len(gold_local)
        base = {"question_id": q.question_id, "doc_id": d}
        within["random"].append({**base, "D": D, "m": m, **random_expectations(D, m)})

        for key, (kind, obj) in scorers.items():
            if kind == "lexical":
                s = obj(q.question, idxs)
            else:
                emb, qmap = obj
                s = emb[idxs] @ qmap[q.question_id]
            order = list(np.argsort(-s, kind="stable"))
            within[key].append({**base, **rank_stats(order, gold_local, D, m)})

            # global top-k over the whole corpus
            if kind == "lexical":
                sg = obj(q.question, list(range(n_all)))
            else:
                sg = emb @ qmap[q.question_id]
            top = list(np.argsort(-sg, kind="stable")[: args.top_k])
            reached = any(chunks[i].doc_id in set(gold_docs) for i in top)
            admitted = [i for i in top if chunks[i].doc_id == d]
            covered = any(
                any(overlap(chunks[i].start_char, chunks[i].end_char, s_.start_char, s_.end_char) > 0
                    for s_ in spans if s_.doc_id == chunks[i].doc_id)
                for i in top)
            precise = sum(1 for i in admitted if any(
                overlap(chunks[i].start_char, chunks[i].end_char, s_.start_char, s_.end_char) > 0
                for s_ in spans if s_.doc_id == d))
            global_view[key].append({**base, "A": reached, "C": covered,
                                     "admitted": len(admitted), "admitted_gold": precise})

        for key, ce in ce_models.items():
            pairs = [(q.question, chunks[i].text) for i in idxs]
            s = np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False))
            order = list(np.argsort(-s, kind="stable"))
            within[key].append({**base, **rank_stats(order, gold_local, D, m)})

        if n % 50 == 0:
            print(f"  {n}/{len(questions)}  {time.time() - t0:.0f}s", flush=True)

    # ---- summaries ----
    def summarise_within(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "mrr": round(float(np.mean([r["rr"] for r in rows])), 4),
            "hit@1": round(float(np.mean([r["hit@1"] for r in rows])), 4),
            "hit@3": round(float(np.mean([r["hit@3"] for r in rows])), 4),
            "hit@5": round(float(np.mean([r["hit@5"] for r in rows])), 4),
            "median_rank": float(np.median([r["rank"] for r in rows if r.get("rank")])) if any(r.get("rank") for r in rows) else None,
            "mean_norm_rank": round(float(np.mean([r["norm_rank"] for r in rows if "norm_rank" in r])), 4) if "norm_rank" in rows[0] else None,
        }

    summary = {
        "dataset": args.dataset, "split": args.split, "limit": args.limit,
        "chunking": {"size": args.chunk_size, "overlap": args.chunk_overlap},
        "top_k": args.top_k,
        "n_documents": stats.n_documents, "n_chunks": len(chunks),
        "n_questions": len(questions),
        "geometry": {
            "mean_chunks_per_gold_doc": round(float(np.mean([r["D"] for r in within["random"]])), 2),
            "mean_gold_chunks_per_doc": round(float(np.mean([r["m"] for r in within["random"]])), 2),
            "median_chunks_per_gold_doc": float(np.median([r["D"] for r in within["random"]])),
        },
        "within_document": {k: summarise_within(v) for k, v in within.items()},
        "global_topk": {},
    }
    for key, rows in global_view.items():
        A = sum(r["A"] for r in rows)
        C = sum(r["C"] for r in rows)
        reached = [r for r in rows if r["A"]]
        summary["global_topk"][key] = {
            "A_reach": round(A / len(rows), 4),
            "C_span": round(C / len(rows), 4),
            "C_given_A": round(C / A, 4) if A else None,
            "mean_admitted_given_A": round(float(np.mean([r["admitted"] for r in reached])), 3) if reached else None,
            "per_chunk_precision_given_A": round(
                sum(r["admitted_gold"] for r in reached) / max(1, sum(r["admitted"] for r in reached)), 4) if reached else None,
        }

    # paired comparisons on within-document reciprocal rank and hit@1
    from src.evaluation.statistics import mcnemar_exact
    keys = [k for k in within if k != "random"]
    paired = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra = {r["question_id"]: r for r in within[a]}
            rb = {r["question_id"]: r for r in within[b]}
            ids = [q for q in ra if q in rb]
            a_only = sum(1 for q in ids if ra[q]["hit@1"] and not rb[q]["hit@1"])
            b_only = sum(1 for q in ids if rb[q]["hit@1"] and not ra[q]["hit@1"])
            diff = [ra[q]["rr"] - rb[q]["rr"] for q in ids]
            wins = sum(1 for x in diff if x > 0)
            losses = sum(1 for x in diff if x < 0)
            p_h1 = mcnemar_exact(a_only, b_only).p_value
            # exact sign test on RR differences
            nn = wins + losses
            p_sign = None
            if nn:
                kk = min(wins, losses)
                p_sign = min(1.0, 2 * sum(comb(nn, j) for j in range(kk + 1)) / 2 ** nn)
            paired[f"{a}_vs_{b}"] = {
                "n": len(ids), "hit1_only_a": a_only, "hit1_only_b": b_only,
                "mcnemar_hit1_p": p_h1, "rr_wins_a": wins, "rr_wins_b": losses,
                "sign_test_rr_p": p_sign,
                "mean_rr_diff": round(float(np.mean(diff)), 4),
            }
    summary["paired_within_document"] = paired

    out = {"summary": summary, "within_document_rows": within, "global_rows": global_view,
           "runtime_seconds": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
