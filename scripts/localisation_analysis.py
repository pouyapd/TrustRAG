#!/usr/bin/env python
"""Mechanism analyses over a localisation probe output.

    python probe_analysis.py probe_qasper.json --dataset qasper --raw ... --split dev --limit 300
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set(["the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "be", "by", "with", "as", "at", "from", "that", "this", "which", "what", "how", "do", "does", "did", "it", "its", "their", "they", "them", "we", "our", "there", "these", "those", "than", "then", "not", "no", "who", "whom", "whose", "when", "where", "why"])


def toks(s: str) -> set[str]:
    return {t for t in TOKEN.findall(s.lower()) if t not in STOP and len(t) > 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--dataset")
    ap.add_argument("--raw")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    args = ap.parse_args()

    P = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    S, W, G = P["summary"], P["within_document_rows"], P["global_rows"]
    retr = [k for k in W if k != "random"]
    print(f"\n=== {S['dataset']} : {S['n_questions']} questions, {S['n_documents']} docs, "
          f"{S['n_chunks']} chunks; mean chunks/gold-doc {S['geometry']['mean_chunks_per_gold_doc']}, "
          f"mean gold chunks/doc {S['geometry']['mean_gold_chunks_per_doc']} ===")

    # ---- 1. main table ----
    print(f"\n{'retriever':22} {'A':>6} {'C':>6} {'C|A':>6} {'adm|A':>6} {'prec|A':>7} | "
          f"{'wMRR':>6} {'wH@1':>6} {'wH@3':>6} {'wH@5':>6} {'lift@1':>7}")
    rnd = S["within_document"]["random"]
    for k in ["random"] + retr:
        w = S["within_document"][k]
        g = S["global_topk"].get(k, {})
        lift = (w["hit@1"] / rnd["hit@1"]) if rnd["hit@1"] else float("nan")
        print(f"{k:22} {g.get('A_reach', float('nan')):6.3f} {g.get('C_span', float('nan')):6.3f} "
              f"{(g.get('C_given_A') or float('nan')):6.3f} {(g.get('mean_admitted_given_A') or float('nan')):6.2f} "
              f"{(g.get('per_chunk_precision_given_A') or float('nan')):7.3f} | "
              f"{w['mrr']:6.3f} {w['hit@1']:6.3f} {w['hit@3']:6.3f} {w['hit@5']:6.3f} {lift:7.2f}")

    # ---- 2. spreads ----
    A = [S["global_topk"][k]["A_reach"] for k in retr if k in S["global_topk"]]
    CA = [S["global_topk"][k]["C_given_A"] for k in retr if k in S["global_topk"]]
    H1 = [S["within_document"][k]["hit@1"] for k in retr]
    MRR = [S["within_document"][k]["mrr"] for k in retr]
    print(f"\nspread across retrievers (max-min):  reach A {100*(max(A)-min(A)):.1f} pp | "
          f"C|A {100*(max(CA)-min(CA)):.1f} pp | within-doc hit@1 {100*(max(H1)-min(H1)):.1f} pp | "
          f"within-doc MRR {max(MRR)-min(MRR):.3f}")

    # paired tests vs bm25 on within-doc rr/hit@1 and on global A / C
    print("\npaired vs bm25 (within-doc hit@1 McNemar p; RR sign-test p; global A McNemar; global C McNemar):")
    from src.evaluation.statistics import mcnemar_exact
    wb = {r["question_id"]: r for r in W["bm25"]}
    gb = {r["question_id"]: r for r in G["bm25"]}
    for k in retr:
        if k == "bm25":
            continue
        wk = {r["question_id"]: r for r in W[k]}
        ids = [q for q in wb if q in wk]
        a_only = sum(1 for q in ids if wk[q]["hit@1"] and not wb[q]["hit@1"])
        b_only = sum(1 for q in ids if wb[q]["hit@1"] and not wk[q]["hit@1"])
        p1 = mcnemar_exact(a_only, b_only).p_value
        d = [wk[q]["rr"] - wb[q]["rr"] for q in ids]
        wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
        psign = stats.binomtest(min(wins, losses), wins + losses).pvalue if wins + losses else None
        line = f"  {k:20} wH@1 {k}-only {a_only:3d} bm25-only {b_only:3d} p={p1:.3g}; RR wins {wins}/{losses} p={psign if psign is None else round(psign,3)}"
        if k in G:
            gk = {r["question_id"]: r for r in G[k]}
            aa = sum(1 for q in ids if gk[q]["A"] and not gb[q]["A"])
            ab = sum(1 for q in ids if gb[q]["A"] and not gk[q]["A"])
            ca = sum(1 for q in ids if gk[q]["C"] and not gb[q]["C"])
            cb = sum(1 for q in ids if gb[q]["C"] and not gk[q]["C"])
            line += f" | A {aa}/{ab} p={mcnemar_exact(aa, ab).p_value:.3g} | C {ca}/{cb} p={mcnemar_exact(ca, cb).p_value:.3g}"
        print(line)

    # ---- 3. does the within-doc curve + admission count predict C|A? ----
    print("\nprediction of C|A from within-document hit@j at j = chunks admitted from the gold doc:")
    for k in retr:
        if k not in G:
            continue
        wk = {r["question_id"]: r for r in W[k]}
        reached = [r for r in G[k] if r["A"]]
        if not reached:
            continue
        pred = np.mean([1.0 if (wk[r["question_id"]]["rank"] and wk[r["question_id"]]["rank"] <= max(1, r["admitted"])) else 0.0
                        for r in reached])
        obs = np.mean([r["C"] for r in reached])
        print(f"  {k:20} observed C|A {obs:.3f}  predicted {pred:.3f}  (mean admitted {np.mean([r['admitted'] for r in reached]):.2f})")

    # ---- 4. geometry model: within-doc hit@1 ~ log D + m (+ retriever dummies) ----
    from sklearn.linear_model import LogisticRegression
    rows = []
    for k in retr:
        for r in W[k]:
            rows.append((k, math.log(r["D"]), r["m"], 1 if r["hit@1"] else 0, r["rr"]))
    ks = sorted({r[0] for r in rows})
    X0 = np.array([[r[1], r[2]] for r in rows])
    Xd = np.array([[1.0 if r[0] == k else 0.0 for k in ks[1:]] for r in rows])
    y = np.array([r[3] for r in rows])

    def deviance(X, y):
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), m
    d0, m0 = deviance(X0, y)
    d1, m1 = deviance(np.hstack([X0, Xd]), y)
    lr = d0 - d1
    p = 1 - stats.chi2.cdf(lr, df=Xd.shape[1])
    print(f"\ngeometry model (hit@1 ~ logD + m): geometry coefs {m0.coef_[0].round(3)}; "
          f"adding {Xd.shape[1]} retriever dummies: LR={lr:.2f}, df={Xd.shape[1]}, p={p:.3g}")
    dg, _ = deviance(Xd, y)
    dnull = -2 * np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))
    print(f"  deviance explained: geometry only {(dnull - d0)/dnull:.3%}; retriever only {(dnull - dg)/dnull:.3%}; both {(dnull - d1)/dnull:.3%}")

    # ---- 5. lexical distinctiveness stratification ----
    if args.dataset and args.raw:
        from run_bm25_baseline import tokenize  # noqa
        from src.data.corpus import chunk_documents
        from src.data.loaders import get_loader
        from src.rag.chunking import DocumentChunker
        loaded = get_loader(args.dataset).load(Path(args.raw), split=args.split, limit=args.limit)
        chunks, _ = chunk_documents(loaded.documents, DocumentChunker(args.chunk_size, args.chunk_overlap))
        by_doc: dict[str, list] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        qmap = {q.question_id: q for q in loaded.questions}
        # distinctiveness = overlap(question, best gold chunk) - max overlap(question, non-gold chunk)
        dist = {}
        for r in W["random"]:
            q = qmap[r["question_id"]]
            qt = toks(q.question)
            cs = by_doc[r["doc_id"]]
            gold_idx = set()
            for j, c in enumerate(cs):
                if any(s.doc_id == c.doc_id and min(s.end_char, c.end_char) - max(s.start_char, c.start_char) > 0
                       for s in q.supporting_spans):
                    gold_idx.add(j)
            ov = [len(qt & toks(c.text)) / max(1, len(qt)) for c in cs]
            g = max((ov[j] for j in gold_idx), default=0.0)
            ng = max((ov[j] for j in range(len(cs)) if j not in gold_idx), default=0.0)
            dist[r["question_id"]] = g - ng
        vals = np.array([dist[q] for q in dist])
        terc = np.quantile(vals, [1 / 3, 2 / 3])
        print(f"\nlexical distinctiveness of the gold chunk (question-term overlap gold - best non-gold): "
              f"terciles at {terc.round(3)}; share with gold NOT the most question-overlapping chunk: "
              f"{np.mean(vals <= 0):.3f}")
        print(f"{'retriever':22} {'low (gold lexically hidden)':>28} {'mid':>8} {'high':>8}   within-doc hit@1")
        for k in retr:
            wk = {r["question_id"]: r for r in W[k]}
            out = []
            for lo, hi in ((-9, terc[0]), (terc[0], terc[1]), (terc[1], 9)):
                ids = [q for q in dist if lo < dist[q] <= hi] if lo > -9 else [q for q in dist if dist[q] <= hi]
                out.append((np.mean([wk[q]["hit@1"] for q in ids]), len(ids)))
            print(f"{k:22} {out[0][0]:>20.3f} (n={out[0][1]:3d}) {out[1][0]:8.3f} {out[2][0]:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
