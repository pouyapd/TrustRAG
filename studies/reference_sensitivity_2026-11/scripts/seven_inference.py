#!/usr/bin/env python
"""Cluster-aware significance for the seven-model comparison on the earlier study's subsets.

Reads the repository's results/exploratory/{qasper,nq}_reference_sensitivity/per_question.json
(per-annotator / per-span gold ranks of the seven study localisers), rebuilds the rank-1 outcome
under the first-annotator and union references, checks that hit@1 reproduces
summary.json exactly, and recomputes the 21 pairwise comparisons per reference with the primary
procedure of cluster_inference.py (paper / page-cluster permutation, Holm) and with exact McNemar.

    python scripts/seven_inference.py   # writes results/audit/seven_inference.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
sys.path.insert(0, str(REV / "scripts"))
import cluster_inference as CI  # noqa: E402

EX = REPO / "results/exploratory"
SEVEN = ["bm25", "minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base"]


def first_annotator(ann):
    f = next((a for a in ann if a.get("status") in ("ok", "not_representable")), None)
    return f["annotator"] if f is not None and f["status"] == "ok" else None


def qasper_ranks(q, m):
    r = q["ranks"][m]
    union = r["union"]
    fa = first_annotator(q["annotations"])
    if fa is None:
        return union, None
    if r.get("spans") is None:  # reference-invariant question: every annotator's evidence gives the same gold
        return union, union
    idx = [i for i, s in enumerate(q["spans"]) if fa in s["annotators"]]
    ranks = [r["spans"][i] for i in idx if r["spans"][i] is not None]
    return union, (min(ranks) if ranks else None)


def nq_ranks(q, m):
    r = q["ranks"][m]
    ok = [str(a["annotator"]) for a in q["annotations"] if a.get("status") == "ok" and str(a["annotator"]) in r]
    union = min(r[a] for a in ok) if ok else None
    fa = first_annotator(q["annotations"])
    return union, (r[str(fa)] if fa is not None and str(fa) in r else None)


def main() -> int:
    out = {}
    for corpus in ("qasper", "nq"):
        pq = json.loads((EX / f"{corpus}_reference_sensitivity/per_question.json").read_text(encoding="utf-8"))["questions"]
        summ = json.loads((EX / f"{corpus}_reference_sensitivity/summary.json").read_text(encoding="utf-8"))
        fn = qasper_ranks if corpus == "qasper" else nq_ranks
        rows = {"union": [], "first": []}
        for q in pq:
            rk = {m: fn(q, m) for m in SEVEN}
            if all(v[0] is not None for v in rk.values()):
                rows["union"].append({"qid": q["question_id"], "doc": q["doc_id"], "rank": {m: rk[m][0] for m in SEVEN}})
            if all(v[1] is not None for v in rk.values()):
                rows["first"].append({"qid": q["question_id"], "doc": q["doc_id"], "rank": {m: rk[m][1] for m in SEVEN}})
        rng = np.random.default_rng(1)
        res = {}
        for s, rs in rows.items():
            check = {m: (float(np.mean([r["rank"][m] == 1 for r in rs])), summ["schemes"][s]["models"][m]["hit@1"]) for m in SEVEN}
            blk = CI.pairwise_block(rs, SEVEN, corpus, rng)
            res[s] = {"n": len(rs), "n_summary": summ["schemes"][s]["n_questions"],
                      # summary.json stores hit@1 rounded to 4 decimal places
                      "hit1_reproduced": all(abs(a - b) < 5.01e-5 for a, b in check.values()),
                      "n_sig_primary": len(blk["significant_primary"]), "n_sig_mcnemar": len(blk["significant_mcnemar"]),
                      "significant_primary": blk["significant_primary"], "significant_mcnemar": blk["significant_mcnemar"],
                      "n_clusters": blk["n_clusters"]}
            print(corpus, s, res[s]["n"], res[s]["n_summary"], res[s]["hit1_reproduced"], res[s]["n_sig_primary"], res[s]["n_sig_mcnemar"], flush=True)
        out[corpus] = res
    (REV / "results/audit/seven_inference.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
