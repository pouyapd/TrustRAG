#!/usr/bin/env python
"""Ranking-tie audit. Stored orderings break ties by document order (stable sort). For BM25,
which gives exactly equal scores to chunks with the same query-term statistics (e.g. zero
overlap), this can favour early chunks. Recomputes BM25 scores, checks they reproduce the
stored orderings, and reports, per reference, hit@1 under stable sort and the exact expected
hit@1 under uniformly random tie-breaking (gold chunks among the tied top chunks / tied top).
Neural scores are float32/float64 similarities; their exact ties are counted too where the
scores can be recomputed cheaply (not done: models would have to be rerun)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
for p in (REPO, REPO / "scripts", REV / "scripts"):
    sys.path.insert(0, str(p))
import analyse_v2 as A  # noqa: E402
from add_model import rebuild_chunks  # noqa: E402
from run_bm25_baseline import tokenize  # noqa: E402
from score_v2 import bm25_matrix, query_vector  # noqa: E402


def main():
    out = {}
    for c in ("qasper", "nq"):
        data = json.load(gzip.open(REV / f"results/raw/{c}_full.json.gz", "rt", encoding="utf-8"))
        by_doc = rebuild_chunks(data["config"], data["corpus"])
        chunks = [ch for d in data["doc_ids"] for ch in by_doc[d]]
        W, vocab = bm25_matrix([tokenize(ch.text) for ch in chunks])
        start, pos = {}, 0
        for d in data["doc_ids"]:
            start[d] = pos; pos += len(by_doc[d])
        reproduced = tied_top = 0
        res = {}
        scores_by_q = {}
        for q in data["questions"]:
            s_all = W @ query_vector(tokenize(q["question"]), vocab)
            s = s_all[start[q["doc_id"]]: start[q["doc_id"]] + q["D"]]
            reproduced += [int(i) for i in np.argsort(-s, kind="stable")] == q["orders"]["bm25"]
            top = np.flatnonzero(s == s.max())
            tied_top += len(top) > 1
            scores_by_q[q["question_id"]] = top
        for scheme in A.SCHEMES[c]:
            stable, expected, n = [], [], 0
            for q in data["questions"]:
                g = A.gold_for(q, c, scheme)
                if not g:
                    continue
                top = scores_by_q[q["question_id"]]
                stable.append(int(q["orders"]["bm25"][0] in g))
                expected.append(len(set(top.tolist()) & g) / len(top))
            res[scheme] = {"n": len(stable), "hit@1_stable_sort": float(np.mean(stable)),
                           "hit@1_random_tie_break_expected": float(np.mean(expected)),
                           "difference_points": float(100 * (np.mean(expected) - np.mean(stable)))}
        out[c] = {"questions": len(data["questions"]), "orderings_reproduced": reproduced,
                  "questions_with_tied_top_score": tied_top, "by_reference": res}
        print(c, json.dumps(out[c]), flush=True)
    dst = REV / "results/audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tie_audit.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
