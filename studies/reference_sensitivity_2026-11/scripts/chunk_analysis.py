#!/usr/bin/env python
"""Chunk-size sensitivity of the reference effects (final audit, 25 Nov 2026).

Reads results/chunk_sensitivity/{qasper300,nq300}_c{128,256,512}.json.gz (score_v2.py on the earlier
study's subsets with BM25, MiniLM-L6 and the ms-marco cross-encoder; identical settings apart from
the chunk size) and reports, per corpus and chunk size: questions, mean chunks per document, chance
hit@1, hit@1 under first and union, union - first per model (cluster bootstrap interval and
cluster permutation p, Holm over models), the neural-minus-BM25 excess, and the model pairs that
are significant (primary test) under first and union.

    python scripts/chunk_analysis.py      # writes results/audit/chunk_sensitivity.json
"""
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
import cluster_inference as CI  # noqa: E402


def main() -> int:
    out = {}
    for corpus, stem in (("qasper", "qasper300"), ("nq", "nq300")):
        out[corpus] = {}
        for cs in (128, 256, 512):
            f = REV / f"results/chunk_sensitivity/{stem}_c{cs}.json.gz"
            if not f.exists():
                print("missing", f)
                continue
            raw = json.load(gzip.open(f, "rt", encoding="utf-8"))
            models = list(raw["config"]["models"])
            rng = np.random.default_rng(1)
            rf = {r["qid"]: r for r in A.rows_for(raw, models, "first")}
            ru = {r["qid"]: r for r in A.rows_for(raw, models, "union")}
            common = [q for q in rf if q in ru]
            kp = [CI.page_key(corpus, rf[q]["doc"]) for q in common]
            kd = [rf[q]["doc"] for q in common]
            eff = {m: CI.compare(kp, kd, np.array([int(ru[q]["rank"][m] == 1) - int(rf[q]["rank"][m] == 1) for q in common], float), rng)
                   for m in models}
            gap = {m: CI.compare(kp, kd, np.array([(int(ru[q]["rank"][m] == 1) - int(rf[q]["rank"][m] == 1))
                                                   - (int(ru[q]["rank"]["bm25"] == 1) - int(rf[q]["rank"]["bm25"] == 1)) for q in common], float), rng)
                   for m in models if m != "bm25"}
            for blk in (eff, gap):
                for m, adj in zip(list(blk), CI.holm([blk[m]["p_perm"] for m in blk]), strict=True):
                    blk[m]["p_perm_holm"] = adj
            sig = {s: CI.pairwise_block(list(rows.values()), models, corpus, rng)["significant_primary"] for s, rows in (("first", rf), ("union", ru))}
            out[corpus][str(cs)] = {
                "n_common": len(common), "mean_chunks_per_doc": float(np.mean([q["D"] for q in raw["questions"]])),
                "chance_hit1_first": float(A.chance(list(rf.values()))["hit@1"]),
                "chance_hit1_union": float(A.chance(list(ru.values()))["hit@1"]),
                "hit1_first": {m: float(np.mean([r["rank"][m] == 1 for r in rf.values()])) for m in models},
                "hit1_union": {m: float(np.mean([r["rank"][m] == 1 for r in ru.values()])) for m in models},
                "union_minus_first": eff, "neural_minus_bm25": gap,
                "significant_first": sig["first"], "significant_union": sig["union"],
                "significance_changed": sorted(set(sig["first"]) ^ set(sig["union"])),
                "config": {k: raw["config"][k] for k in ("chunk_size", "chunk_overlap", "ce_max_length", "n_chunks", "limit_per_file")},
            }
            print(corpus, cs, len(common), {m: round(100 * e["delta"], 1) for m, e in eff.items()},
                  {m: round(100 * g["delta"], 1) for m, g in gap.items()}, out[corpus][str(cs)]["significance_changed"], flush=True)
    (REV / "results/audit/chunk_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
