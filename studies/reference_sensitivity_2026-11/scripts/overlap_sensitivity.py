#!/usr/bin/env python
"""Sensitivity of the reference effects to the chunk-evidence overlap rule (final audit, 25 Nov 2026).

The study counts a chunk as gold if it overlaps an evidence span by >= 1 character. Two
alternative rules were fixed before any result was computed (DECISIONS.md, D14):
  * min50   - overlap >= 50 characters (removes boundary slivers: the chunker's 32-token overlap
              means a span that only touches the end of one chunk also hits the next one);
  * prop50  - overlap >= 50% of the shorter of the evidence span and the chunk.
No other rule was tried. The model orderings are unchanged (they do not depend on the gold), so
only the gold sets are rebuilt, from the character spans stored for every annotator.

For each rule: questions retained, hit@1 under first and union, union - first per model with the
page/paper-cluster bootstrap interval and permutation p (cluster_inference.py), the neural-minus-BM25
excess, and the significant model pairs (cluster permutation, Holm) under first and union.

    python scripts/overlap_sensitivity.py     # writes results/audit/overlap_sensitivity.json
"""
from __future__ import annotations

import copy
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

ORIGINAL_GOLD = A.gold_chunks


def rule_fn(rule: str):
    def ov(c0, c1, s0, s1):
        return min(c1, s1) - max(c0, s0)

    if rule == "any":
        return lambda c0, c1, s0, s1: ov(c0, c1, s0, s1) > 0
    if rule == "min50":
        return lambda c0, c1, s0, s1: ov(c0, c1, s0, s1) >= 50
    if rule == "prop50":
        return lambda c0, c1, s0, s1: ov(c0, c1, s0, s1) > 0 and ov(c0, c1, s0, s1) >= 0.5 * min(c1 - c0, s1 - s0)
    raise ValueError(rule)


def make_gold_chunks(rule: str):
    f = rule_fn(rule)

    def gold_chunks(chunk_ranges, spans):
        spans = list(spans)
        return {i for i, (c0, c1) in enumerate(chunk_ranges) if any(f(c0, c1, s0, s1) for s0, s1 in spans)}
    return gold_chunks


def apply_rule(raw: dict, rule: str) -> dict:
    data = copy.deepcopy(raw)
    gc = make_gold_chunks(rule)
    A.gold_chunks = gc  # QASPER gold is rebuilt from spans inside analyse_v2.gold_for
    if data["corpus"] == "nq":
        for q in data["questions"]:
            q["gold_by_annotator"] = {str(a["annotator"]): sorted(gc(q["chunk_ranges"], [(a["start_char"], a["end_char"])]))
                                      for a in q["annotations"] if a.get("status") == "ok"}
    return data


def main() -> int:
    out = {"rules": {"any": ">= 1 character (study)", "min50": ">= 50 characters",
                     "prop50": ">= 50% of the shorter of span and chunk"}}
    for corpus, f in (("qasper", "qasper_full"), ("nq", "nq_full")):
        raw = json.load(gzip.open(REV / f"results/raw/{f}.json.gz", "rt", encoding="utf-8"))
        models = list(raw["config"]["models"])
        res = {}
        for rule in ("any", "min50", "prop50"):
            data = apply_rule(raw, rule)
            rng = np.random.default_rng(1)
            rf = {r["qid"]: r for r in A.rows_for(data, models, "first")}
            ru = {r["qid"]: r for r in A.rows_for(data, models, "union")}
            common = [q for q in rf if q in ru]
            kp = [CI.page_key(corpus, rf[q]["doc"]) for q in common]
            kd = [rf[q]["doc"] for q in common]
            eff, gap = {}, {}
            for m in models:
                d = np.array([int(ru[q]["rank"][m] == 1) - int(rf[q]["rank"][m] == 1) for q in common], dtype=float)
                eff[m] = CI.compare(kp, kd, d, rng)
            for m in models:
                if m == "bm25":
                    continue
                d = np.array([(int(ru[q]["rank"][m] == 1) - int(rf[q]["rank"][m] == 1))
                              - (int(ru[q]["rank"]["bm25"] == 1) - int(rf[q]["rank"]["bm25"] == 1)) for q in common], dtype=float)
                gap[m] = CI.compare(kp, kd, d, rng)
            for blk in (eff, gap):
                for m, adj in zip(list(blk), CI.holm([blk[m]["p_perm"] for m in blk]), strict=True):
                    blk[m]["p_perm_holm"] = adj
            sig = {s: CI.pairwise_block(list(rows.values()), models, corpus, rng)["significant_primary"]
                   for s, rows in (("first", rf), ("union", ru))}
            res[rule] = {
                "n_first": len(rf), "n_union": len(ru), "n_common": len(common),
                "mean_gold_first": float(np.mean([r["m"] for r in rf.values()])),
                "mean_gold_union": float(np.mean([r["m"] for r in ru.values()])),
                "hit1_first": {m: float(np.mean([r["rank"][m] == 1 for r in rf.values()])) for m in models},
                "hit1_union": {m: float(np.mean([r["rank"][m] == 1 for r in ru.values()])) for m in models},
                "union_minus_first": eff, "neural_minus_bm25": gap,
                "significant_first": sig["first"], "significant_union": sig["union"],
                "significance_changed": sorted(set(sig["first"]) ^ set(sig["union"])),
            }
            A.gold_chunks = ORIGINAL_GOLD
            print(corpus, rule, res[rule]["n_common"],
                  {m: round(100 * e["delta"], 1) for m, e in eff.items()}, res[rule]["significance_changed"], flush=True)
        out[corpus] = res
    dst = REV / "results/audit/overlap_sensitivity.json"
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
