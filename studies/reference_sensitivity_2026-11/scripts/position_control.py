#!/usr/bin/env python
"""Position-matched random-gold control (complements control G of stat_audit.py).

Control G adds to the first annotator's gold, on every question where the union adds chunks, the same number of
chunks drawn uniformly from the document's non-union chunks. Other annotators' evidence is often close to the first
annotator's, and neighbouring chunks share text (32-token overlap) and topic, so part of the real-versus-random
contrast could reflect proximity alone. This control matches position: every real added chunk a is replaced by a
random non-union chunk at the same distance d(a) = min_f |a - f| (in chunk positions) from the first annotator's
gold chunks f, drawn without replacement within the question; when no unused non-union chunk exists at exactly
that distance, the nearest available distance is used (counted as a fallback).

Uncertainty, as for control G: mean and central 95% range of the random gain over 200 draws (seed 3). In addition,
a cluster bootstrap 95% interval (4,000 resamples of clusters, seed 0; QASPER paper, NQ Wikipedia page) of the
difference between the real gain and the mean position-matched gain, computed per question.

Self-check: the uniform control is recomputed here by calling stat_audit.random_gold_control and must equal the
values stored in results/audit/statistical_audit.json.

    python scripts/position_control.py   ->  results/audit/position_control.json
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "scripts"))
import analyse_v2 as A  # noqa: E402  (reference construction and cluster key, shared on purpose)
import stat_audit as S  # noqa: E402  (first_rank and the uniform control, for the self-check)
from cluster_inference import page_key  # noqa: E402  (the clustering unit of the primary analysis)

N_REP, SEED, N_BOOT = 200, 3, 4000


def items_for(raw):
    corpus = raw["corpus"]
    out = []
    for q in raw["questions"]:
        f, u = A.gold_for(q, corpus, "first"), A.gold_for(q, corpus, "union")
        if f and u and u != f:
            out.append((q, f, u))
    return out


def dist(c, f):
    return min(abs(c - x) for x in f)


def matched_draw(q, f, u, rng):
    """Random non-union chunks, one per real added chunk, at the same distance from the first annotator's gold."""
    pool = [c for c in range(q["D"]) if c not in u]
    by_d: dict[int, list[int]] = {}
    for c in pool:
        by_d.setdefault(dist(c, f), []).append(c)
    used, extra, dev = set(), set(), []
    for a in sorted(u - f):
        d = dist(a, f)
        cand = [c for c in by_d.get(d, []) if c not in used]
        if not cand:
            avail = [c for c in pool if c not in used]
            if not avail:
                break
            best = min(abs(dist(c, f) - d) for c in avail)
            cand = [c for c in avail if abs(dist(c, f) - d) == best]
        c = rng.choice(cand)
        dev.append(abs(dist(c, f) - d))
        used.add(c)
        extra.add(c)
    return extra, dev


def control(raw):
    corpus = raw["corpus"]
    models = list(raw["config"]["models"])
    items = items_for(raw)
    keys = [page_key(corpus, q["doc_id"]) for q, _, _ in items]
    added_d = [dist(a, f) for q, f, u in items for a in (u - f)]
    out = {"n_questions_union_adds_chunks": len(items), "n_added_chunks": len(added_d),
           "added_chunk_distance": {"adjacent_share": float(np.mean([d == 1 for d in added_d])),
                                    "within_2_share": float(np.mean([d <= 2 for d in added_d])),
                                    "median": float(np.median(added_d))},
           "models": {}}
    rng = random.Random(SEED)
    devs = []
    for m in models:
        real_q = np.array([int(S.first_rank(q["orders"][m], u) == 1) - int(S.first_rank(q["orders"][m], f) == 1)
                           for q, f, u in items], dtype=float)
        rand_q = np.zeros(len(items))
        rand_means = []
        for _ in range(N_REP):
            g = np.zeros(len(items))
            for i, (q, f, u) in enumerate(items):
                extra, dv = matched_draw(q, f, u, rng)
                devs += dv
                g[i] = int(S.first_rank(q["orders"][m], f | extra) == 1) - int(S.first_rank(q["orders"][m], f) == 1)
            rand_q += g
            rand_means.append(g.mean())
        rand_q /= N_REP
        diff = real_q - rand_q
        # cluster bootstrap of the mean per-question difference real - matched
        uk = sorted(set(keys))
        idx = {k: i for i, k in enumerate(uk)}
        sums = np.zeros(len(uk)); sizes = np.zeros(len(uk))
        for k, v in zip(keys, diff):
            sums[idx[k]] += v; sizes[idx[k]] += 1
        brng = np.random.default_rng(0)
        boots = []
        for _ in range(N_BOOT):
            s = brng.integers(0, len(uk), len(uk))
            boots.append(sums[s].sum() / sizes[s].sum())
        mean_r = float(np.mean(rand_means))
        out["models"][m] = {
            "gain_real_union": float(real_q.mean()),
            "gain_matched_mean": mean_r,
            "gain_matched_95": [float(np.percentile(rand_means, 2.5)), float(np.percentile(rand_means, 97.5))],
            "ratio_real_to_matched": float(real_q.mean() / mean_r) if mean_r else None,
            "real_minus_matched": float(diff.mean()),
            "real_minus_matched_cluster95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    devs = np.array(devs)
    out["matching"] = {"exact_distance_share": float(np.mean(devs == 0)), "within_1_chunk_share": float(np.mean(devs <= 1)),
                       "within_2_chunks_share": float(np.mean(devs <= 2)), "mean_abs_distance_deviation": float(devs.mean())}
    return out


def main():
    aud = json.loads((REV / "results/audit/statistical_audit.json").read_text(encoding="utf-8"))
    report = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "n_draws": N_REP, "seed": SEED, "n_boot": N_BOOT}
    for c in ("qasper", "nq"):
        raw = json.load(gzip.open(REV / f"results/raw/{c}_full.json.gz", "rt", encoding="utf-8"))
        uni = S.random_gold_control(raw)
        assert uni == aud[c]["G_random_gold_control"] or json.loads(json.dumps(uni)) == aud[c]["G_random_gold_control"], \
            f"{c}: uniform control does not reproduce statistical_audit.json"
        report[c] = {"uniform_control_reproduced": True, **control(raw)}
        print(c, json.dumps({m: (round(v["ratio_real_to_matched"], 2), [round(x, 3) for x in v["real_minus_matched_cluster95"]])
                             for m, v in report[c]["models"].items()}), report[c]["matching"], flush=True)
    dst = REV / "results/audit/position_control.json"
    dst.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
