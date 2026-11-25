#!/usr/bin/env python
"""Independent statistical audit of the revised study.

Every check recomputes from the scored files (results/raw/*_full.json.gz) through code
paths separate from analyse_v2.py, and compares with results/*_full/analysis.json.

 A. Metric re-derivation (hit@1/5/10, MRR, median rank, chance) for every reference/model.
 B. Exact chance formula vs brute-force enumeration.
 C. Krippendorff's alpha vs the `krippendorff` package; question-level cluster bootstrap CI;
    alternative unit definition (per-question alpha, mean).
 D. McNemar vs scipy binomtest; Holm vs an independent step-down implementation.
 E. Dependence-robust alternative: paired cluster permutation test (sign-flip of
    document-level differences), compared with the McNemar/Holm conclusions.
 F. Power: power of the exact McNemar test for the observed effect and discordance, and the
    minimum detectable difference at 80% power, per corpus/reference for each pair.
 G. Random-gold control: for questions where the union adds chunks to the first annotator's
    gold, add the same number of random non-union chunks instead; the model's gain under
    real vs random extra gold separates "more targets" from "other annotators' evidence".
 H. Split sensitivity: QASPER dev vs test; NQ file 0 vs file 1.
 I. Missing data: questions excluded per reference and why.

    python stat_audit.py   ->  results/audit/statistical_audit.json and .md
"""
from __future__ import annotations

import gzip
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REV / "scripts"))
import analyse_v2 as A  # noqa: E402  (only for reference construction, shared on purpose)

NAMES = {"bm25": "BM25", "minilm": "MiniLM", "ce_msmarco_minilm": "CE"}


def load(c):
    raw = json.load(gzip.open(REV / f"results/raw/{c}_full.json.gz", "rt", encoding="utf-8"))
    ana = json.loads((REV / f"results/{c}_full/analysis.json").read_text(encoding="utf-8"))
    return raw, ana


# --------------------------------------------------------------------------- A
def first_rank(order, gold):
    for i, c in enumerate(order):
        if c in gold:
            return i + 1
    raise ValueError("no gold")


def chance_hit1(D, m):
    return m / D  # P(first chunk gold) under a random order


def check_metrics(raw, ana):
    corpus = raw["corpus"]
    models = list(raw["config"]["models"])
    worst = 0.0
    n_checked = 0
    for s in A.SCHEMES[corpus]:
        rows = []
        for q in raw["questions"]:
            g = A.gold_for(q, corpus, s)
            if g:
                rows.append((q, g))
        for m in models:
            ranks = np.array([first_rank(q["orders"][m], g) for q, g in rows])
            mine = {"hit@1": (ranks == 1).mean(), "hit@5": (ranks <= 5).mean(), "hit@10": (ranks <= 10).mean(),
                    "mrr": (1 / ranks).mean(), "median_rank": float(np.median(ranks))}
            theirs = ana["schemes"][s]["metrics"][m]
            for k, v in mine.items():
                worst = max(worst, abs(v - theirs[k])); n_checked += 1
        ch = np.mean([chance_hit1(q["D"], len(g)) for q, g in rows])
        worst = max(worst, abs(ch - ana["schemes"][s]["chance"]["hit@1"])); n_checked += 1
    return {"values_checked": n_checked, "max_abs_difference": worst}


# --------------------------------------------------------------------------- B
def check_chance():
    from localisation_probe import random_expectations
    worst = 0.0
    for D, m in [(5, 1), (7, 2), (8, 3), (10, 4), (12, 2), (9, 9)]:
        perms_rr, perms = 0.0, 0
        hits = {k: 0 for k in (1, 3, 5)}
        for gold in itertools.combinations(range(D), m):  # positions of gold chunks in a random order
            r = min(gold) + 1
            perms_rr += 1 / r; perms += 1
            for k in hits:
                hits[k] += r <= k
        exact = random_expectations(D, m, ks=(1, 3, 5))
        worst = max(worst, abs(exact["rr"] - perms_rr / perms),
                    *(abs(exact[f"hit@{k}"] - hits[k] / perms) for k in hits))
    return {"cases": 6, "max_abs_difference_vs_enumeration": worst}


# --------------------------------------------------------------------------- C
def alpha_units(raw):
    units, qidx = [], []
    for qi, q in enumerate(raw["questions"]):
        sets = [set(v) for v in q["gold_by_annotator"].values() if v]
        if len(sets) < 2:
            continue
        for c in range(q["D"]):
            units.append([int(c in s) for s in sets]); qidx.append(qi)
    return units, qidx


def check_alpha(raw, ana, n_boot=1000, seed=0):
    import krippendorff
    units, qidx = alpha_units(raw)
    maxc = max(len(u) for u in units)
    mat = np.full((maxc, len(units)), np.nan)
    for j, u in enumerate(units):
        mat[: len(u), j] = u
    pkg = float(krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal"))
    ours = A.krippendorff_binary(units)
    # cluster bootstrap over questions
    byq: dict[int, list] = {}
    for u, q in zip(units, qidx):
        byq.setdefault(q, []).append(u)
    qs = list(byq)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(qs), len(qs))
        boots.append(A.krippendorff_binary([u for i in pick for u in byq[qs[i]]]))
    # alternative unit of analysis: alpha per question, then the mean (undefined when a question
    # has no variation; those are counted separately)
    per_q = []
    undefined = 0
    for q in qs:
        a = A.krippendorff_binary(byq[q])
        if math.isnan(a):
            undefined += 1
        else:
            per_q.append(a)
    return {"alpha_analysis_json": ana["annotator_agreement"]["krippendorff_alpha_chunk_level"],
            "alpha_reimplemented": ours, "alpha_krippendorff_package": pkg,
            "abs_difference_package": abs(pkg - ours),
            "cluster_bootstrap_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_units_chunks": len(units), "n_questions": len(qs),
            "per_question_alpha_mean": float(np.mean(per_q)), "per_question_alpha_median": float(np.median(per_q)),
            "questions_alpha_undefined": undefined}


# --------------------------------------------------------------------------- D
def holm_independent(p):
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * p[i]))
        adj[i] = prev
    return adj


def check_tests(raw, ana):
    corpus = raw["corpus"]
    worst_p = worst_holm = 0.0
    for s, v in ana["schemes"].items():
        ps = []
        for x in v["pairwise"]:
            t = binomtest(min(x["only_a"], x["only_b"]), x["only_a"] + x["only_b"], 0.5) if x["only_a"] + x["only_b"] else None
            p = 1.0 if t is None else t.pvalue
            worst_p = max(worst_p, abs(p - x["p"]))
            ps.append(x["p"])
        adj = holm_independent(np.array(ps))
        worst_holm = max(worst_holm, *(abs(a - x["p_holm"]) for a, x in zip(adj, v["pairwise"])))
    return {"max_abs_difference_mcnemar_vs_scipy_binomtest": worst_p, "max_abs_difference_holm": worst_holm}


# --------------------------------------------------------------------------- E
def cluster_permutation(raw, n_perm=20000, seed=1):
    """Paired sign-flip test at document level for every pair, Holm-corrected; compare."""
    corpus = raw["corpus"]
    models = list(raw["config"]["models"])
    rng = np.random.default_rng(seed)
    out = {}
    for s in A.SCHEMES[corpus]:
        rows = A.rows_for(raw, models, s)
        res = []
        for a, b in itertools.combinations(models, 2):
            byd: dict[str, float] = {}
            for r in rows:
                byd[r["doc"]] = byd.get(r["doc"], 0) + int(r["rank"][a] == 1) - int(r["rank"][b] == 1)
            d = np.array(list(byd.values()), dtype=float)
            obs = abs(d.sum())
            flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
            null = np.abs((flips * d).sum(axis=1))
            p = (1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)
            res.append({"a": a, "b": b, "p_perm": float(p)})
        adj = holm_independent(np.array([x["p_perm"] for x in res]))
        ana_pairs = {(x["a"], x["b"]): x["significant"] for x in A_cache[corpus]["schemes"][s]["pairwise"]}
        for x, pa in zip(res, adj):
            x["p_perm_holm"] = float(pa)
            x["significant_perm"] = bool(pa < 0.05)
            x["significant_mcnemar"] = ana_pairs[(x["a"], x["b"])]
        out[s] = {"pairs": res, "conclusions_agree": all(x["significant_perm"] == x["significant_mcnemar"] for x in res)}
    return out


# --------------------------------------------------------------------------- F
def mcnemar_power(n, p10, p01, alpha=0.05, sims=4000, seed=2):
    """Simulated power of the exact McNemar test with n questions and discordance rates."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        k = rng.multinomial(n, [p10, p01, 1 - p10 - p01])
        disc = k[0] + k[1]
        if disc and binomtest(min(k[0], k[1]), disc, 0.5).pvalue < alpha:
            hits += 1
    return hits / sims


def power_analysis(raw, ana):
    corpus = raw["corpus"]
    out = {}
    for s in ("first", "union"):
        for x in ana["schemes"][s]["pairwise"]:
            n = ana["schemes"][s]["n"]
            p10, p01 = x["only_a"] / n, x["only_b"] / n
            pw = mcnemar_power(n, p10, p01, alpha=0.05 / 3)  # Bonferroni-level alpha for 3 pairs (conservative)
            # minimum detectable difference at 80% power with the observed discordance rate
            disc = p10 + p01
            # binary search for the smallest difference with >= 80% power (power is monotone in d)
            mdd = None
            if mcnemar_power(n, disc, 0.0, alpha=0.05 / 3, sims=800) >= 0.8:
                lo, hi = 0.0, disc
                for _ in range(10):
                    mid = (lo + hi) / 2
                    if mcnemar_power(n, (disc + mid) / 2, (disc - mid) / 2, alpha=0.05 / 3, sims=800) >= 0.8:
                        hi = mid
                    else:
                        lo = mid
                mdd = float(hi)
            out[f"{s}:{x['a']}-{x['b']}"] = {"n": n, "observed_difference": x["delta_hit@1"], "discordant_rate": disc,
                                             "power_at_observed": pw, "min_detectable_difference_80pct": mdd}
    return out


# --------------------------------------------------------------------------- G
def random_gold_control(raw, n_rep=200, seed=3):
    corpus = raw["corpus"]
    models = list(raw["config"]["models"])
    rng = random.Random(seed)
    items = []
    for q in raw["questions"]:
        f, u = A.gold_for(q, corpus, "first"), A.gold_for(q, corpus, "union")
        if f and u and u != f:
            items.append((q, f, u))
    out = {"n_questions_union_adds_chunks": len(items), "models": {}}
    for m in models:
        real = np.mean([int(first_rank(q["orders"][m], u) == 1) - int(first_rank(q["orders"][m], f) == 1)
                        for q, f, u in items])
        rand = []
        for _ in range(n_rep):
            tot = 0
            for q, f, u in items:
                k = len(u - f)
                pool = [c for c in range(q["D"]) if c not in u]
                extra = set(rng.sample(pool, min(k, len(pool))))
                tot += int(first_rank(q["orders"][m], f | extra) == 1) - int(first_rank(q["orders"][m], f) == 1)
            rand.append(tot / len(items))
        out["models"][m] = {"gain_real_union": float(real), "gain_random_extra_mean": float(np.mean(rand)),
                            "gain_random_extra_95": [float(np.percentile(rand, 2.5)), float(np.percentile(rand, 97.5))],
                            "ratio_real_to_random": float(real / np.mean(rand)) if np.mean(rand) else None}
    return out


# --------------------------------------------------------------------------- H
def split_sensitivity(raw):
    corpus = raw["corpus"]
    models = list(raw["config"]["models"])
    key = (lambda q: q["split"]) if corpus == "qasper" else (lambda q: "file0" if q["question_id"] in file0 else "file1")
    groups: dict[str, list] = {}
    for q in raw["questions"]:
        groups.setdefault(key(q), []).append(q)
    out = {}
    for g, qs in groups.items():
        sub = dict(raw); sub["questions"] = qs
        rf = A.rows_for(sub, models, "first"); ru = A.rows_for(sub, models, "union")
        common = {r["qid"] for r in rf} & {r["qid"] for r in ru}
        fd = {r["qid"]: r for r in rf}; ud = {r["qid"]: r for r in ru}
        out[g] = {"n": len(common), "union_minus_first_hit@1": {m: float(np.mean([int(ud[q]["rank"][m] == 1) - int(fd[q]["rank"][m] == 1) for q in common])) for m in models},
                  "sig_first": [x["winner"] + ">" + (x["b"] if x["winner"] == x["a"] else x["a"]) for x in A.pairwise(rf, models, n_boot=1) if x["significant"]],
                  "sig_union": [x["winner"] + ">" + (x["b"] if x["winner"] == x["a"] else x["a"]) for x in A.pairwise(ru, models, n_boot=1) if x["significant"]]}
    return out


# --------------------------------------------------------------------------- I
def missing(raw):
    corpus = raw["corpus"]
    out = {}
    for s in A.SCHEMES[corpus]:
        excluded = sum(1 for q in raw["questions"] if not A.gold_for(q, corpus, s))
        out[s] = {"excluded": excluded, "of": len(raw["questions"])}
    return out


A_cache: dict = {}
file0: set = set()


def main():
    report = {}
    report["B_chance_formula"] = check_chance()
    for c in ("qasper", "nq"):
        raw, ana = load(c)
        A_cache[c] = ana
        if c == "nq":
            # questions of file 0 are those of the first raw file: reload ids from the repro file order
            n0 = raw["config"]["raw_files"][0]["questions_loaded"]
            file0.update(q["question_id"] for q in raw["questions"][:n0])
        r = {"A_metrics": check_metrics(raw, ana), "C_alpha": check_alpha(raw, ana), "D_tests": check_tests(raw, ana),
             "E_cluster_permutation": cluster_permutation(raw), "F_power": power_analysis(raw, ana),
             "G_random_gold_control": random_gold_control(raw), "H_split_sensitivity": split_sensitivity(raw),
             "I_missing": missing(raw)}
        report[c] = r
        print(c, "done", flush=True)
    dst = REV / "results/audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "statistical_audit.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
