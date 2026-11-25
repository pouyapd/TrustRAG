#!/usr/bin/env python
"""Analyse the revised study's scored runs (score_v2.py output). Seconds; no models.

For one corpus file it computes, under every evidence reference the dataset supports:

* within-document localisation: hit@k, MRR, median rank, analytic chance, lift and a
  chance-corrected hit@1 (hit - chance) / (1 - chance), Wilson and document-cluster
  bootstrap intervals;
* the effect of the reference on each model's score (paired, cluster-bootstrapped);
* pairwise model comparisons (exact McNemar, Holm over all pairs) per reference, and
  which conclusions change between references;
* single-annotator resampling: the spread of each score over random annotator draws;
* agreement between the dataset's own annotators at chunk level (pairwise identical /
  overlapping / disjoint gold chunks, mean Jaccard, Krippendorff's alpha, binary);
* penalised valid evidence: rank-1 chunks counted as misses under a single annotator's
  reference although another annotator marked them;
* for BM25 and bi-encoders, global top-5 reach (A), span coverage (C), C/A and admitted
  chunks under each reference.

    python analyse_v2.py results/raw/qasper_full.json.gz --out results/qasper_full
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from localisation_probe import random_expectations  # noqa: E402
from localisation_report import holm  # noqa: E402
from localisation_robustness import cluster_bootstrap  # noqa: E402

from src.evaluation.references import (  # noqa: E402
    draw_single,
    draw_single_evidence,
    gold_chunks,
    reference_annotators,
    reference_spans,
    usable_annotations,
)
from src.evaluation.statistics import mcnemar_exact  # noqa: E402

KS = (1, 3, 5, 10)
SCHEMES = {"qasper": ("union", "first", "plurality", "agreement"),
           "nq": ("first", "union", "official", "plurality")}
STUDY_SCHEME = {"qasper": "union", "nq": "first"}
LABEL = {"union": "union of annotators", "first": "first annotator", "plurality": "plurality",
         "agreement": "marked by >=2 annotators", "official": "NQ official rule (>=2 non-null, union)",
         "single": "one annotator drawn at random"}


# --------------------------------------------------------------------------- references

def gold_for(q: dict, corpus: str, scheme: str) -> set[int] | None:
    ann = q["annotations"]
    if corpus == "qasper":
        spans = reference_spans(ann, scheme)
        return None if spans is None else gold_chunks(q["chunk_ranges"], spans) or None
    annotators = reference_annotators(ann, scheme)
    if annotators is None:
        return None
    gold = set().union(*(set(q["gold_by_annotator"][str(a)]) for a in annotators))
    return gold or None


def gold_single(q: dict, corpus: str, rng: random.Random) -> set[int]:
    if corpus == "qasper":
        return gold_chunks(q["chunk_ranges"], draw_single_evidence(q["annotations"], rng))
    (a,) = draw_single(q["annotations"], rng)
    return set(q["gold_by_annotator"][str(a)])


def rank_of(order: list[int], gold: set[int]) -> int:
    return next(p for p, i in enumerate(order, 1) if i in gold)


# --------------------------------------------------------------------------- metrics

def rows_for(data: dict, models: list[str], scheme: str, rng: random.Random | None = None) -> list[dict]:
    corpus = data["corpus"]
    rows = []
    for q in data["questions"]:
        if scheme == "single":
            if not usable_annotations(q["annotations"]):
                continue
            gold = gold_single(q, corpus, rng)
        else:
            gold = gold_for(q, corpus, scheme)
        if not gold:
            continue
        rows.append({"qid": q["question_id"], "doc": q["doc_id"], "D": q["D"], "m": len(gold), "gold": gold,
                     "rank": {m: rank_of(q["orders"][m], gold) for m in models}})
    return rows


def chance(rows: list[dict]) -> dict:
    exp = [random_expectations(r["D"], r["m"], ks=KS) for r in rows]
    return {"mrr": float(np.mean([e["rr"] for e in exp])),
            **{f"hit@{k}": float(np.mean([e[f"hit@{k}"] for e in exp])) for k in KS}}


def by_doc(rows: list[dict], values: list[float]) -> list[list[float]]:
    groups: dict[str, list[float]] = {}
    for r, v in zip(rows, values, strict=True):
        groups.setdefault(r["doc"], []).append(v)
    return list(groups.values())


def model_metrics(rows: list[dict], model: str, ch: dict, n_boot: int) -> dict:
    ranks = np.array([r["rank"][model] for r in rows])
    n = len(ranks)
    out = {f"hit@{k}": float((ranks <= k).mean()) for k in KS}
    out["mrr"] = float((1.0 / ranks).mean())
    out["median_rank"] = float(np.median(ranks))
    lo, hi = _wilson(int((ranks == 1).sum()), n)
    out["hit@1_wilson95"] = [float(lo), float(hi)]
    cb = cluster_bootstrap(by_doc(rows, list((ranks == 1).astype(int))), n_boot=n_boot, seed=0)
    out["hit@1_cluster95"] = cb["ci95"]
    out["chance_hit@1"] = ch["hit@1"]
    out["lift@1"] = out["hit@1"] / ch["hit@1"]
    out["kappa@1"] = (out["hit@1"] - ch["hit@1"]) / (1 - ch["hit@1"])
    out["share_rank_2_5"] = float(((ranks >= 2) & (ranks <= 5)).mean())
    return out


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def pairwise(rows: list[dict], models: list[str], n_boot: int) -> list[dict]:
    out = []
    for a, b in itertools.combinations(models, 2):
        ha = np.array([r["rank"][a] == 1 for r in rows])
        hb = np.array([r["rank"][b] == 1 for r in rows])
        only_a, only_b = int((ha & ~hb).sum()), int((hb & ~ha).sum())
        test = mcnemar_exact(only_a, only_b)
        cb = cluster_bootstrap(by_doc(rows, list(ha.astype(int) - hb.astype(int))), n_boot=n_boot, seed=0)
        out.append({"a": a, "b": b, "delta_hit@1": float(ha.mean() - hb.mean()), "only_a": only_a,
                    "only_b": only_b, "p": float(test.p_value), "cluster95": cb["ci95"]})
    adj = holm([x["p"] for x in out])
    for x, p in zip(out, adj, strict=True):
        x["p_holm"] = float(p)
        x["significant"] = bool(p < 0.05)
        x["winner"] = (x["a"] if x["delta_hit@1"] > 0 else x["b"]) if x["significant"] else None
    return out


def ordering(metrics: dict, key: str = "hit@1") -> list[str]:
    return sorted(metrics, key=lambda m: -metrics[m][key])


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    pos_b = {m: i for i, m in enumerate(order_b)}
    conc = disc = 0
    for x, y in itertools.combinations(order_a, 2):
        if pos_b[x] < pos_b[y]:
            conc += 1
        else:
            disc += 1
    return (conc - disc) / (conc + disc)


# --------------------------------------------------------------------------- annotators

def annotator_agreement(data: dict) -> dict:
    """Chunk-level agreement between the dataset's own annotators (no models)."""
    pairs = identical = overlap = disjoint = 0
    jacc, alpha_units = [], []
    q_multi = 0
    q_any_disjoint = 0
    for q in data["questions"]:
        sets = [set(v) for v in q["gold_by_annotator"].values() if v]
        if len(sets) < 2:
            continue
        q_multi += 1
        any_disjoint = False
        for s, t in itertools.combinations(sets, 2):
            pairs += 1
            inter = len(s & t)
            jacc.append(inter / len(s | t))
            if s == t:
                identical += 1
            elif inter:
                overlap += 1
            else:
                disjoint += 1
                any_disjoint = True
        q_any_disjoint += any_disjoint
        # Krippendorff's alpha (binary, nominal): every chunk of the document is a unit
        # coded gold / not gold by each annotator who gave evidence.
        for c in range(q["D"]):
            alpha_units.append([int(c in s) for s in sets])
    alpha = krippendorff_binary(alpha_units)
    return {"questions_with_2plus_annotators": q_multi, "annotator_pairs": pairs,
            "pairs_identical": identical / pairs, "pairs_overlapping_not_identical": overlap / pairs,
            "pairs_disjoint": disjoint / pairs, "mean_jaccard": float(np.mean(jacc)),
            "questions_with_a_disjoint_pair": q_any_disjoint / q_multi,
            "krippendorff_alpha_chunk_level": alpha}


def krippendorff_binary(units: list[list[int]]) -> float:
    """Krippendorff's alpha for nominal (binary) data, units with >= 2 codes."""
    o = np.zeros((2, 2))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        c = np.bincount(u, minlength=2)
        for a in (0, 1):
            for b in (0, 1):
                o[a, b] += (c[a] * (c[b] - (a == b))) / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d_o = o[0, 1] + o[1, 0]
    d_e = (n_c[0] * n_c[1] * 2) / (n - 1)
    return float(1 - d_o / d_e) if d_e else float("nan")


def penalised_valid_evidence(data: dict, models: list[str], rng_seed: int = 0, n_draws: int = 200) -> dict:
    """Rank-1 misses under one annotator's reference that another annotator marked as gold.

    Over questions with >= 2 annotators who gave evidence: for each draw, one annotator is
    the reference; a model's top chunk that misses that reference but is gold for some
    other annotator is a 'penalised valid' miss.
    """
    rng = random.Random(rng_seed)
    multi = [q for q in data["questions"] if sum(1 for v in q["gold_by_annotator"].values() if v) >= 2]
    out = {}
    for m in models:
        miss_share, pen_share = [], []
        for _ in range(n_draws):
            misses = pen = 0
            for q in multi:
                sets = {a: set(v) for a, v in q["gold_by_annotator"].items() if v}
                ref = rng.choice(sorted(sets))
                top = q["orders"][m][0]
                if top not in sets[ref]:
                    misses += 1
                    if any(top in s for a, s in sets.items() if a != ref):
                        pen += 1
            miss_share.append(misses / len(multi))
            pen_share.append(pen / max(1, misses))
        out[m] = {"n_questions": len(multi), "rank1_miss_rate_single_reference": float(np.mean(miss_share)),
                  "share_of_misses_marked_by_another_annotator": float(np.mean(pen_share)),
                  "share_95_of_draws": [float(np.percentile(pen_share, 2.5)), float(np.percentile(pen_share, 97.5))]}
    return out


# --------------------------------------------------------------------------- global reach

def global_reach(data: dict, models: list[str], scheme: str, k: int = 5) -> dict:
    corpus = data["corpus"]
    doc_index = {d: i for i, d in enumerate(data["doc_ids"])}
    out = {}
    for m in models:
        if not data["questions"] or m not in data["questions"][0]["global_top"]:
            continue
        a = c = n = 0
        admitted = []
        for q in data["questions"]:
            gold = gold_for(q, corpus, scheme)
            if not gold:
                continue
            n += 1
            top = q["global_top"][m][:k]
            gdoc = doc_index[q["doc_id"]]
            in_doc = [pos for d, pos in top if d == gdoc]
            if in_doc:
                a += 1
                admitted.append(len(in_doc))
                if any(pos in gold for pos in in_doc):
                    c += 1
        out[m] = {"n": n, "A": a / n, "C": c / n, "C_over_A": c / a if a else None,
                  "admitted_mean": float(np.mean(admitted)) if admitted else None}
    return out


# --------------------------------------------------------------------------- sample size

def subsample_conclusions(data: dict, models: list[str], ref_a: str, ref_b: str,
                          sizes=(100, 300, 1000), n_sub: int = 500, seed: int = 7) -> dict:
    """How often do Holm conclusions under two references differ, for random subsamples?

    Questions defined under both references are subsampled without replacement; on each
    subsample all pairs are tested (exact McNemar, Holm) under each reference.
    """
    ra = {r["qid"]: r for r in rows_for(data, models, ref_a)}
    rb = {r["qid"]: r for r in rows_for(data, models, ref_b)}
    common = sorted(set(ra) & set(rb))
    rng = random.Random(seed)
    out = {}
    for n in sizes:
        if n > len(common):
            continue
        differ = 0
        sig_a = sig_b = 0
        for _ in range(n_sub):
            pick = rng.sample(common, n)
            sa = {(x["a"], x["b"], x["winner"]) for x in pairwise([ra[q] for q in pick], models, n_boot=1) if x["significant"]}
            sb = {(x["a"], x["b"], x["winner"]) for x in pairwise([rb[q] for q in pick], models, n_boot=1) if x["significant"]}
            differ += sa != sb
            sig_a += len(sa)
            sig_b += len(sb)
        out[str(n)] = {"share_conclusions_differ": differ / n_sub,
                       f"mean_significant_pairs_{ref_a}": sig_a / n_sub,
                       f"mean_significant_pairs_{ref_b}": sig_b / n_sub}
    full_a = {(x["a"], x["b"], x["winner"]) for x in pairwise([ra[q] for q in common], models, n_boot=1) if x["significant"]}
    full_b = {(x["a"], x["b"], x["winner"]) for x in pairwise([rb[q] for q in common], models, n_boot=1) if x["significant"]}
    out["full"] = {"n": len(common), "conclusions_differ": full_a != full_b}
    return out


# --------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()
    data = json.load(gzip.open(args.scored, "rt", encoding="utf-8"))
    corpus = data["corpus"]
    models = list(data["config"]["models"])

    # Sanity: the study's reference must be what the scheme rebuilds.
    study = STUDY_SCHEME[corpus]
    bad = [q["question_id"] for q in data["questions"]
           if (gold_for(q, corpus, study) or set()) != set(q["study_gold"])]
    result = {"corpus": corpus, "scored_file": Path(args.scored).name, "models": models,
              "n_questions_loaded_answerable": len(data["questions"]),
              "n_documents": data["config"]["n_documents"], "n_chunks": data["config"]["n_chunks"],
              "check_study_reference_rebuilt": {"scheme": study, "mismatching_questions": len(bad)},
              "schemes": {}, "settings": vars(args)}

    for scheme in SCHEMES[corpus]:
        rows = rows_for(data, models, scheme)
        ch = chance(rows)
        metrics = {m: model_metrics(rows, m, ch, args.n_boot) for m in models}
        pw = pairwise(rows, models, args.n_boot)
        result["schemes"][scheme] = {
            "label": LABEL[scheme], "n": len(rows), "n_docs": len({r["doc"] for r in rows}),
            "mean_gold_chunks": float(np.mean([r["m"] for r in rows])),
            "mean_D": float(np.mean([r["D"] for r in rows])), "chance": ch, "metrics": metrics,
            "ordering_hit@1": ordering(metrics), "pairwise": pw,
            "significant_pairs": [f"{x['winner']}>{x['b'] if x['winner'] == x['a'] else x['a']}"
                                  for x in pw if x["significant"]],
            "some_model_rank1": float(np.mean([any(r["rank"][m] == 1 for m in models) for r in rows])),
        }

    # Effect of the reference on each model, on questions where both schemes define gold.
    base, alt = SCHEMES[corpus][0], SCHEMES[corpus][1]
    effects = {}
    for s1, s2 in itertools.combinations(SCHEMES[corpus], 2):
        r1 = {r["qid"]: r for r in rows_for(data, models, s1)}
        r2 = {r["qid"]: r for r in rows_for(data, models, s2)}
        common = [q for q in r1 if q in r2]
        changed = sum(1 for q in common if r1[q]["gold"] != r2[q]["gold"])
        eff = {}
        for m in models:
            diffs = [int(r2[q]["rank"][m] == 1) - int(r1[q]["rank"][m] == 1) for q in common]
            cb = cluster_bootstrap(by_doc([r1[q] for q in common], diffs), n_boot=args.n_boot, seed=0)
            eff[m] = {"delta_hit@1": cb["delta_hit@1"], "cluster95": cb["ci95"]}
            d5 = [int(r2[q]["rank"][m] <= 5) - int(r1[q]["rank"][m] <= 5) for q in common]
            cb5 = cluster_bootstrap(by_doc([r1[q] for q in common], d5), n_boot=args.n_boot, seed=0)
            eff[m]["delta_hit@5"], eff[m]["cluster95_hit@5"] = cb5["delta_hit@1"], cb5["ci95"]
            drr = [1 / r2[q]["rank"][m] - 1 / r1[q]["rank"][m] for q in common]
            cbr = cluster_bootstrap(by_doc([r1[q] for q in common], drr), n_boot=args.n_boot, seed=0)
            eff[m]["delta_mrr"], eff[m]["cluster95_mrr"] = cbr["delta_hit@1"], cbr["ci95"]
        effects[f"{s2}_minus_{s1}"] = {"n_common": len(common), "questions_with_different_gold": changed,
                                       "models": eff}
    result["reference_effects"] = effects

    # Taus between orderings
    ords = {s: result["schemes"][s]["ordering_hit@1"] for s in SCHEMES[corpus]}
    result["ordering_kendall_tau"] = {f"{a}|{b}": kendall_tau(ords[a], ords[b])
                                      for a, b in itertools.combinations(ords, 2)}

    # Single-annotator resampling
    rng = random.Random(args.seed)
    draws = {m: [] for m in models}
    best = {m: 0 for m in models}
    sig_sets = []
    for d in range(args.n_draws):
        rows = rows_for(data, models, "single", rng)
        h = {m: float(np.mean([r["rank"][m] == 1 for r in rows])) for m in models}
        for m in models:
            draws[m].append(h[m])
        best[max(models, key=lambda m: h[m])] += 1
        if d < 200:  # Holm conclusions on the first 200 draws (tests are the slow part)
            sig_sets.append(tuple(sorted(f"{x['winner']}>{x['b'] if x['winner'] == x['a'] else x['a']}"
                                         for x in pairwise(rows, models, n_boot=1) if x["significant"])))
    study_sig = tuple(sorted(result["schemes"][study]["significant_pairs"]))
    result["single_annotator"] = {
        "n_draws": args.n_draws, "seed": args.seed,
        "hit@1": {m: {"mean": float(np.mean(v)), "draw95": [float(np.percentile(v, 2.5)),
                                                            float(np.percentile(v, 97.5))],
                      "draw_range": float(np.percentile(v, 97.5) - np.percentile(v, 2.5))} for m, v in draws.items()},
        "share_best_at_hit@1": {m: best[m] / args.n_draws for m in models},
        "holm_conclusion_same_as_study_share_of_200": float(np.mean([s == study_sig for s in sig_sets])),
        "distinct_holm_conclusions_in_200": len(set(sig_sets)),
    }
    result["annotator_agreement"] = annotator_agreement(data)
    result["penalised_valid_evidence"] = penalised_valid_evidence(data, models)
    result["global_top5"] = {s: global_reach(data, models, s) for s in SCHEMES[corpus]}
    result["subsample_first_vs_union"] = subsample_conclusions(data, models, "first", "union")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"check": result["check_study_reference_rebuilt"],
                      **{s: {"n": v["n"], "hit@1": {m: round(v["metrics"][m]["hit@1"], 3) for m in models},
                             "sig": v["significant_pairs"]} for s, v in result["schemes"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
