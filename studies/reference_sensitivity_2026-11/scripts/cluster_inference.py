#!/usr/bin/env python
"""Cluster-aware primary inference for the final audit (25 November 2026).

Primary procedure for every paired comparison of hit@1 (model A vs model B under one reference,
and reference R1 vs R2 for one model):
  * clustering unit: the QASPER paper (doc_id); for Natural Questions the Wikipedia *page*
    (title extracted from the document URL, URL-decoded, case-folded), so that different
    revisions of the same page (different oldid) and the duplicated '#fileN' copy are one cluster;
  * test: paired sign-flip permutation test on cluster sums of the per-question differences
    (20,000 random sign vectors, two-sided, p = (1 + #{|T*| >= |T|}) / (1 + B));
  * interval: cluster bootstrap of the mean difference (4,000 resamples of clusters, percentile 95%);
  * multiplicity: Holm over all model pairs within a corpus and reference.
Secondary procedure (sensitivity): exact McNemar on the question-level discordant pairs, Holm.
Also reported: the same permutation test with the document id as the cluster (the earlier version),
to show whether the page-level clustering changes any conclusion.

    python scripts/cluster_inference.py      # writes results/audit/cluster_inference.json
"""
from __future__ import annotations

import gzip
import html
import itertools
import json
import re
import sys
import urllib.parse
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
for p in (REPO, REPO / "scripts", REV / "scripts"):
    sys.path.insert(0, str(p))

import analyse_v2 as A  # noqa: E402
from src.evaluation.statistics import mcnemar_exact  # noqa: E402

N_PERM, N_BOOT, ALPHA = 20000, 4000, 0.05


def page_key(corpus: str, doc_id: str) -> str:
    if corpus == "qasper":
        return doc_id
    base = doc_id.split("#")[0]
    m = re.search(r"title=([^&]+)", html.unescape(base))
    if not m:
        return base
    return urllib.parse.unquote(m.group(1)).replace("_", " ").strip().casefold()


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj = [0.0] * len(p)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = prev
    return adj


def grouped(keys: list[str], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cluster sums and cluster sizes."""
    idx: dict[str, int] = {}
    for k in keys:
        idx.setdefault(k, len(idx))
    g = np.array([idx[k] for k in keys])
    return np.bincount(g, weights=values, minlength=len(idx)), np.bincount(g, minlength=len(idx)).astype(float)


def perm_p(sums: np.ndarray, rng: np.random.Generator) -> float:
    obs = abs(sums.sum())
    flips = rng.choice(np.array([-1.0, 1.0]), size=(N_PERM, len(sums)))
    null = np.abs(flips @ sums)
    return float((1 + (null >= obs - 1e-9).sum()) / (N_PERM + 1))


def boot_ci(sums: np.ndarray, sizes: np.ndarray, rng: np.random.Generator) -> list[float]:
    pick = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    est = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def compare(keys_page, keys_doc, diff, rng):
    s_page, n_page = grouped(keys_page, diff)
    s_doc, _ = grouped(keys_doc, diff)
    return {"delta": float(diff.mean()), "ci95_cluster": boot_ci(s_page, n_page, rng),
            "p_perm": perm_p(s_page, rng), "p_perm_doc_cluster": perm_p(s_doc, rng),
            "n_clusters": int(len(s_page))}


def pairwise_block(rows, models, corpus, rng):
    kp = [page_key(corpus, r["doc"]) for r in rows]
    kd = [r["doc"] for r in rows]
    res = []
    for a, b in itertools.combinations(models, 2):
        ha = np.array([int(r["rank"][a] == 1) for r in rows], dtype=float)
        hb = np.array([int(r["rank"][b] == 1) for r in rows], dtype=float)
        x = compare(kp, kd, ha - hb, rng)
        only_a, only_b = int(((ha == 1) & (hb == 0)).sum()), int(((ha == 0) & (hb == 1)).sum())
        x.update({"a": a, "b": b, "only_a": only_a, "only_b": only_b,
                  "discordant_ratio": (only_a / only_b) if only_b else None,
                  "p_mcnemar": float(mcnemar_exact(only_a, only_b).p_value)})
        res.append(x)
    for key in ("p_perm", "p_mcnemar", "p_perm_doc_cluster"):
        for x, adj in zip(res, holm([x[key] for x in res]), strict=True):
            x[key + "_holm"] = float(adj)
    for x in res:
        winner = x["a"] if x["delta"] > 0 else x["b"]
        loser = x["b"] if winner == x["a"] else x["a"]
        x["label"] = f"{winner}>{loser}"
        x["sig_perm"] = x["p_perm_holm"] < ALPHA
        x["sig_mcnemar"] = x["p_mcnemar_holm"] < ALPHA
        x["sig_perm_doc_cluster"] = x["p_perm_doc_cluster_holm"] < ALPHA
    sp = sorted(x["label"] for x in res if x["sig_perm"])
    sm = sorted(x["label"] for x in res if x["sig_mcnemar"])
    sd = sorted(x["label"] for x in res if x["sig_perm_doc_cluster"])
    return {"n": len(rows), "n_clusters": len(set(kp)), "n_doc_ids": len(set(kd)), "pairs": res,
            "significant_primary": sp, "significant_mcnemar": sm, "significant_perm_doc_cluster": sd,
            "mcnemar_agrees_with_primary": sp == sm, "doc_cluster_agrees_with_primary": sp == sd}


def main() -> int:
    out = {"procedure": __doc__.split("\n\n")[1].strip(), "n_perm": N_PERM, "n_boot": N_BOOT, "alpha": ALPHA}
    for corpus, f in (("qasper", "qasper_full"), ("nq", "nq_full")):
        raw = json.load(gzip.open(REV / f"results/raw/{f}.json.gz", "rt", encoding="utf-8"))
        models = list(raw["config"]["models"])
        rng = np.random.default_rng(1)
        c = {"models": models, "schemes": {}, "reference_effects": {}, "splits": {}}
        rows_by = {}
        for s in A.SCHEMES[corpus]:
            rows = A.rows_for(raw, models, s)
            rows_by[s] = rows
            c["schemes"][s] = pairwise_block(rows, models, corpus, rng)
        # reference effects: union minus first, per model, on questions with gold under both
        f_, u_ = ({r["qid"]: r for r in rows_by[s]} for s in ("first", "union"))
        common = [q for q in f_ if q in u_]
        kp = [page_key(corpus, f_[q]["doc"]) for q in common]
        kd = [f_[q]["doc"] for q in common]
        eff = {}
        for m in models:
            d = np.array([int(u_[q]["rank"][m] == 1) - int(f_[q]["rank"][m] == 1) for q in common], dtype=float)
            eff[m] = compare(kp, kd, d, rng)
        for m, adj in zip(models, holm([eff[m]["p_perm"] for m in models]), strict=True):
            eff[m]["p_perm_holm"] = float(adj)
        c["reference_effects"] = {"n": len(common), "union_minus_first": eff}
        # union-minus-first gap between each neural model and BM25 (does the union help neural models more?)
        gap = {}
        for m in models:
            if m == "bm25":
                continue
            d = np.array([(int(u_[q]["rank"][m] == 1) - int(f_[q]["rank"][m] == 1))
                          - (int(u_[q]["rank"]["bm25"] == 1) - int(f_[q]["rank"]["bm25"] == 1)) for q in common], dtype=float)
            gap[m] = compare(kp, kd, d, rng)
        for m, adj in zip(list(gap), holm([gap[m]["p_perm"] for m in gap]), strict=True):
            gap[m]["p_perm_holm"] = float(adj)
        c["neural_minus_bm25_reference_effect"] = gap
        # sample splits (QASPER dev/test; NQ file 0/1)
        n0 = raw["config"]["raw_files"][0]["questions_loaded"] if corpus == "nq" else None
        file0 = {q["question_id"] for q in raw["questions"][:n0]} if n0 else set()
        key = (lambda q: q["split"]) if corpus == "qasper" else (lambda q: "file0" if q["question_id"] in file0 else "file1")
        groups: dict[str, list] = {}
        for q in raw["questions"]:
            groups.setdefault(key(q), []).append(q)
        for g, qs in groups.items():
            sub = dict(raw)
            sub["questions"] = qs
            blk = {}
            for s in ("first", "union"):
                b = pairwise_block(A.rows_for(sub, models, s), models, corpus, rng)
                blk[s] = {"n": b["n"], "significant_primary": b["significant_primary"],
                          "significant_mcnemar": b["significant_mcnemar"]}
            rf = {r["qid"]: r for r in A.rows_for(sub, models, "first")}
            ru = {r["qid"]: r for r in A.rows_for(sub, models, "union")}
            cm = [q for q in rf if q in ru]
            blk["n_common"] = len(cm)
            blk["union_minus_first_hit@1"] = {m: float(np.mean([int(ru[q]["rank"][m] == 1) - int(rf[q]["rank"][m] == 1) for q in cm])) for m in models}
            c["splits"][g] = blk
        out[corpus] = c
        print(corpus, {s: (v["significant_primary"], v["mcnemar_agrees_with_primary"], v["doc_cluster_agrees_with_primary"]) for s, v in c["schemes"].items()}, flush=True)
    dst = REV / "results/audit/cluster_inference.json"
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
