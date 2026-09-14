#!/usr/bin/env python
"""Consolidate the within-document localisation runs for one corpus into a single
machine-readable summary, a markdown table block and a rank-distribution figure.

Inputs are the per-run JSON files written by localisation_probe.py (bi-encoders and
BM25, possibly split over several files) and localisation_extra.py (cross-encoders,
hybrid and oracle variants). Every number in the summary is recomputed here from the
per-question rows, so the tables cannot drift from the raw files.

    python scripts/localisation_report.py --dataset qasper \
        --probe results/localisation/within_document_qasper.json \
        --extra results/localisation/localisers_qasper.json \
        --reranker results/localisation/reranker_bge_base_qasper.json \
        --out results/localisation/summary_qasper.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.provenance import collect_provenance  # noqa: E402
from src.evaluation.statistics import mcnemar_exact, wilson_proportion_ci  # noqa: E402

#: Identity of every localiser, in the order tables are printed.
MODELS = {
    "bm25": ("BM25 (Okapi, k1=1.5, b=0.75)", "lexical", None),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "dense bi-encoder", 22),
    "bge": ("BAAI/bge-small-en-v1.5", "dense bi-encoder", 33),
    "e5": ("intfloat/e5-small-v2", "dense bi-encoder", 33),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "dense bi-encoder", 110),
    "ce_msmarco_minilm": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder", 22),
    "bge_reranker_base": ("BAAI/bge-reranker-base", "cross-encoder", 278),
}
VARIANTS = {
    "bm25_local_idf": "BM25 with document-local IDF",
    "rrf_bm25+bge": "RRF(BM25, BGE-small), k=60",
    "rrf_bm25+minilm": "RRF(BM25, MiniLM), k=60",
    "oracle_union_bm25+dense": "best-of(BM25, dense partner) per question - a ceiling, not a system",
}
BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10), (">10", 11, 10 ** 9)]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def summarise(rows: list[dict], chance: dict | None) -> dict:
    ranks = np.array([r["rank"] for r in rows], dtype=float)
    n = len(rows)
    hit = {k: float(np.mean(ranks <= k)) for k in (1, 3, 5)}
    lo, hi = wilson_proportion_ci(int(np.sum(ranks <= 1)), n).lower, wilson_proportion_ci(int(np.sum(ranks <= 1)), n).upper
    out = {
        "n": n,
        "hit@1": round(hit[1], 4), "hit@1_wilson95": [round(lo, 4), round(hi, 4)],
        "hit@3": round(hit[3], 4), "hit@5": round(hit[5], 4),
        "mrr": round(float(np.mean(1.0 / ranks)), 4),
        "median_rank": float(np.median(ranks)),
        "mean_rank": round(float(np.mean(ranks)), 2),
        "rank_distribution": {name: round(float(np.mean((ranks >= lo_) & (ranks <= hi_))), 4)
                              for name, lo_, hi_ in BUCKETS},
        "near_miss_share_ranks_2_to_5": round(float(np.mean((ranks >= 2) & (ranks <= 5))), 4),
        "gold_in_top_quarter_of_document": round(float(np.mean(
            [r["rank"] <= max(1, round(0.25 * r["D"])) for r in rows])), 4),
    }
    if chance:
        out["lift@1_over_chance"] = round(hit[1] / chance["hit@1"], 3) if chance["hit@1"] else None
        out["lift@5_over_chance"] = round(hit[5] / chance["hit@5"], 3) if chance["hit@5"] else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--reranker", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    within: dict[str, dict[str, dict]] = {}
    global_rows: dict[str, dict[str, dict]] = {}
    chance_rows: dict[str, dict] = {}
    config = None
    for path in args.probe:
        P = json.loads(Path(path).read_text(encoding="utf-8"))
        S = P["summary"]
        config = config or {"dataset": S["dataset"], "split": S["split"], "limit": S["limit"],
                            "chunking": S["chunking"], "top_k": S["top_k"],
                            "n_documents": S["n_documents"], "n_chunks": S["n_chunks"],
                            "n_questions": S["n_questions"], "geometry": S["geometry"]}
        for key, rows in P["within_document_rows"].items():
            if key == "random":
                for r in rows:
                    chance_rows.setdefault(r["question_id"], r)
                continue
            within.setdefault(key, {}).update({r["question_id"]: r for r in rows})
        for key, rows in P["global_rows"].items():
            global_rows.setdefault(key, {}).update({r["question_id"]: r for r in rows})
    variants: dict[str, dict[str, dict]] = {}
    for path in args.extra:
        E = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
        for key, rows in E.items():
            target = "ce_msmarco_minilm" if key == "cross_encoder" else key
            if target in MODELS:
                within.setdefault(target, {}).update({r["question_id"]: r for r in rows})
            elif key in VARIANTS:
                variants.setdefault(key, {}).update({r["question_id"]: r for r in rows})
    if args.reranker:
        R = json.loads(Path(args.reranker).read_text(encoding="utf-8"))["rows"]["cross_encoder"]
        within["bge_reranker_base"] = {r["question_id"]: r for r in R}

    ids = sorted(set.intersection(*[set(v) for v in within.values()]))
    present = [k for k in MODELS if k in within]
    chance = {
        "hit@1": float(np.mean([chance_rows[q]["hit@1"] for q in ids])),
        "hit@3": float(np.mean([chance_rows[q]["hit@3"] for q in ids])),
        "hit@5": float(np.mean([chance_rows[q]["hit@5"] for q in ids])),
        "mrr": float(np.mean([chance_rows[q]["rr"] for q in ids])),
    }

    summary = {"config": config, "n_questions_common": len(ids),
               "evidence_definition": (
                   "Rank every chunk of the question's gold document (the first gold document when a "
                   "question cites several) against the question; the gold rank is the rank of the "
                   "first chunk whose character range overlaps any gold span by at least one "
                   "character (any_sufficient). Chance is the analytic expectation under a random "
                   "ordering of the document's D chunks with m gold-overlapping chunks."),
               "chance": {k: round(v, 4) for k, v in chance.items()},
               "models": {}, "variants": {}, "global_topk": {}, "pairwise": {}, "rank_correlation": {},
               "union": {}, "size_family": []}
    for key in present:
        rows = [within[key][q] for q in ids]
        summary["models"][key] = {"identity": MODELS[key][0], "family": MODELS[key][1],
                                  "params_millions": MODELS[key][2], **summarise(rows, chance)}
    for key, rows_by_id in variants.items():
        rows = [rows_by_id[q] for q in ids if q in rows_by_id]
        if rows:
            summary["variants"][key] = {"description": VARIANTS[key], **summarise(rows, chance)}
    for key, rows_by_id in global_rows.items():
        rows = [rows_by_id[q] for q in ids if q in rows_by_id]
        if not rows:
            continue
        A = sum(r["A"] for r in rows)
        C = sum(r["C"] for r in rows)
        reached = [r for r in rows if r["A"]]
        summary["global_topk"][key] = {
            "A_reach": round(A / len(rows), 4), "C_span": round(C / len(rows), 4),
            "C_given_A": round(C / A, 4) if A else None,
            "chunks_admitted_from_gold_doc_given_A": round(float(np.mean([r["admitted"] for r in reached])), 3) if reached else None,
        }

    # pairwise: exact McNemar on hit@1, exact sign test on reciprocal rank; Holm over the McNemar family
    pairs = list(itertools.combinations(present, 2))
    raw_p = []
    for a, b in pairs:
        ra, rb = within[a], within[b]
        a_only = sum(1 for q in ids if ra[q]["rank"] == 1 and rb[q]["rank"] != 1)
        b_only = sum(1 for q in ids if rb[q]["rank"] == 1 and ra[q]["rank"] != 1)
        mc = mcnemar_exact(a_only, b_only)
        d = [1 / ra[q]["rank"] - 1 / rb[q]["rank"] for q in ids]
        wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
        p_sign = float(stats.binomtest(min(wins, losses), wins + losses).pvalue) if wins + losses else 1.0
        raw_p.append(mc.p_value if mc.p_value is not None else 1.0)
        summary["pairwise"][f"{a}_vs_{b}"] = {
            "hit@1_diff": round(summary["models"][a]["hit@1"] - summary["models"][b]["hit@1"], 4),
            "hit@1_only_a": a_only, "hit@1_only_b": b_only,
            "mcnemar_exact_p": None if mc.p_value is None else round(mc.p_value, 4),
            "mrr_diff": round(float(np.mean(d)), 4), "rr_wins_a": wins, "rr_wins_b": losses,
            "sign_test_p": round(p_sign, 4),
        }
    for (a, b), adj in zip(pairs, holm(raw_p), strict=True):
        summary["pairwise"][f"{a}_vs_{b}"]["mcnemar_holm_adjusted_p"] = round(adj, 4)

    for a in present:
        summary["rank_correlation"][a] = {}
        for b in present:
            if a == b:
                continue
            rho = stats.spearmanr([within[a][q]["rank"] for q in ids], [within[b][q]["rank"] for q in ids]).correlation
            summary["rank_correlation"][a][b] = round(float(rho), 3)

    H = np.array([[within[k][q]["rank"] == 1 for k in present] for q in ids])
    T5 = np.array([[within[k][q]["rank"] <= 5 for k in present] for q in ids])
    bi = [k for k in present if MODELS[k][1] == "dense bi-encoder"]
    ce = [k for k in present if MODELS[k][1] == "cross-encoder"]
    idx = {k: i for i, k in enumerate(present)}
    summary["union"] = {
        "models": present,
        "any_hit@1": round(float(H.any(1).mean()), 4),
        "none_hit@1": round(float((~H.any(1)).mean()), 4),
        "all_hit@1": round(float(H.all(1).mean()), 4),
        "independence_expectation_all_hit@1": round(float(np.prod(H.mean(0))), 4),
        "independence_expectation_none_hit@1": round(float(np.prod(1 - H.mean(0))), 4),
        "any_bi_encoder_hit@1": round(float(H[:, [idx[k] for k in bi]].any(1).mean()), 4) if bi else None,
        "any_cross_encoder_hit@1": round(float(H[:, [idx[k] for k in ce]].any(1).mean()), 4) if ce else None,
        "gold_outside_top5_for_every_model": round(float((~T5.any(1)).mean()), 4),
        "best_single_hit@1": max(summary["models"][k]["hit@1"] for k in present),
    }
    summary["size_family"] = sorted(
        [{"model": k, "family": MODELS[k][1], "params_millions": MODELS[k][2],
          "hit@1": summary["models"][k]["hit@1"], "mrr": summary["models"][k]["mrr"]} for k in present],
        key=lambda d: (d["params_millions"] or 0))
    summary["provenance"] = collect_provenance(
        inputs={"probe": args.probe, "extra": args.extra, "reranker": args.reranker},
        determinism="no random seed is used: rankings are stable argsorts of deterministic scores; "
                    "chance levels are analytic",
        bm25={"k1": 1.5, "b": 0.75, "tokeniser": "[a-z0-9]+ on lowercased text", "idf": "corpus-level"},
        dense={"similarity": "cosine on L2-normalised vectors", "query_prefixes": "as registered in src/rag/embedders.py",
               "chunk_vectors": "read from the stored Chroma indexes where they exist, otherwise encoded once and cached"},
        cross_encoders={"max_length": 512, "batch_size": 32, "device": "cpu"},
    )
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # ---- markdown table block ----
    L = []
    c = summary["config"]
    L.append(f"**{args.dataset.upper()}** — {c['n_questions']} answerable questions with locatable gold spans, "
             f"{c['n_documents']} documents, {c['n_chunks']} chunks of {c['chunking']['size']} tokens "
             f"(overlap {c['chunking']['overlap']}); {c['geometry']['mean_chunks_per_gold_doc']:.1f} chunks per gold "
             f"document, {c['geometry']['mean_gold_chunks_per_doc']:.2f} gold-overlapping chunks per document.\n")
    L.append("| Model | Family | Params | hit@1 [95% CI] | hit@3 | hit@5 | MRR | median rank | ranks 2–5 | lift@1 |")
    L.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    L.append(f"| chance | — | — | {chance['hit@1']:.3f} | {chance['hit@3']:.3f} | {chance['hit@5']:.3f} | "
             f"{chance['mrr']:.3f} | — | — | 1.00 |")
    for k in present:
        m = summary["models"][k]
        L.append(f"| {m['identity']} | {m['family']} | {m['params_millions'] or '—'}{'M' if m['params_millions'] else ''} | "
                 f"{m['hit@1']:.3f} [{m['hit@1_wilson95'][0]:.3f}, {m['hit@1_wilson95'][1]:.3f}] | {m['hit@3']:.3f} | "
                 f"{m['hit@5']:.3f} | {m['mrr']:.3f} | {m['median_rank']:.0f} | {m['near_miss_share_ranks_2_to_5']:.3f} | "
                 f"{m['lift@1_over_chance']:.2f} |")
    for m in summary["variants"].values():
        L.append(f"| *{m['description']}* | variant | — | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['hit@5']:.3f} | "
                 f"{m['mrr']:.3f} | {m['median_rank']:.0f} | {m['near_miss_share_ranks_2_to_5']:.3f} | "
                 f"{m['lift@1_over_chance']:.2f} |")
    u = summary["union"]
    L.append("")
    L.append(f"Union over the {len(present)} models: some model ranks the gold first for {u['any_hit@1']:.3f} of questions "
             f"(best single model {u['best_single_hit@1']:.3f}); no model does for {u['none_hit@1']:.3f} "
             f"(independence would give {u['independence_expectation_none_hit@1']:.3f}); every model does for "
             f"{u['all_hit@1']:.3f}; the gold is outside every model's top 5 for {u['gold_outside_top5_for_every_model']:.3f}.")
    L.append("")
    L.append("Rank distribution of the gold chunk (share of questions):\n")
    L.append("| Model | " + " | ".join(f"rank {b[0]}" for b in BUCKETS) + " | top quarter of doc |")
    L.append("|---|" + "---:|" * (len(BUCKETS) + 1))
    for k in present:
        m = summary["models"][k]
        L.append(f"| {k} | " + " | ".join(f"{m['rank_distribution'][b[0]]:.2f}" for b in BUCKETS) +
                 f" | {m['gold_in_top_quarter_of_document']:.2f} |")
    L.append("")
    L.append("Pairwise, hit@1 (exact McNemar; Holm-adjusted over all pairs) and reciprocal rank (exact sign test):\n")
    L.append("| A vs B | Δhit@1 | A-only / B-only | p | p (Holm) | ΔMRR | RR wins A/B | sign p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, v in summary["pairwise"].items():
        L.append(f"| {name.replace('_vs_', ' vs ')} | {v['hit@1_diff']:+.3f} | {v['hit@1_only_a']}/{v['hit@1_only_b']} | "
                 f"{v['mcnemar_exact_p']} | {v['mcnemar_holm_adjusted_p']} | {v['mrr_diff']:+.3f} | "
                 f"{v['rr_wins_a']}/{v['rr_wins_b']} | {v['sign_test_p']} |")
    L.append("")
    L.append("Spearman correlation of the gold chunk's within-document rank between models:\n")
    L.append("| | " + " | ".join(present) + " |")
    L.append("|---|" + "---:|" * len(present))
    for a in present:
        L.append(f"| {a} | " + " | ".join("—" if a == b else f"{summary['rank_correlation'][a][b]:.2f}" for b in present) + " |")
    if summary["global_topk"]:
        L.append("")
        L.append(f"Global top-{c['top_k']} over the whole corpus (same questions): reach A, span coverage C, "
                 "localisation given reach, chunks admitted from the gold document:\n")
        L.append("| Model | A | C | C given A | admitted given A |")
        L.append("|---|---:|---:|---:|---:|")
        for k, g in summary["global_topk"].items():
            L.append(f"| {k} | {g['A_reach']:.3f} | {g['C_span']:.3f} | {g['C_given_A']:.3f} | "
                     f"{g['chunks_admitted_from_gold_doc_given_A']:.2f} |")
    md_path = Path(args.out).with_suffix(".md")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {Path(args.out).as_posix()} and {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
