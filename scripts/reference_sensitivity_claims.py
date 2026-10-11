#!/usr/bin/env python
"""The localisation study's claims under harmonised evidence references (exploratory).

Not part of the submitted manuscript. The study builds its two references differently:
QASPER counts every annotator's evidence paragraphs (the union), NQ counts the first
annotator's long answer only. This script re-evaluates the claims of
results/localisation/README.md under the manuscript's pairing and under pairings that treat
both corpora alike, using the per-question ranks of the two exploratory analyses
(``nq_reference_sensitivity.py`` and ``qasper_reference_sensitivity.py``). No data or
models are needed; it never writes to results/localisation/.

    python scripts/reference_sensitivity_claims.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nq_reference_sensitivity as nq  # noqa: E402
import qasper_reference_sensitivity as qasper  # noqa: E402
from localisation_robustness import MODELS  # noqa: E402

from src.evaluation.provenance import collect_provenance  # noqa: E402
from src.evaluation.references import reference_annotators  # noqa: E402

OUT_JSON = "reference_sensitivity_claims.json"
OUT_MD = "reference_sensitivity_claims.md"
NEURAL = qasper.NEURAL
BI_ENCODERS = ("minilm", "bge", "e5", "mpnet")

#: Which reference each corpus is scored against in each pairing.
VIEWS = {
    "manuscript": {"qasper": "union", "nq": "first",
                   "label": "manuscript: QASPER union, NQ first annotator"},
    "both_union": {"qasper": "union", "nq": "union",
                   "label": "both corpora: union of annotators"},
    "both_first": {"qasper": "first", "nq": "first",
                   "label": "both corpora: first annotator"},
    "each_official": {"qasper": "union", "nq": "official",
                      "label": "each corpus by its own evaluator's rule"},
}


def corpus_rows(corpus: str, questions: list[dict], models: list[str], scheme: str) -> list[dict]:
    if corpus == "qasper":
        return qasper.scheme_rows(questions, models, scheme)
    rows = nq.scheme_rows(questions, models, scheme)
    # an NQ annotator marks one span; the reference's span count is its distinct spans
    by_id = {q["question_id"]: q for q in questions}
    for row in rows:
        anns = by_id[row["question_id"]]["annotations"]
        chosen = set(reference_annotators(anns, scheme))
        row["n_spans"] = len({(a["start_char"], a["end_char"]) for a in anns if a["annotator"] in chosen})
    return rows


def profile(rows: list[dict], models: list[str], n_boot: int) -> dict:
    metrics = {m: nq.model_metrics(rows, m) for m in models}
    hit1 = {m: metrics[m]["hit@1"] for m in models}
    neural = [m for m in models if m in NEURAL]
    tests = nq.pairwise(rows, models, n_boot=n_boot, seed=0)
    signs = nq._holm_significant(rows, models)
    n = len(rows)
    return {
        "n_questions": n,
        "mean_gold_spans_per_question": round(float(np.mean([r["n_spans"] for r in rows])), 2),
        "mean_gold_chunks_per_question": round(float(np.mean([r["m"] for r in rows])), 2),
        "chance_hit@1": metrics[models[0]]["chance_hit@1"],
        "hit@1": hit1,
        "hit@5": {m: metrics[m]["hit@5"] for m in models},
        "lift@1": {m: metrics[m]["lift@1_over_chance"] for m in models},
        "median_rank": {m: metrics[m]["median_rank"] for m in models},
        "share_ranks_2_to_5": {m: round(sum(metrics[m]["rank_buckets"][b] for b in ("2", "3", "4-5")), 4)
                               for m in models},
        "best": max(models, key=lambda m: hit1[m]),
        "worst_neural": min(neural, key=lambda m: hit1[m]),
        "neural_spread_pp": round(100 * (max(hit1[m] for m in neural) - min(hit1[m] for m in neural)), 1),
        "any_localiser_hit@1": round(sum(any(r["rank"][m] == 1 for m in models) for r in rows) / n, 4),
        "no_localiser_hit@1": round(sum(all(r["rank"][m] != 1 for m in models) for r in rows) / n, 4),
        "no_localiser_hit@1_under_independence": round(float(np.prod([1 - hit1[m] for m in models])), 4),
        "gold_outside_every_top5": round(sum(all(r["rank"][m] > 5 for m in models) for r in rows) / n, 4),
        "holm_significant": {pair: sign for pair, sign in signs.items() if sign},
        "cluster_bootstrap_ci95": {pair: tests[pair]["cluster_bootstrap_ci95"] for pair in tests},
        "mcnemar_holm_adjusted_p": {pair: tests[pair]["mcnemar_holm_adjusted_p"] for pair in tests},
    }


def _pair(p: dict, a: str, b: str) -> int:
    """+1 when ``a`` beats ``b`` after Holm correction, -1 when ``b`` beats ``a``, else 0."""
    if f"{a}_vs_{b}" in p["holm_significant"]:
        return p["holm_significant"][f"{a}_vs_{b}"]
    return -p["holm_significant"].get(f"{b}_vs_{a}", 0)


def _range(values) -> str:
    values = list(values)
    return f"{min(values):.3f}–{max(values):.3f}"


def _losers(p: dict) -> set[str]:
    out = set()
    for pair, sign in p["holm_significant"].items():
        a, b = pair.split("_vs_")
        out.add(b if sign > 0 else a)
    return out


def claims(q: dict, n: dict) -> list[dict]:
    """Each claim of results/localisation/README.md that the evidence reference can move."""
    both = (q, n)
    out = []

    def add(cid: str, where: str, text: str, holds: bool, detail: str, kind: str = "qualitative") -> None:
        # "numeric": the claim states numbers, and failing means they moved outside them
        out.append({"id": cid, "where": where, "claim": text, "kind": kind, "holds": bool(holds),
                    "detail": detail})

    top1 = max(max(p["hit@1"].values()) for p in both)
    add("C1", "What replicates 1", "No model exceeds hit@1 = 0.51 on either corpus",
        top1 <= 0.51, f"highest hit@1 {top1:.3f}")
    add("C2", "Claim", "Rank 1 for about half of questions or fewer (0.29–0.51), top five for 0.69–0.86",
        min(min(p["hit@1"].values()) for p in both) >= 0.285 and top1 <= 0.515
        and min(min(p["hit@5"].values()) for p in both) >= 0.685 and max(max(p["hit@5"].values()) for p in both) <= 0.865,
        f"hit@1 QASPER {_range(q['hit@1'].values())}, NQ {_range(n['hit@1'].values())}; "
        f"hit@5 QASPER {_range(q['hit@5'].values())}, NQ {_range(n['hit@5'].values())}", "numeric")
    medians = [sum(1 for v in p["median_rank"].values() if v == 2) for p in both]
    add("C3", "What replicates 1; Claim", "Median rank 2 for almost every model (near-miss structure)",
        min(medians) >= 5, f"models with median rank 2: QASPER {medians[0]}/7, NQ {medians[1]}/7; "
        f"median 1: QASPER {sum(1 for v in q['median_rank'].values() if v == 1)}, "
        f"NQ {sum(1 for v in n['median_rank'].values() if v == 1)}")
    add("C4", "What replicates 1; Claim", "Gold chunk at ranks 2–5 for a third to a half of questions",
        all(0.325 <= v <= 0.505 for p in both for v in p["share_ranks_2_to_5"].values()),
        f"QASPER {_range(q['share_ranks_2_to_5'].values())}, NQ {_range(n['share_ranks_2_to_5'].values())}",
        "numeric")
    ratio = [p["no_localiser_hit@1"] / p["no_localiser_hit@1_under_independence"] for p in both]
    add("C5", "What replicates 2; Claim",
        "Failures only partly shared: some model ranks the gold first for 0.77–0.78, best single model 0.40–0.51",
        all(p["any_localiser_hit@1"] - max(p["hit@1"].values()) >= 0.2 for p in both) and min(ratio) > 1,
        f"some model: QASPER {q['any_localiser_hit@1']:.3f}, NQ {n['any_localiser_hit@1']:.3f}; best single: "
        f"QASPER {max(q['hit@1'].values()):.3f}, NQ {max(n['hit@1'].values()):.3f}; no model vs independence: "
        f"{ratio[0]:.1f}x, {ratio[1]:.1f}x")
    weakest = [min(p["hit@1"], key=p["hit@1"].get) for p in both]
    add("C6", "What replicates 3", "BM25 is the weakest localiser on both corpora",
        weakest == ["bm25", "bm25"], f"weakest: QASPER {weakest[0]}, NQ {weakest[1]}")
    common = _losers(q) & _losers(n)
    add("C7", "What replicates 3", "BM25 is the only model with a Holm-significant deficit on both corpora",
        common == {"bm25"}, f"models losing a Holm-significant pair on both corpora: {sorted(common) or 'none'}")
    mpnet_below = all(p["hit@1"]["mpnet"] < min(p["hit@1"]["bge"], p["hit@1"]["e5"]) for p in both)
    mpnet_sig = [pair for p in both for pair in ("bge_vs_mpnet", "e5_vs_mpnet") if pair in p["holm_significant"]]
    add("C8", "What replicates 4; Claim",
        "MPNet-base (110M) below both 33M bi-encoders on both corpora, never significantly different",
        mpnet_below and not mpnet_sig,
        f"MPNet/BGE/E5 hit@1: QASPER {q['hit@1']['mpnet']:.3f}/{q['hit@1']['bge']:.3f}/{q['hit@1']['e5']:.3f}, "
        f"NQ {n['hit@1']['mpnet']:.3f}/{n['hit@1']['bge']:.3f}/{n['hit@1']['e5']:.3f}; "
        f"Holm-significant: {mpnet_sig or 'none'}")
    bi_sig = [f"{corpus}:{a}_vs_{b}" for corpus, p in (("qasper", q), ("nq", n))
              for i, a in enumerate(BI_ENCODERS) for b in BI_ENCODERS[i + 1:] if _pair(p, a, b) != 0]
    add("C8b", "Parameter count does not order the bi-encoders; Claim",
        "Differences among the four bi-encoders are not significant after correction, on either corpus",
        not bi_sig, f"Holm-significant bi-encoder pairs: {bi_sig or 'none'}")
    add("C9a", "What does not replicate; Claim", "On QASPER bge-reranker-base is the worst neural model",
        q["worst_neural"] == "bge_reranker_base", f"worst neural on QASPER: {q['worst_neural']}")
    add("C9b", "What does not replicate", "On QASPER bge-reranker-base is indistinguishable from BM25",
        _pair(q, "bm25", "bge_reranker_base") == 0,
        f"Holm-adjusted p {q['mcnemar_holm_adjusted_p']['bm25_vs_bge_reranker_base']}")
    add("C9c", "What does not replicate; Claim", "On NQ bge-reranker-base is the best model",
        n["best"] == "bge_reranker_base", f"best on NQ: {n['best']}")
    above = {m: _pair(n, "bge_reranker_base", m) for m in ("bm25", "minilm", "mpnet")}
    add("C9d", "What does not replicate",
        "On NQ bge-reranker-base is significantly above BM25, MiniLM and MPNet after correction",
        all(v == 1 for v in above.values()),
        "Holm-significant above: " + ", ".join(m for m, v in above.items() if v == 1))
    add("C9e", "What does not replicate", "On NQ the reranker's lead over BGE-small is not significant",
        _pair(n, "bge", "bge_reranker_base") == 0,
        f"Holm-adjusted p {n['mcnemar_holm_adjusted_p']['bge_vs_bge_reranker_base']}")
    best_bi = max(BI_ENCODERS, key=lambda m: q["hit@1"][m])
    add("C9f", "What does not replicate", "On QASPER the 22M ms-marco cross-encoder matches the best bi-encoder",
        _pair(q, "ce_msmarco_minilm", best_bi) == 0,
        f"ms-marco {q['hit@1']['ce_msmarco_minilm']:.3f} vs {best_bi} {q['hit@1'][best_bi]:.3f}, not significant"
        if _pair(q, "ce_msmarco_minilm", best_bi) == 0 else "significantly different")
    gains = [n["hit@1"]["bge_reranker_base"] - n["hit@1"][m] for m in BI_ENCODERS]
    add("C9g", "Claim (what it does not license)", "On NQ the reranker stage is worth +4 to +12 points at rank 1 "
        "over the bi-encoders", min(gains) >= 0.035 and max(gains) <= 0.125,
        f"reranker minus bi-encoders on NQ: {100 * min(gains):+.1f} to {100 * max(gains):+.1f} points", "numeric")
    lifts = [m for m in MODELS if n["lift@1"][m] <= q["lift@1"][m]]
    add("C10", "Why the corpora differ", "Every NQ lift over chance is higher than the QASPER lift",
        not lifts, f"lift@1 QASPER {_range(q['lift@1'].values())}, NQ {_range(n['lift@1'].values())}"
        + (f"; not higher for {lifts}" if lifts else ""))
    add("C11", "QASPER against NQ", "Holm-significant pairs: 2 on QASPER (both BM25 vs neural), 6 on NQ",
        len(q["holm_significant"]) == 2 and len(n["holm_significant"]) == 6
        and all("bm25" in pair for pair in q["holm_significant"]),
        f"QASPER {len(q['holm_significant'])}, NQ {len(n['holm_significant'])}", "numeric")
    unstable = [f"{corpus}:{pair}" for corpus, p in (("qasper", q), ("nq", n))
                for pair in p["holm_significant"]
                if not (p["cluster_bootstrap_ci95"][pair][0] > 0 or p["cluster_bootstrap_ci95"][pair][1] < 0)]
    add("C12", "Robustness checks",
        "Every Holm-significant pair keeps an interval excluding zero under the document-cluster bootstrap",
        not unstable, "all intervals exclude zero" if not unstable else f"interval includes zero: {unstable}")
    add("C13", "QASPER against NQ", "Gold spans per question: 2.09 on QASPER, 1.00 on NQ",
        (q["mean_gold_spans_per_question"], n["mean_gold_spans_per_question"]) == (2.09, 1.0),
        f"QASPER {q['mean_gold_spans_per_question']:.2f}, NQ {n['mean_gold_spans_per_question']:.2f}",
        "numeric")
    return out


def build(nq_dir: Path, qasper_dir: Path, frozen_dir: Path, n_boot: int) -> dict:
    data = {"nq": json.loads((nq_dir / nq.PER_QUESTION).read_text(encoding="utf-8")),
            "qasper": json.loads((qasper_dir / qasper.PER_QUESTION).read_text(encoding="utf-8"))}
    models = [m for m in MODELS if all(m in d["localisers"] for d in data.values())]
    profiles: dict[tuple[str, str], dict] = {}
    for view in VIEWS.values():
        for corpus in ("qasper", "nq"):
            key = (corpus, view[corpus])
            if key not in profiles:
                rows = corpus_rows(corpus, data[corpus]["questions"], models, view[corpus])
                profiles[key] = profile(rows, models, n_boot)
    # The manuscript pairing must give the manuscript's numbers back.
    check = {}
    for corpus, scheme in (("qasper", "union"), ("nq", "first")):
        stored = json.loads((frozen_dir / f"summary_{corpus}.json").read_text(encoding="utf-8"))
        ours = profiles[(corpus, scheme)]
        check[corpus] = all(ours["hit@1"][m] == stored["models"][m]["hit@1"]
                            and ours["hit@5"][m] == stored["models"][m]["hit@5"] for m in models)
    views = {}
    for name, view in VIEWS.items():
        q, n = profiles[("qasper", view["qasper"])], profiles[("nq", view["nq"])]
        views[name] = {"label": view["label"], "references": {"qasper": view["qasper"], "nq": view["nq"]},
                       "profiles": {"qasper": q, "nq": n}, "claims": claims(q, n)}
    return {"models": models, "manuscript_pairing_reproduces_summaries": check, "views": views}


def render_markdown(result: dict, resampling: dict) -> str:
    views = result["views"]
    names = list(views)
    first = views[names[0]]["claims"]
    lines = ["# The localisation claims under harmonised evidence references", "",
             "**Exploratory. Not part of the submitted manuscript.** Generated by "
             "`scripts/reference_sensitivity_claims.py` from the per-question ranks in "
             "`nq_reference_sensitivity/` and `qasper_reference_sensitivity/`. Claims are quoted from "
             "`results/localisation/README.md`; the manuscript pairing reproduces its numbers "
             f"(QASPER: {result['manuscript_pairing_reproduces_summaries']['qasper']}, "
             f"NQ: {result['manuscript_pairing_reproduces_summaries']['nq']}).", "",
             "| Pairing | QASPER reference | NQ reference |", "|---|---|---|"]
    for name in names:
        refs = views[name]["references"]
        lines.append(f"| {views[name]['label']} | {refs['qasper']} | {refs['nq']} |")
    def verdict(c: dict) -> str:
        if c["holds"]:
            return "holds"
        return "**outside stated range**" if c["kind"] == "numeric" else "**does not hold**"

    lines += ["", "## Claims", "", "| Claim (section of results/localisation/README.md) | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for i, claim in enumerate(first):
        cells = [verdict(views[name]["claims"][i]) for name in names]
        lines.append(f"| {claim['id']} {claim['claim']} ({claim['where']}) | " + " | ".join(cells) + " |")
    lines += ["", "### Numbers behind each verdict", ""]
    for i, claim in enumerate(first):
        lines.append(f"- **{claim['id']}** " + "; ".join(
            f"*{name}*: {views[name]['claims'][i]['detail']}" for name in names))
    lines += ["", "## Corpus contrast under each pairing", "",
              "| | " + " | ".join(f"{name}: QASPER | {name}: NQ" for name in names) + " |",
              "|---|" + "---:|" * (2 * len(names))]
    rows = [("questions", lambda p: str(p["n_questions"])),
            ("gold spans per question", lambda p: f"{p['mean_gold_spans_per_question']:.2f}"),
            ("gold chunks per question", lambda p: f"{p['mean_gold_chunks_per_question']:.2f}"),
            ("chance hit@1", lambda p: f"{p['chance_hit@1']:.3f}"),
            ("hit@1, seven models", lambda p: _range(p["hit@1"].values())),
            ("hit@5, seven models", lambda p: _range(p["hit@5"].values())),
            ("lift over chance at rank 1", lambda p: _range(p["lift@1"].values())),
            ("best model", lambda p: p["best"]),
            ("worst neural model", lambda p: p["worst_neural"]),
            ("spread across neural models (pp)", lambda p: f"{p['neural_spread_pp']:.1f}"),
            ("some model ranks gold first", lambda p: f"{p['any_localiser_hit@1']:.3f}"),
            ("Holm-significant pairs", lambda p: str(len(p["holm_significant"])))]
    for label, fn in rows:
        cells = [fn(views[name]["profiles"][corpus]) for name in names for corpus in ("qasper", "nq")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## One annotator drawn at random per question", "",
              f"- QASPER: bge-reranker-base is the worst neural model in "
              f"{resampling['qasper_reranker_worst_neural']:.1%} of {resampling['qasper_draws']} draws.",
              f"- NQ: bge-reranker-base is the best model in {resampling['nq_reranker_best']:.1%} of "
              f"{resampling['nq_draws']} draws.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nq-dir", default="results/exploratory/nq_reference_sensitivity")
    ap.add_argument("--qasper-dir", default="results/exploratory/qasper_reference_sensitivity")
    ap.add_argument("--frozen-dir", default="results/localisation")
    ap.add_argument("--out-dir", default="results/exploratory")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.resolve().is_relative_to((REPO / "results" / "localisation").resolve()):
        raise SystemExit("refusing to write into results/localisation/, which holds the manuscript's results")
    provenance = collect_provenance(script="scripts/reference_sensitivity_claims.py")
    nq_dir, qasper_dir = Path(args.nq_dir), Path(args.qasper_dir)
    result = build(nq_dir, qasper_dir, Path(args.frozen_dir), args.n_boot)
    nq_summary = json.loads((nq_dir / nq.SUMMARY_JSON).read_text(encoding="utf-8"))
    qasper_summary = json.loads((qasper_dir / qasper.SUMMARY_JSON).read_text(encoding="utf-8"))
    resampling = {
        "qasper_reranker_worst_neural":
            qasper_summary["single_annotator_resampling"]["share_of_draws_worst_neural_at_hit@1"]["bge_reranker_base"],
        "qasper_draws": qasper_summary["single_annotator_resampling"]["n_draws"],
        "nq_reranker_best": nq_summary["single_annotator_resampling"]["share_of_draws_best_at_hit@1"]["bge_reranker_base"],
        "nq_draws": nq_summary["single_annotator_resampling"]["n_draws"],
    }
    out = {"status": "exploratory; not part of the submitted manuscript",
           "inputs": {str(p.as_posix()): nq.sha256_lf(p) for p in
                      (nq_dir / nq.PER_QUESTION, qasper_dir / qasper.PER_QUESTION,
                       nq_dir / nq.SUMMARY_JSON, qasper_dir / qasper.SUMMARY_JSON)},
           "settings": {"n_boot": args.n_boot, "bootstrap_seed": 0},
           **result, "single_annotator_resampling": resampling, "provenance": provenance}
    out_dir.mkdir(parents=True, exist_ok=True)
    nq.write_lf(out_dir / OUT_JSON, json.dumps(out, indent=1) + "\n")
    nq.write_lf(out_dir / OUT_MD, render_markdown(result, resampling))
    print(f"wrote {out_dir / OUT_JSON} and {out_dir / OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
