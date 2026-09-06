#!/usr/bin/env python
"""Figures for the paper, each drawn from a committed result file.

Every number is read at render time. If a source file changes, the figure changes
with it; nothing here is hard-coded.

    python scripts/make_paper_figures.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "figures"

DENSE = "#2f6fd0"
LEX = "#a8620a"
GREY = "#8c8f96"
GOOD = "#1d7a4c"
INK = "#1a1c1f"


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def setup(plt):
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif"],
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.edgecolor": "#444",
    })


def figure_bm25(plt) -> bool:
    """Dense vs BM25 under document- and span-level definitions."""
    corpora = [("QASPER dev", "results/decomp_qasper_dev_300.json",
                "results/bm25_qasper_dev_300.json"),
               ("Natural Questions", "results/decomp_nq_val_300_fixed.json",
                "results/bm25_nq_val_300.json"),
               ("HotpotQA", "results/decomp_hotpot_150.json",
                "results/bm25_hotpot_150.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, cond, title in zip(
            axes,
            ("A_document_any", "C_span_quantified"),
            ("Document-level (A): any chunk from a relevant document",
             "Span-level (C): a chunk containing the gold span"), strict=True):
        names, dense_v, bm25_v = [], [], []
        for label, dense_path, bm25_path in corpora:
            d = load(dense_path)["comparison"]["conditions"]
            b = load(bm25_path)["conditions"]
            names.append(label)
            dense_v.append(d[cond])
            bm25_v.append(b[cond])
        x = range(len(names))
        w = 0.36
        for off, vals, colour, lab in ((-w / 2, dense_v, DENSE, "dense (MiniLM)"),
                                       (w / 2, bm25_v, LEX, "BM25")):
            bars = ax.bar([i + off for i in x], vals, w, color=colour, label=lab)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=8.2, loc="left", color=INK)
        ax.set_ylabel("share of answerable questions")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].annotate("BM25 ahead", xy=(0, 0.60), fontsize=7.5, color=LEX, ha="center")
    axes[1].annotate("dense ahead", xy=(0, 0.36), fontsize=7.5, color=DENSE, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "bm25_vs_dense.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_oracle(plt) -> bool:
    """Paired oracle-evidence repair rate by evidence stratum."""
    r = load("reports/experiments/oracle_qasper_qwen/summary.json")
    order = ["COMPLETE", "NONE_DOC_HIT", "NONE"]
    pretty = {"COMPLETE": "Evidence complete\nunder retrieval",
              "NONE_DOC_HIT": "Document retrieved,\nspan missing",
              "NONE": "Nothing from any\ngold document"}
    strata = [s for s in order if r["by_stratum"].get(s, {}).get("n")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = range(len(strata))
    w = 0.36
    for off, key, colour, lab in ((-w / 2, "retrieved_rate", GREY, "retrieved context"),
                                  (w / 2, "oracle_rate", GOOD, "oracle: gold spans supplied")):
        vals = [r["by_stratum"][s][key] for s in strata]
        bars = ax.bar([i + off for i in x], vals, w, color=colour, label=lab)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for i, s in enumerate(strata):
        v = r["by_stratum"][s]
        ax.text(i, max(v["retrieved_rate"], v["oracle_rate"]) + 0.055,
                f"{v['difference_pp']:+.1f} pp\np = {v['paired_test']['p_value']:.3g}",
                ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{pretty[s]}\nn = {r['by_stratum'][s]['n']}" for s in strata],
                       fontsize=7.8)
    ax.set_ylim(0, 0.48)
    ax.set_ylabel("answers with all\nreference key facts")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Within-question paired oracle control — same generator, prompt and "
                 "decoding; only the context differs",
                 fontsize=8.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "oracle_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_human(plt) -> bool:
    """Both gates scored against the final human-reviewed labels."""
    r = load("reports/annotation/qasper_dev_300_full_context/final_human_reviewed/"
             "headline_vs_final_human.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names = ["Accuracy", "Macro F1", "Cohen's $\\kappa$"]
    d = [r["document_gated"]["accuracy"], r["document_gated"]["macro_f1"],
         r["document_gated"]["kappa"]]
    e = [r["evidence_gated"]["accuracy"], r["evidence_gated"]["macro_f1"],
         r["evidence_gated"]["kappa"]]
    x = range(3)
    w = 0.36
    for off, vals, colour, lab in ((-w / 2, d, GREY, "document-gated"),
                                   (w / 2, e, DENSE, "evidence-gated")):
        bars = ax.bar([i + off for i in x], vals, w, color=colour, label=lab)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("agreement with human labels")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Against the final human-reviewed labels (n = 200)",
                 fontsize=8.5, loc="left", color=INK)

    ax2 = axes[1]
    p = r["paired"]
    labels = ["both\ncorrect", "only\nevidence", "only\ndocument", "neither"]
    vals = [p["both"], p["only_evidence"], p["only_document"], p["neither"]]
    bars = ax2.bar(labels, vals, color=["#c9cdd4", DENSE, GREY, "#e4e6ea"], width=0.66)
    for bar, v in zip(bars, vals, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(v),
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, max(vals) * 1.3)
    ax2.set_ylabel("units")
    ax2.set_title(f"Paired — exact McNemar $p$ = {p['mcnemar']['p_value']:.2g}",
                  fontsize=8.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "human_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_goldspan(plt) -> bool:
    """How far the span-based gold standard can be trusted."""
    r = load("reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json")
    order = [("A_genuinely_unsupported", "Both signals agree:\nanswer absent\n(span rule correct)", GOOD),
             ("B_supported_outside_gold_span", "Both agree: answer present\noutside the gold span\n(span rule wrong)", "#b23b3b"),
             ("C_possibly_inferable", "Semantically close,\nlexically different", "#d9a441"),
             ("D_ambiguous_lexical_only", "Lexical overlap only", "#c9a227"),
             ("D_ambiguous", "Signals disagree or\nboth mid-range", GREY)]
    present = [(k, lab, c) for k, lab, c in order if r["buckets"].get(k)]
    vals = [r["buckets"][k] for k, _, _ in present]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    left = 0
    for (_key, _label, colour), v in zip(present, vals, strict=True):
        ax.barh([0], [v], left=left, color=colour, height=0.52, edgecolor="white")
        ax.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        left += v
    ax.set_xlim(0, sum(vals))
    ax.set_ylim(-0.6, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"answerable units with zero gold-span coverage (n = {r['n_units']})",
                  fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in present]
    ax.legend(handles, [lab for _, lab, _ in present], frameon=False, fontsize=6.8,
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42), handlelength=1.1)
    fig.tight_layout()
    fig.savefig(OUT / "gold_span_validity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


BUILDERS = {"bm25": figure_bm25, "oracle": figure_oracle,
            "human": figure_human, "goldspan": figure_goldspan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--figure", action="append", default=[], choices=sorted(BUILDERS))
    args = ap.parse_args()
    if not (args.all or args.figure):
        ap.print_help()
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: pip install -r requirements-research.txt", file=sys.stderr)
        return 1
    setup(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sorted(BUILDERS) if args.all else args.figure):
        BUILDERS[name](plt)
        print(f"  wrote results/figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
