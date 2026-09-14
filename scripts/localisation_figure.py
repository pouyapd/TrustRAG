#!/usr/bin/env python
"""Rank-distribution figure for the within-document localisation study.

One panel per corpus; one horizontal stacked bar per model showing where the gold
chunk lands in the model's ranking of its own document (rank 1, 2, 3, 4-5, 6-10,
>10). An ordinal single-hue ramp encodes depth, darkest at rank 1; the chance level
for rank 1 is drawn as a reference line and hit@1 is labelled directly.

    python scripts/localisation_figure.py results/localisation/summary_qasper.json \
        results/localisation/summary_nq.json --out results/localisation/rank_distribution.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Ordinal ramp, steps 700/550/450/350/250/150 of the reference blue scale (darkest = rank 1).
RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
BUCKETS = ["1", "2", "3", "4-5", "6-10", ">10"]
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#d9d9d9"
LABELS = {
    "bm25": "BM25", "minilm": "MiniLM-L6 (22M)", "bge": "BGE-small (33M)", "e5": "E5-small (33M)",
    "mpnet": "MPNet-base (110M)", "ce_msmarco_minilm": "CE ms-marco MiniLM-L6 (22M)",
    "bge_reranker_base": "CE bge-reranker-base (278M)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panels = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.summaries]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, S in zip(axes, panels, strict=True):
        models = [k for k in LABELS if k in S["models"]]
        y = list(range(len(models)))[::-1]
        for yi, k in zip(y, models, strict=True):
            left = 0.0
            dist = S["models"][k]["rank_distribution"]
            for colour, b in zip(RAMP, BUCKETS, strict=True):
                w = dist[b]
                ax.barh(yi, w, left=left, height=0.62, color=colour, edgecolor="white", linewidth=1.5)
                left += w
            ax.text(S["models"][k]["hit@1"] - 0.012, yi, f"{S['models'][k]['hit@1']:.2f}",
                    va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
        chance = S["chance"]["hit@1"]
        ax.axvline(chance, color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)))
        ax.text(chance + 0.01, -0.75, f"chance hit@1 = {chance:.2f}", fontsize=8, color=MUTED,
                ha="left", va="center")
        ax.set_ylim(-1.05, len(models) - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[k] for k in models], fontsize=8.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8, color=MUTED)
        ax.set_xlabel("share of questions, by rank of the gold chunk inside its document", fontsize=8.5, color=MUTED)
        c = S["config"]
        ax.set_title(f"{c['dataset'].upper()}: n = {c['n_questions']}, "
                     f"{c['geometry']['mean_chunks_per_gold_doc']:.0f} chunks per document",
                     fontsize=10, color=INK, loc="left")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RAMP]
    fig.legend(handles, [f"rank {b}" for b in BUCKETS], loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where the gold chunk lands when each model ranks the chunks of its own document",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
