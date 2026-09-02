"""Draw the pipeline-and-evaluation figure used in the README.

The figure is generated from committed result files, not drawn by hand: the
evaluation panel reads `final_evaluation.json` from an annotation package, and
the context-integrity panel reads that package's `TRUNCATION_AUDIT.json`. If a
number changes in the repository, re-running this script changes the figure.

    python scripts/make_pipeline_figure.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --out docs/figures/pipeline_evaluation.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1b2733"
MUTED = "#5b6b7c"
LINE = "#c7d2dd"
STAGE = "#eef3f8"
EVAL = "#e8f1ea"
ACCENT = "#2f6f4f"
WARN = "#b4552d"


def box(ax, x, y, w, h, title, subtitle, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.1, edgecolor=LINE, facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
            fontsize=8.4, color=MUTED)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.2, color=MUTED, shrinkA=0, shrinkB=0)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the README pipeline figure")
    parser.add_argument("--package", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    package = Path(args.package)
    evaluation = json.loads((package / "final_evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((package / "TRUNCATION_AUDIT.json").read_text(encoding="utf-8"))

    tvr = evaluation["taxonomy_vs_reference"]
    per_category = tvr["per_category"]
    totals = audit["totals"]

    fig = plt.figure(figsize=(13.6, 7.4), dpi=170)
    fig.patch.set_facecolor("white")

    # ---- top: the pipeline ------------------------------------------------
    top = fig.add_axes([0.0, 0.60, 1.0, 0.38])
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")

    top.text(0.035, 0.90, "TrustRAG — evidence-aware RAG evaluation",
             fontsize=15.5, color=INK, fontweight="bold")
    top.text(0.035, 0.775,
             "Character offsets travel the whole path, so “did the supporting passage reach the generator?” "
             "is arithmetic, not string search.",
             fontsize=9.6, color=MUTED)

    stages = [
        ("Question", "corpus_can_answer"),
        ("Chunker", "start_char / end_char"),
        ("ChromaDB", "vectors + offsets"),
        ("Retriever", "top-k, rank order"),
        ("Generator", "LLM or extractive control"),
    ]
    w, h, y = 0.163, 0.30, 0.30
    xs = [0.035 + i * 0.187 for i in range(len(stages))]
    for (title, sub), x in zip(stages, xs, strict=True):
        box(top, x, y, w, h, title, sub, STAGE)
    for x in xs[:-1]:
        arrow(top, x + w + 0.004, y + h / 2, x + 0.187 - 0.004, y + h / 2)

    top.text(0.035, 0.20, "retrieval", fontsize=8.6, color=MUTED, style="italic")
    top.text(0.60, 0.20, "generation", fontsize=8.6, color=MUTED, style="italic")

    # ---- middle: evaluation layer ----------------------------------------
    mid = fig.add_axes([0.0, 0.40, 1.0, 0.22])
    mid.set_xlim(0, 1)
    mid.set_ylim(0, 1)
    mid.axis("off")

    evals = [
        ("Evidence alignment", "gold span ∩ retrieved chunk"),
        ("Failure taxonomy v2", "9 categories, versioned rules"),
        ("Annotation package", "blinded, stratified, full context"),
        ("Reference scoring", "confusion matrix · per-category F1"),
    ]
    w2, h2, y2 = 0.213, 0.52, 0.24
    xs2 = [0.035 + i * 0.234 for i in range(len(evals))]
    for (title, sub), x in zip(evals, xs2, strict=True):
        box(mid, x, y2, w2, h2, title, sub, EVAL)
    for x in xs2[:-1]:
        arrow(mid, x + w2 + 0.004, y2 + h2 / 2, x + 0.234 - 0.004, y2 + h2 / 2)
    arrow(mid, 0.117, 1.02, 0.117, y2 + h2 + 0.02)

    # ---- bottom left: context integrity ----------------------------------
    left = fig.add_axes([0.045, 0.06, 0.28, 0.30])
    left.axis("off")
    left.text(0, 1.02, "Annotation context integrity", fontsize=11.5,
              color=INK, fontweight="bold", transform=left.transAxes)
    before = totals["displayed_chars_before"] / 1000
    after = totals["displayed_chars_after"] / 1000
    left.barh([1, 0], [before, after], color=[WARN, ACCENT], height=0.46)
    left.set_xlim(0, after * 1.02)
    left.set_ylim(-0.6, 1.6)
    left.text(before * 0.97, 1, f"{before:,.0f}k chars shown  ", va="center", ha="right",
              fontsize=9, color="white", fontweight="bold")
    left.text(after * 0.985, 0, f"{after:,.0f}k chars shown  ", va="center", ha="right",
              fontsize=9, color="white", fontweight="bold")
    left.text(0, 1.42, f"before — {audit['legacy_display_limit']}-char display cut",
              fontsize=8.8, color=MUTED)
    left.text(0, 0.42, "after — full retrieved chunk", fontsize=8.8, color=MUTED)
    left.text(0, -0.52,
              f"{totals['chunks_recovered']} of {totals['retrieved_chunks']} chunks recovered · "
              f"{totals['chunks_now_complete']}/{totals['retrieved_chunks']} complete · "
              f"{totals['chunks_unreconstructable']} unreconstructable",
              fontsize=8.6, color=MUTED)

    # ---- bottom right: taxonomy vs reference -----------------------------
    right = fig.add_axes([0.53, 0.09, 0.425, 0.27])
    order = [c for c in ["wrong_retrieval", "incorrect_answer", "ok",
                         "answered_when_unanswerable", "partial_answer"]
             if c in per_category and per_category[c]["support"]]
    f1s = [per_category[c]["f1"] or 0.0 for c in order]
    supports = [per_category[c]["support"] for c in order]
    ypos = list(range(len(order)))[::-1]
    right.barh(ypos, f1s, color=ACCENT, height=0.55)
    right.set_yticks(ypos)
    right.set_yticklabels([f"{c}  (n={s})" for c, s in zip(order, supports, strict=True)],
                          fontsize=8.8, color=INK)
    right.set_xlim(0, 1.0)
    right.set_xlabel("F1 against the reference set", fontsize=8.8, color=MUTED)
    right.tick_params(axis="x", labelsize=8.2, colors=MUTED)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        right.spines[spine].set_color(LINE)
    for y_, f1 in zip(ypos, f1s, strict=True):
        right.text(f1 + 0.015, y_, f"{f1:.2f}", va="center", fontsize=8.6, color=INK)
    right.text(0, 1.14,
               f"Taxonomy vs reference set — accuracy {tvr['accuracy']:.2f}, "
               f"macro F1 {tvr['macro_f1']:.2f}, n={tvr['n']}",
               fontsize=11.5, color=INK, fontweight="bold", transform=right.transAxes)

    fig.text(0.045, 0.015,
             "Generated by scripts/make_pipeline_figure.py from "
             f"{package.as_posix()}/final_evaluation.json and TRUNCATION_AUDIT.json.",
             fontsize=7.8, color=MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
