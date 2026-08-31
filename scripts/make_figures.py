"""Generate the study's figures from committed result files.

Four figures, chosen because each answers a question the tables answer less
directly. Anything a table already communicates clearly is not plotted.

    1. abc_decomposition   Where the gap comes from, per corpus. The A/B/C bars
                           make the point that the *same* retrieval output gives
                           very different success rates, and that the mechanism
                           differs by corpus.
    2. granularity_vs_k    Does the gap survive realistic retrieval depth? A
                           table of five k values invites reading the endpoints
                           only; the curve shows the shape.
    3. embedder_robustness Is the effect a property of one embedding model?
    4. attribution_shift   The consequence: where failures get charged, before
                           and after evidence-aware evaluation.

Conventions applied to all of them: sample sizes in the axis labels, Wilson
intervals wherever a proportion is drawn, axes starting at zero, no colour
carrying information that is not also carried by position or label, and a
caption stating what was held constant. Figures are written as PNG (for the
README) and PDF (vector, for a paper).

    pip install -r requirements-research.txt
    python scripts/make_figures.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REPORTS = REPO / "reports" / "experiments"
FIGURES = REPO / "results" / "figures"

#: One accessible colour per condition, used consistently across every figure.
COLOURS = {
    "A": "#4C72B0",   # document-level, ANY  (the conventional metric)
    "B": "#DD8452",   # document-level, quantified
    "C": "#55A868",   # span-level, quantified
    "gap": "#C44E52",
}


def load_decomp(tag: str) -> dict | None:
    path = REPORTS / f"decomp_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["comparison"]


def wilson_bounds(ci: dict, point: float) -> tuple[float, float]:
    """Asymmetric error-bar offsets from a Wilson interval."""
    lower = ci.get("lower")
    upper = ci.get("upper")
    if lower is None or upper is None:
        return 0.0, 0.0
    return max(0.0, point - lower), max(0.0, upper - point)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"  wrote results/figures/{name}.png and .pdf")


def figure_abc(plt) -> bool:
    """A/B/C per corpus, with Wilson intervals."""
    corpora = [
        ("QASPER\n(papers)", "qasper_dev_300"),
        ("Natural Questions\n(Wikipedia)", "nq_val_300_fixed"),
        ("HotpotQA\n(2-hop)", "hotpot_150"),
        ("2WikiMultihopQA\n(2-4 hop)", "twowiki_150"),
    ]
    loaded = [(label, load_decomp(tag)) for label, tag in corpora]
    loaded = [(label, c) for label, c in loaded if c]
    if not loaded:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.26
    conditions = [
        ("A_document_any", "A  document-level, ANY\n(the conventional metric)", COLOURS["A"]),
        ("B_document_quantified", "B  document-level, all required", COLOURS["B"]),
        ("C_span_quantified", "C  span-level (evidence actually retrieved)", COLOURS["C"]),
    ]
    for offset, (key, label, colour) in enumerate(conditions):
        xs = [i + (offset - 1) * width for i in range(len(loaded))]
        ys = [c["conditions"][key] for _, c in loaded]
        errs = list(zip(*[
            wilson_bounds(c["confidence_intervals"][key], c["conditions"][key])
            for _, c in loaded
        ], strict=True))
        ax.bar(xs, ys, width, label=label, color=colour,
               yerr=errs, capsize=3, error_kw={"lw": 1, "ecolor": "#444"})

    ax.set_xticks(range(len(loaded)))
    ax.set_xticklabels([
        f"{label}\nn={c['n_paired']}" for label, c in loaded
    ], fontsize=9)
    ax.set_ylabel("retrieval success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("The same retrieval output, three definitions of success",
                 fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.22,
        "Error bars are 95% Wilson intervals. Retrieval runs once per corpus; the three "
        "conditions are applied to the same stored output,\nso the comparison is paired. "
        "A→B is the quantifier effect (multi-hop only); B→C is the granularity effect "
        "(long documents only).",
        ha="center", fontsize=7.5, color="#333",
    )
    save(fig, "abc_decomposition")
    plt.close(fig)
    return True


def figure_topk(plt) -> bool:
    """Granularity and quantifier gaps against retrieval depth."""
    from scripts.reproduce_study import TOPK_VALUES

    series = [
        ("QASPER", "qasper_dev_300", "granularity_B_to_C", "o-"),
        ("Natural Questions", "nq_val_300_fixed", "granularity_B_to_C", "s-"),
        ("HotpotQA", "hotpot_150", "quantifier_A_to_B", "^--"),
        ("2WikiMultihopQA", "twowiki_150", "quantifier_A_to_B", "d--"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    drawn = 0
    for label, base, step, style in series:
        xs, ys = [], []
        for k in TOPK_VALUES:
            c = load_decomp(f"{base}_topk_k{k}")
            if c:
                xs.append(k)
                ys.append(c["steps"][step]["absolute_gap_pp"])
        if not xs:
            continue
        kind = "granularity" if step.startswith("gran") else "quantifier"
        ax.plot(xs, ys, style, label=f"{label} — {kind}", linewidth=1.8, markersize=6)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return False

    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(list(TOPK_VALUES))
    ax.set_xticklabels([str(k) for k in TOPK_VALUES])
    ax.set_xlabel("retrieval depth k (log scale)")
    ax.set_ylabel("gap (percentage points)")
    ax.set_ylim(bottom=0)
    ax.set_title("Both effects persist across retrieval depth — with very different slopes",
                 fontsize=11.5, pad=12)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.13,
        "Every depth is retrieved natively — the query is re-issued at each k rather than one "
        "deep ranking being truncated. Corpus, chunking (256/32)\nand embedder (MiniLM) are held "
        "constant; only k varies. Solid: granularity (B→C). Dashed: quantifier (A→B). "
        "Every point is significant (exact McNemar, p < 1e-6).\nThe k=1 multi-hop points are "
        "definitional — a 2-hop question cannot hold both required documents in one slot — and "
        "carry no evidential weight.",
        ha="center", fontsize=7.5, color="#333",
    )
    save(fig, "gap_vs_topk")
    plt.close(fig)
    return True


def figure_embedders(plt) -> bool:
    """The effect under four independently trained embedding models."""
    from src.rag.embedders import EMBEDDERS

    keys = ("minilm", "mpnet", "bge", "e5")
    panels = [
        ("QASPER — granularity (B→C)", "qasper_dev_300", "granularity_B_to_C"),
        ("HotpotQA — quantifier (A→B)", "hotpot_150", "quantifier_A_to_B"),
    ]
    available = [
        (title, [(k, load_decomp(f"{base}_emb_{k}")) for k in keys], step)
        for title, base, step in panels
    ]
    available = [
        (title, [(k, c) for k, c in rows if c], step) for title, rows, step in available
    ]
    available = [item for item in available if item[1]]
    if not available:
        return False

    fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.4))
    if len(available) == 1:
        axes = [axes]
    for ax, (title, rows, step) in zip(axes, available, strict=True):
        labels = [EMBEDDERS[k].repo_id.split("/")[-1] for k, _ in rows]
        gaps = [c["steps"][step]["absolute_gap_pp"] for _, c in rows]
        ax.bar(range(len(rows)), gaps, color=COLOURS["gap"], width=0.6)
        top = max(gaps)
        for i, (gap, (_, c)) in enumerate(zip(gaps, rows, strict=True)):
            p = c["steps"][step]["mcnemar"]["p_value"]
            ax.text(i, gap + top * 0.03, f"{gap:.1f}", ha="center", fontsize=9.5)
            # p-values sit inside the bar: below the axis they collide with the
            # rotated model names, which are the labels a reader needs first.
            if p is not None:
                ax.text(i, top * 0.04, f"p={p:.0e}", ha="center", fontsize=7,
                        color="white")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("gap (percentage points)")
        ax.set_ylim(0, top * 1.15)
        ax.set_title(f"{title}   (n={rows[0][1]['n_paired']})", fontsize=10.5, pad=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Both effects replicate under four embedders — "
                 "but only one of them is stable", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.text(
        0.5, -0.13,
        "Four models from three training lineages (Sentence-Transformers, BAAI BGE, Microsoft E5). "
        "Corpus, questions, chunking and k=5 held constant;\nonly the embedder changes. E5 and BGE "
        "are asymmetric and are called with their documented query/passage prefixes. "
        "p-values are exact McNemar.\nThe granularity gap varies little (14.5-18.3 pp); the "
        "quantifier gap nearly halves under the two instruction-trained retrievers.",
        ha="center", fontsize=7.5, color="#333",
    )
    save(fig, "embedder_robustness")
    plt.close(fig)
    return True


def figure_attribution(plt) -> bool:
    """Where failures are charged, under each reading of retrieval success."""
    corpora = [
        ("QASPER", "qasper_dev_300"),
        ("Natural Questions", "nq_val_300_fixed"),
        ("HotpotQA", "hotpot_150"),
        ("2WikiMultihopQA", "twowiki_150"),
    ]
    loaded = [(label, load_decomp(tag)) for label, tag in corpora]
    loaded = [(label, c) for label, c in loaded if c]
    if not loaded:
        return False

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.36
    for offset, (view, colour) in enumerate(
        [("document_level", COLOURS["A"]), ("evidence_level", COLOURS["C"])]
    ):
        xs = [i + (offset - 0.5) * width for i in range(len(loaded))]
        ys = [c["attribution"][view].get("retrieval", 0) for _, c in loaded]
        label = ("document-level reading" if view == "document_level"
                 else "evidence-level reading")
        bars = ax.bar(xs, ys, width, label=label, color=colour)
        ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(range(len(loaded)))
    ax.set_xticklabels([f"{label}\nn={c['n_paired']}" for label, c in loaded], fontsize=9)
    ax.set_ylabel("failures charged to retrieval")
    ax.set_title("Evidence-aware evaluation moves the blame", fontsize=12, pad=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.10,
        "Counts, not rates: the same runs, re-attributed. A document-level reading credits "
        "retrieval whenever any chunk of a relevant\ndocument appears; the evidence-level reading "
        "requires the supporting span itself. The difference is work an engineer would "
        "otherwise\nspend on the generator.",
        ha="center", fontsize=7.5, color="#333",
    )
    save(fig, "attribution_shift")
    plt.close(fig)
    return True


FIGURE_BUILDERS = {
    "abc": figure_abc,
    "topk": figure_topk,
    "embedders": figure_embedders,
    "attribution": figure_attribution,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the study's figures")
    parser.add_argument("--all", action="store_true", help="build every available figure")
    parser.add_argument("--figure", action="append", default=[],
                        choices=sorted(FIGURE_BUILDERS), help="build one figure (repeatable)")
    args = parser.parse_args()

    if not (args.all or args.figure):
        parser.print_help()
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. pip install -r requirements-research.txt",
              file=sys.stderr)
        return 1

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    chosen = sorted(FIGURE_BUILDERS) if args.all else args.figure
    built, skipped = 0, []
    for name in chosen:
        print(f"[{name}]")
        if FIGURE_BUILDERS[name](plt):
            built += 1
        else:
            skipped.append(name)
            print("  skipped - the runs it needs are not present")

    print(f"\n{built} figure(s) written to results/figures/")
    if skipped:
        print(f"skipped (missing runs): {', '.join(skipped)}")
        print("run `python scripts/reproduce_study.py --everything` first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
