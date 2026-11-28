"""README results figure for TrustRAG, drawn only from the verified result files
(results/<corpus>_full/analysis.json, results/audit/cluster_inference.json, statistical_audit.json, position_control.json)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ST = Path(sys.argv[1])          # studies/reference_sensitivity_2026-11 (or the anonymized mirror's study/analysis)
OUT = Path(sys.argv[2])
A = ST / "results/audit"
sa = json.loads((A / "statistical_audit.json").read_text(encoding="utf-8"))
pc = json.loads((A / "position_control.json").read_text(encoding="utf-8"))
an = {c: json.loads((ST / f"results/{c}_full/analysis.json").read_text(encoding="utf-8")) for c in ("qasper", "nq")}
# same values as Table 3: intervals from analysis.json, point estimate exact from cluster_inference.json
ci = json.loads((A / "cluster_inference.json").read_text(encoding="utf-8"))
EFF = {"qasper": ("first_minus_union", -1), "nq": ("union_minus_first", 1)}

ORDER = ["bm25", "minilm", "granite30m", "colbert_small", "ce_msmarco_minilm", "jina_turbo"]
NAMES = {"bm25": "BM25", "minilm": "MiniLM-L6", "granite30m": "Granite-30M", "colbert_small": "ColBERT-small",
         "ce_msmarco_minilm": "ms-marco CE", "jina_turbo": "Jina-turbo"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#bbbbbb",
                     "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT, "text.color": TEXT})

fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=150, gridspec_kw={"hspace": 0.55, "wspace": 0.32})
for col, (corpus, label) in enumerate((("qasper", "QASPER (2,197 questions)"), ("nq", "Natural Questions (1,278 questions)"))):
    key, sign = EFF[corpus]
    eff = an[corpus]["reference_effects"][key]["models"]
    models = [m for m in ORDER if m in eff]
    y = list(range(len(models)))[::-1]
    # row 1: union minus first-annotator hit@1, cluster bootstrap 95% interval
    ax = axes[0, col]
    for yi, m in zip(y, models):
        r = eff[m]
        lo, hi = sorted(100 * sign * v for v in r["cluster95"])
        d = 100 * ci[corpus]["reference_effects"]["union_minus_first"][m]["delta"]
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(d, yi, "o", ms=8, color=BLUE, mec="white", mew=1.5)
        ax.text(hi + 0.25, yi, f"{d:+.1f}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, [NAMES[m] for m in models])
    ax.set_xlim(0, 12)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("union minus first-annotator hit@1 (points)")
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # row 2: real gain vs uniform and position-matched random gain
    ax = axes[1, col]
    g, p = sa[corpus]["G_random_gold_control"]["models"], pc[corpus]["models"]
    for yi, m in zip(y, models):
        real = 100 * g[m]["gain_real_union"]
        ulo, uhi = (100 * v for v in g[m]["gain_random_extra_95"])
        plo, phi = (100 * v for v in p[m]["gain_matched_95"])
        ax.plot([ulo, uhi], [yi + 0.18] * 2, color=ORANGE, lw=2, solid_capstyle="round")
        ax.plot(100 * g[m]["gain_random_extra_mean"], yi + 0.18, "s", ms=7, color=ORANGE, mec="white", mew=1.2)
        ax.plot([plo, phi], [yi - 0.18] * 2, color=AQUA, lw=2, solid_capstyle="round")
        ax.plot(100 * p[m]["gain_matched_mean"], yi - 0.18, "^", ms=8, color=AQUA, mec="white", mew=1.2)
        ax.plot(real, yi, "o", ms=8, color=BLUE, mec="white", mew=1.5)
        ax.text(real + 0.6, yi, f"{real:.1f}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, [NAMES[m] for m in models])
    ax.set_xlim(0, 34)
    ax.set_xlabel("gain in hit@1 on questions where the union adds chunks (points)")
    n = sa[corpus]["G_random_gold_control"]["n_questions_union_adds_chunks"]
    ax.set_title(f"Random-gold controls ({n:,} questions)", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

h = [plt.Line2D([], [], marker="o", color=BLUE, ls="", ms=8, label="real: other annotators' evidence added"),
     plt.Line2D([], [], marker="s", color=ORANGE, ls="-", lw=2, ms=7, label="uniform random chunks (mean, 95% range of 200 draws)"),
     plt.Line2D([], [], marker="^", color=AQUA, ls="-", lw=2, ms=8, label="position-matched random chunks (mean, 95% range)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.text(0.01, 0.985, "Top: effect of the evidence reference, with cluster bootstrap 95% intervals (QASPER: paper; NQ: Wikipedia page).",
         fontsize=9, color=TEXT2, va="top")
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
