# Figures

Five figures exist as files. `results/figures/` is tracked by git;
`docs/figures/` is new in the working tree and not gitignored. Each entry gives what it shows, what it is
evidence *for*, the data it is drawn from, and the command that regenerates it.
Regenerating requires `pip install -r requirements-research.txt` (matplotlib).
All figures are written at 200 dpi; `scripts/make_figures.py` emits PDF beside
PNG for camera-ready use.

---

## Figure 1 — A/B/C retrieval decomposition

**File:** `results/figures/abc_decomposition.png` (+ `.pdf`)
**Command:** `python scripts/make_figures.py --figure abc`
**Data:** `results/decomp_*.json`
**Shows:** grouped bars of retrieval success under the three definitions per
corpus, with Wilson 95% intervals.
**Evidence for:** the two gaps are separable, and each is null on the corpus
where the other dominates — the quantifier effect on multi-hop corpora,
the granularity effect on long-document corpora.
**Pairs with:** Table 1. **Placement:** Section 6.1.

## Figure 2 — Attribution shift

**File:** `results/figures/attribution_shift.png` (+ `.pdf`)
**Command:** `python scripts/make_figures.py --figure attribution`
**Data:** `results/*.json`
**Shows:** failures charged to retrieval under a document-level versus an
evidence-level gate, per corpus.
**Evidence for:** the paper's motivating claim. On Natural Questions the same
stored run yields 1 versus 81 retrieval failures depending only on the gate.
**Pairs with:** Table 2. **Placement:** Section 1 or 6.2 — strong candidate for
the teaser figure, since it states the problem in one panel.

## Figure 3 — Granularity gap versus retrieval depth

**File:** `results/figures/gap_vs_topk.png` (+ `.pdf`)
**Command:** `python scripts/make_figures.py --figure topk`
**Data:** top-k sweep, k ∈ {1, 3, 5, 10, 20}
**Shows:** the B→C gap as a function of k.
**Evidence for:** the gap is largest where retrieval budgets are tightest and
narrows as k grows (NQ: 57.3 pp at k=1 → 7.7 pp at k=20). This is a scope
statement as much as a result — it bounds when the distinction matters.
**Placement:** Section 6.1 or appendix.

## Figure 4 — Embedder robustness

**File:** `results/figures/embedder_robustness.png` (+ `.pdf`)
**Command:** `python scripts/make_figures.py --figure embedders`
**Data:** four embedding models on the same questions
**Shows:** the decomposition repeated per embedder.
**Evidence for:** the gap is a property of the corpus and the metric definition,
not of one retriever. All four models are small and English (`limitations.md` §9).
**Placement:** appendix.

## Figure 5 — Pipeline and evaluation overview

**File:** `docs/figures/pipeline_evaluation.png`
**Command:**
```bash
python scripts/make_pipeline_figure.py \
    --package reports/annotation/qasper_dev_300_full_context \
    --out docs/figures/pipeline_evaluation.png
```
**Data:** `final_evaluation.json`, `TRUNCATION_AUDIT.json` — every number in the
figure is read from those files at render time, none is hard-coded.
**Shows:** four panels — the offset-carrying pipeline; the evaluation loop
(stored run → two gates → reference set); context integrity before and after the
truncation fix; per-category F1 for both gates.
**Evidence for:** orientation rather than a single result; it is the figure used
in the README.
**Placement:** Section 4 (method overview), or drop from the paper if the
individual panels are promoted to Figures 6 and 7 below.

---

## Figures the paper would want that do not exist yet

Listed, not drafted. Each is cheap to add from data already produced; none is
claimed as existing.

| Proposed figure | Would show | Data available? |
|---|---|---|
| Paired variant comparison | The 139 / 22 / 9 / 30 split of Table 3 as a 2×2, with the McNemar result | Yes — `final_evaluation.json` |
| Side-by-side confusion matrices | Document-gated vs evidence-gated as heatmaps; makes the 22-unit `wrong_retrieval` recovery visible in one glance | Yes — same file |
| Generation accuracy by evidence status | Table 8's three strata as bars with intervals | Yes — `docs/EXPERIMENTS.md` artifacts |
| Human-vs-taxonomy agreement | Reliability of the reference set at n ≥ 30 | **No** — requires a human pass (`limitations.md` §1) |
| Evidence gating on a second corpus | Generality of the headline result | **No** — requires a second annotation package (`limitations.md` §4) |

The first three are figure work only, no new experiments. The last two require
the missing work named in `limitations.md`.

---

## Conventions

- Colour is used to distinguish conditions, never as the sole carrier of meaning;
  every series is also labelled.
- Error bars are Wilson 95% intervals for proportions; where a comparison is
  paired, the exact test result is stated in the caption rather than implied by
  overlapping intervals.
- Captions in the paper must repeat n and the source corpus — several figures mix
  corpora with different n.
