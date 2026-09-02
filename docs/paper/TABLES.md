# Tables

Publication-ready tables. Every cell is copied from a run artifact in this
working tree; the source file and the command that regenerates it are named per
table. Numbers are identical to `results.md` — this file is the paper-facing
selection and ordering, not a second computation. Tables 1–2 come from `results/`,
which is tracked; the rest come from `reports/`, which is not (see
`reproducibility.md` §12).

Suggested paper placement is given for each. `results.md` holds several further
tables (full confusion matrices, per-category breakdowns of both variants) that
belong in an appendix rather than the body.

---

## Table 1 — Retrieval success under three definitions

**Placement:** Section 6.1, main result of the decomposition.
**Source:** `results/decomp_*.json`, `docs/EXPERIMENTS.md`.
**Regenerate:** `python scripts/reproduce_study.py --all`

| Corpus | n | median chunks / gold doc | A: document, any | B: document, quantified | C: span, quantified |
|---|---|---|---|---|---|
| QASPER dev | 290 | 19 | 0.441 | 0.441 | 0.276 |
| Natural Questions | 300 | 31 | 0.997 | 0.997 | 0.730 |
| HotpotQA | 150 | 2 | 0.993 | 0.507 | 0.507 |
| 2WikiMultihopQA | 150 | — | — | — | — |

A = at least one chunk from any relevant document. B = the corpus's quantifier
applied at document level (`all_required` for multi-hop). C = the same quantifier
applied to gold spans. `C ≤ B ≤ A` holds by construction; the finding is the size
of each drop.

Decomposed gaps with exact tests:

| Corpus | quantifier A→B | granularity B→C |
|---|---|---|
| QASPER dev | 0.0 pp | 16.6 pp (p = 7.1e-15) |
| Natural Questions | 0.0 pp | 26.7 pp (p = 1.7e-24) |
| HotpotQA | 48.7 pp (p = 2.1e-22) | 0.0 pp |
| 2WikiMultihopQA | 64.7 pp (p = 1.3e-29) | 1.3 pp (n.s., p = 0.5) |

> The two effects are orthogonal and each is null on the corpus where the other
> dominates. `results/twowiki_150.json` holds the 2Wiki experiment; a
> `decomp_twowiki_*.json` file is not present in `results/`, so the A/B/C point
> estimates for that corpus are left blank above rather than inferred.

## Table 2 — Attribution shift

**Placement:** Section 6.2 — the engineering consequence of Table 1.
**Source:** `docs/EXPERIMENTS.md`, `results/*.json`.

| Corpus | n | failures charged to retrieval, document-level | evidence-level |
|---|---|---|---|
| QASPER dev | 290 | 162 | 210 |
| Natural Questions | 300 | 1 | 81 |
| HotpotQA | 150 | 1 | 74 |
| 2WikiMultihopQA | 150 | 5 | 104 |

Same stored retrieval output in both columns; only the gate differs.

## Table 3 — Taxonomy against the 200-unit reference set (headline)

**Placement:** Section 6.3 — the paper's central table.
**Source:** `reports/annotation/qasper_dev_300_full_context/final_evaluation.json`.
**Regenerate:** `scripts/score_against_reference.py` — see `reproducibility.md` §7.

| Variant | Accuracy | Macro F1 | Cohen's kappa |
|---|---|---|---|
| Document-gated (`failure_mode_v2`) | 0.7400 | 0.6223 | 0.5728 |
| Evidence-gated (`failure_mode_evidence`) | **0.8050** | **0.6295** | **0.6305** |

Paired over the same 200 units: 139 both correct, 22 only evidence-gated, 9 only
document-gated, 30 neither. Exact McNemar on the 31 discordant pairs,
**p = 0.0294**.

## Table 4 — Per-category effect of the gate

**Placement:** Section 6.3, immediately after Table 3 — shows *where* the
improvement comes from and what it costs.
**Source:** same file.

| Category | Support | Doc-gated P / R / F1 | Evidence-gated P / R / F1 |
|---|---|---|---|
| `wrong_retrieval` | 130 | 1.000 / 0.769 / 0.870 | 0.917 / 0.938 / 0.928 |
| `incorrect_answer` | 42 | 0.561 / 0.762 / 0.646 | 0.727 / 0.571 / 0.640 |
| `ok` | 16 | 0.750 / 0.375 / 0.500 | 0.833 / 0.313 / 0.455 |
| `answered_when_unanswerable` | 9 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| `partial_answer` | 3 | 0.056 / 0.333 / 0.095 | 0.077 / 0.333 / 0.125 |
| `hallucination` | 0 | 0.000 / — / — | 0.000 / — / — |

Recall on `wrong_retrieval` rises 0.769 → 0.938; precision falls 1.000 → 0.917.
Categories with support < 30 are below the repository's `MIN_N_FOR_INFERENCE`
convention and should be read as descriptive.

## Table 5 — Context-integrity audit

**Placement:** Section 4.6 (method) or 6.4 (results) — a methodological finding
about annotation instruments.
**Source:** `reports/annotation/qasper_dev_300_full_context/TRUNCATION_AUDIT.json`.
**Regenerate:** `scripts/audit_annotation_truncation.py` — `reproducibility.md` §5.

| Quantity | Value |
|---|---|
| Annotation units audited | 200 |
| Retrieved chunks audited | 1,000 |
| Chunks cut at the 600-character display limit | 941 |
| Chunks recovered from source records | 941 |
| Chunks complete after rebuild | 1,000 / 1,000 |
| Chunks unreconstructable | 0 |
| Characters visible to the annotator | 588,671 → 1,163,638 |
| Characters previously hidden | 574,967 (49.4%) |

## Table 6 — Labels that moved when full context was restored

**Placement:** Section 6.4, beside Table 5 — the measured effect of the defect on
the labels themselves.
**Source:** `final_evaluation.json` → `agreement_with_other_passes.truncated_adjudicated`.

| | Value |
|---|---|
| Units compared | 200 |
| Identical labels | 187 (93.5%) |
| Changed labels | 13 |
| `wrong_retrieval` → `incorrect_answer` | 10 |
| `partial_answer` → `ok` | 3 |
| Changes in the opposite direction | 0 |
| Cohen's kappa | 0.8710 |

Every change runs from a retrieval label to a generation label; none runs the
other way.

## Table 7 — Annotation agreement

**Placement:** Section 6.5 — reliability of the labelling task. **Must be read
with `limitations.md` §1–2**: the reference set is model-generated, and no
inter-annotator statistic exists for the full-context package.
**Source:** `reports/annotation/qasper_dev_300/final_agreement_report.json`,
`final_evaluation.json`.

| Comparison | n | Raw agreement | Cohen's kappa |
|---|---|---|---|
| Two independent passes, **truncated** package | 200 | 0.925 | 0.8365 |
| Reference set vs truncated pass A | 200 | 0.890 | 0.7766 |
| Reference set vs truncated pass B | 200 | 0.905 | 0.8100 |
| Reference set vs truncated adjudicated labels | 200 | 0.935 | 0.8710 |
| Reference set vs the 22 human-labelled units | 22 | 0.909 | 0.7412 |

The last row is below `MIN_N_FOR_INFERENCE = 30` and is directional only.

## Table 8 — Generation replay by evidence status

**Placement:** Section 6.6 — why span-level coverage matters downstream.
**Source:** `docs/EXPERIMENTS.md`. Retrieval reused verbatim; only the generator
changes. n = 150 per model.

Qwen2.5-0.5B-Instruct:

| Evidence that reached the generator | n | correct | abstained | answered |
|---|---|---|---|---|
| Complete | 44 | 18.2% | 9.1% | 90.9% |
| Document retrieved, span missing | 26 | 0.0% | 0.0% | 100% |
| Nothing from any gold document | 75 | 1.3% | 4.0% | 96.0% |

| Generator | P(correct \| complete) | P(correct \| incomplete) | difference | p |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.182 | 0.010 | 17.2 pp | 0.0004 |
| SmolLM2-360M-Instruct | 0.136 | 0.030 | 10.7 pp | 0.023 |

The middle row of the first table is the set a document-level metric scores as
retrieval success. Both models are very small (`limitations.md` §8); no model
ranking or absolute quality claim is made.

## Table 9 — Experimental configuration (appendix)

**Placement:** appendix.
**Source:** `reports/experiments/qasper_dev_300/summary.json`,
`reports/annotation/qasper_dev_300_full_context/manifest.json`.

| Field | Value |
|---|---|
| Corpus / split | QASPER, dev (CC BY 4.0) |
| Questions / documents / chunks | 300 / 111 / 2,272 |
| Answerable / unanswerable | 290 / 10 |
| Embedder | `all-MiniLM-L6-v2` |
| Chunk size / overlap / top-k | 256 / 32 / 5 |
| Generator | `MockExtractiveLLM` (extractive control) |
| Offset mismatches | 0 |
| Annotation units / sampling seed | 200 / 20260826 |
| Chunks stored complete | 1,000 / 1,000 |
| Taxonomy version / fingerprint | v2.0 / `4672f4ea2b70` |

---

## Tables the paper would want that the repository cannot yet fill

Named here rather than drafted, per `limitations.md`:

- Inter-annotator agreement on the **full-context** package (needs a second pass).
- Human-vs-taxonomy agreement at n ≥ 30 (needs a human pass).
- The evidence-gating comparison on a second corpus.
- A threshold-tuning ablation on a held-out split of the reference set.
