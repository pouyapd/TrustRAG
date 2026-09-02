# Results

Measured values only. Interpretation is confined to blocks marked
**Interpretation**. Source file is named for every table.

---

## 1. Taxonomy against the 200-unit reference set

Source: `reports/annotation/qasper_dev_300_full_context/final_evaluation.json`.

### 1.1 Headline

| Variant | Accuracy | Macro F1 | Cohen's kappa | Observed agreement |
|---|---|---|---|---|
| Document-gated (`failure_mode_v2`) | 0.7400 | 0.6223 | 0.5728 | 0.740 |
| **Evidence-gated (`failure_mode_evidence`)** | **0.8050** | **0.6295** | **0.6305** | 0.805 |

Paired over the same 200 units:

| | n |
|---|---|
| Both variants correct | 139 |
| Only evidence-gated correct | **22** |
| Only document-gated correct | 9 |
| Neither correct | 30 |

Exact McNemar on the 31 discordant pairs: **p = 0.0294**.

> **Interpretation.** Changing one gate — from "did a chunk from a relevant
> document arrive?" to "did a chunk covering the gold span arrive?" — moves the
> taxonomy closer to an independent reading of the same 200 units, and the
> improvement is not attributable to chance at the 5% level. This is the
> repository's most direct evidence for the central claim.

### 1.2 Per-category, document-gated variant

| Category | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `answered_when_unanswerable` | 9 | 9 | 1.0000 | 1.0000 | 1.0000 |
| `wrong_retrieval` | 130 | 100 | 1.0000 | 0.7692 | 0.8696 |
| `incorrect_answer` | 42 | 57 | 0.5614 | 0.7619 | 0.6465 |
| `ok` | 16 | 8 | 0.7500 | 0.3750 | 0.5000 |
| `partial_answer` | 3 | 18 | 0.0556 | 0.3333 | 0.0952 |
| `hallucination` | 0 | 8 | 0.0000 | — | — |

### 1.3 Per-category, evidence-gated variant

| Category | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `answered_when_unanswerable` | 9 | 9 | 1.0000 | 1.0000 | 1.0000 |
| `wrong_retrieval` | 130 | 133 | 0.9173 | 0.9385 | 0.9278 |
| `incorrect_answer` | 42 | 33 | 0.7273 | 0.5714 | 0.6400 |
| `ok` | 16 | 6 | 0.8333 | 0.3125 | 0.4545 |
| `partial_answer` | 3 | 13 | 0.0769 | 0.3333 | 0.1250 |
| `hallucination` | 0 | 6 | 0.0000 | — | — |

### 1.4 Confusion matrix, document-gated (rows = reference, columns = taxonomy)

| ref \ sys | ans_unans | halluc | incorrect | ok | partial | wrong_retr |
|---|---|---|---|---|---|---|
| `answered_when_unanswerable` | 9 | 0 | 0 | 0 | 0 | 0 |
| `hallucination` | 0 | 0 | 0 | 0 | 0 | 0 |
| `incorrect_answer` | 0 | 3 | 32 | 0 | 7 | 0 |
| `ok` | 0 | 3 | 2 | 6 | 5 | 0 |
| `partial_answer` | 0 | 0 | 1 | 1 | 1 | 0 |
| `wrong_retrieval` | 0 | 2 | 22 | 1 | 5 | 100 |

### 1.5 Confusion matrix, evidence-gated

| ref \ sys | ans_unans | halluc | incorrect | ok | partial | wrong_retr |
|---|---|---|---|---|---|---|
| `answered_when_unanswerable` | 9 | 0 | 0 | 0 | 0 | 0 |
| `hallucination` | 0 | 0 | 0 | 0 | 0 | 0 |
| `incorrect_answer` | 0 | 3 | 24 | 0 | 7 | 8 |
| `ok` | 0 | 3 | 1 | 5 | 4 | 3 |
| `partial_answer` | 0 | 0 | 1 | 1 | 1 | 0 |
| `wrong_retrieval` | 0 | 0 | 7 | 0 | 1 | 122 |

---

## 2. Context-integrity audit

Source: `reports/annotation/qasper_dev_300_full_context/TRUNCATION_AUDIT.json`.

| Quantity | Value |
|---|---|
| Annotation units audited | 200 |
| Retrieved chunks audited | 1,000 |
| Chunks cut at the 600-character display limit | 941 |
| Chunks already complete | 59 |
| Chunks recovered from source records | 941 |
| Chunks complete after rebuild | 1,000 / 1,000 |
| Chunks unreconstructable | 0 |
| Units affected | 200 / 200 |
| Characters visible to the annotator | 588,671 → 1,163,638 |
| Characters previously hidden | 574,967 |

> **Interpretation.** 49.4% of the retrieved text the generator saw was not shown
> to the annotator, while the sheet still displayed the full `char_range`.
> Step 2 of the guidelines — *did the evidence reach the system?* — was therefore
> not answerable from the sheet as displayed.

---

## 3. Effect of restoring full context on the labels

Comparison of the full-context reference set against the adjudicated labels from
the truncated package (`reports/annotation/qasper_dev_300/final_adjudicated_labels.jsonl`).

| | Value |
|---|---|
| Units compared | 200 |
| Identical labels | 187 (93.5%) |
| Changed labels | 13 |
| `wrong_retrieval` → `incorrect_answer` | 10 |
| `partial_answer` → `ok` | 3 |
| Changes in the opposite direction | 0 |
| Cohen's kappa | 0.8710 |

Changed unit ids: `unit_0027`, `unit_0032`, `unit_0035`, `unit_0041`,
`unit_0070`, `unit_0083`, `unit_0114`, `unit_0122`, `unit_0128`, `unit_0141`,
`unit_0146`, `unit_0193`, `unit_0194`.

> **Interpretation.** Every change ran from a retrieval label to a generation
> label. Truncated context biases annotation toward blaming retrieval, because
> evidence hidden past the cut is indistinguishable from evidence never
> retrieved.

---

## 4. Annotation agreement

Source: `reports/annotation/qasper_dev_300/final_agreement_report.json` and
`final_evaluation.json`.

| Comparison | n | Raw agreement | Cohen's kappa |
|---|---|---|---|
| Two independent passes over the **truncated** package | 200 | 0.925 | 0.8365 |
| Reference set vs truncated pass A | 200 | 0.890 | 0.7766 |
| Reference set vs truncated pass B | 200 | 0.905 | 0.8100 |
| Reference set vs truncated adjudicated labels | 200 | 0.935 | 0.8710 |
| Reference set vs the 22 human-labelled units | 22 | 0.909 | 0.7412 |

The 15 disagreements between the two truncated passes were adjudicated against
the written guidelines; the adjudicated distribution was `incorrect_answer` 8,
`ok` 4, `wrong_retrieval` 3.

> **Interpretation and caveat.** The kappa of 0.8365 shows the *task* is
> well-defined enough for two independent passes to agree, but it was measured on
> the truncated package and both passes were model-produced. The n=22 human
> comparison is below the repository's own `MIN_N_FOR_INFERENCE = 30` and is
> reported as directional only.

---

## 5. Error analysis

### 5.1 Misattributed retrieval failures — the dominant error

Under the document-gated variant, 30 units the reference calls `wrong_retrieval`
are charged to generation (22 `incorrect_answer`, 5 `partial_answer`, 2
`hallucination`, 1 `ok`). Joining those units back to the stored rows:

| `evidence_status` of those 30 units | n |
|---|---|
| `none` — no gold span reached the generator | **22** |
| `complete` | 8 |

The evidence-gated variant recovers 22 of the 30 (its `wrong_retrieval` recall
rises 0.769 → 0.938) at the cost of 11 new false positives (precision
1.000 → 0.917): 8 units the reference calls `incorrect_answer` and 3 it calls
`ok` are now charged to retrieval.

Rules that fired on those 30 units: R9 (18), R11 (5), R10 (4), R6 (2), R8 (1).
Across all 52 units where the document-gated variant disagrees with the
reference: R9 (20), R11 (17), R6 (8), R10 (5), R8 (2).

> **Interpretation.** The document-level gate lets a generation rule fire on rows
> where the pipeline's own evidence field already records that nothing usable
> arrived. That is the misattribution the paper is about, and it is visible
> without any human judgement — but the reference set is what shows the
> correction is an improvement rather than a different arbitrary choice.

### 5.2 `partial_answer` over-prediction

Support 3, predicted 18 (document-gated) / 13 (evidence-gated); F1 0.095 / 0.125.
Of the 18 document-gated predictions only 1 is correct; the other 17 are units
the reference calls `incorrect_answer` (7), `wrong_retrieval` (5) or `ok` (5).

> **Interpretation.** Rule R11 fires on partial token overlap. The guidelines
> require that a `partial_answer` contain part of *what the reference states*;
> token overlap with a retrieved sentence is a weak proxy for that. This is a
> threshold/rule problem, not a gating problem — the evidence gate barely helps.

### 5.3 `hallucination` predicted against zero support

The taxonomy assigns `hallucination` to 8 units (6 under evidence gating); the
reference assigns it to none. Of the 8, the reference calls 3
`incorrect_answer`, 3 `ok`, 2 `wrong_retrieval`.

> **Interpretation.** Expected given the setup: the annotated run uses a
> deterministic *extractive* control that copies sentences from retrieved
> context, so by construction it cannot invent content absent from the context —
> which is the guidelines' definition of hallucination. The category cannot be
> validated on this run; it needs a run with a real generative model.

### 5.4 `ok` recall

`ok` recall is 0.375 (document-gated) / 0.3125 (evidence-gated) on a support of
16. Ten of the 16 are labelled as some failure by the taxonomy — 5 as
`partial_answer`, 3 as `hallucination`, 2 as `incorrect_answer` (§1.4).

> **Interpretation.** The reference credits an answer that conveys any one of
> QASPER's accepted alternative answers; the taxonomy's `answer_f1_ok = 0.6`
> threshold is stricter than that. The disagreement is a definition mismatch,
> and it is the clearest candidate for threshold re-tuning — which has not been
> done (see `limitations.md`).

### 5.5 Annotation confidence

The reference set carries high confidence on 155 units, medium on 39, low on 6
(`unit_0016`, `unit_0027`, `unit_0041`, `unit_0081`, `unit_0122`, `unit_0189`).
The low-confidence units are cases where the reference answers are internally
contradictory or marked "content missing" in QASPER itself.

---

## 6. Retrieval decomposition (prior experiments, unchanged)

Source: `docs/EXPERIMENTS.md`, `results/`.

| | QASPER dev | NQ validation | HotpotQA | 2WikiMultihopQA |
|---|---|---|---|---|
| n | 290 | 300 | 150 | 150 |
| median chunks per gold document | 19 | 31 | 2 | — |
| A — document, any | 0.441 | 0.997 | 0.993 | — |
| B — document, quantified | 0.441 | 0.997 | 0.507 | — |
| C — span, quantified | 0.276 | 0.730 | 0.507 | — |
| quantifier A→B | 0.0 pp | 0.0 pp | 48.7 pp (p=2.1e-22) | 64.7 pp (p=1.3e-29) |
| granularity B→C | 16.6 pp (p=7.1e-15) | 26.7 pp (p=1.7e-24) | 0.0 pp | 1.3 pp (p=0.5, n.s.) |

Attribution shift, failures charged to retrieval:

| Corpus | n | document-level | evidence-level |
|---|---|---|---|
| QASPER | 290 | 162 | 210 |
| Natural Questions | 300 | 1 | 81 |
| HotpotQA | 150 | 1 | 74 |
| 2WikiMultihopQA | 150 | 5 | 104 |

Chunk-size dependence (QASPER, same questions): 128 tokens → 43 chunks/doc →
18.6 pp; 256 → 19 → 16.6 pp; 512 → 9 → 11.0 pp; HotpotQA paragraphs → 2 →
0.0 pp.

## 7. Generation replay (prior experiment, unchanged)

Source: `docs/EXPERIMENTS.md`. Retrieval reused verbatim; only the generator
changes. n = 150 per model.

Qwen2.5-0.5B-Instruct, by evidence status:

| Evidence that reached the generator | n | correct | abstained | answered |
|---|---|---|---|---|
| Complete | 44 | 18.2% | 9.1% | 90.9% |
| Document retrieved, span missing | 26 | 0.0% | 0.0% | 100% |
| Nothing from any gold document | 75 | 1.3% | 4.0% | 96.0% |

| Generator | P(correct \| complete) | P(correct \| incomplete) | difference | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.182 | 0.010 | 17.2 pp | 0.0004 |
| SmolLM2-360M | 0.136 | 0.030 | 10.7 pp | 0.023 |

> **Interpretation.** The 26 middle-row questions are exactly the rows a
> document-level metric scores as retrieval *success*. The model answered all 26,
> abstained on none, and was correct on none. No hallucination rate, faithfulness
> benchmark or model ranking is claimed from this.
