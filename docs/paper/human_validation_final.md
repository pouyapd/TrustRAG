# Human validation — final result

Status: **complete**. Two human passes, an audit between them, and a derived final
dataset. This file supersedes every earlier statement in the repository about the
human study. Numbers are recomputed from the artifacts named per section; none is
carried over from an earlier draft.

---

## 1. What exists, and what each thing is

| Artifact | Provenance | Units |
|---|---|---|
| `annotator_human/completed.jsonl` | **Human**, original pass | 200 |
| `review_43_flagged/annotator_review/completed.jsonl` | **Human**, second review of the audited-flagged units | 43 |
| `final_human_reviewed/completed.jsonl` | **Derived**: original label where not flagged, review decision where flagged | 200 |
| `annotator_a/completed.jsonl` | **Automated** — a language-model annotator, *not* ground truth | 200 |

Both human passes are preserved unmodified; the derived file is a new artifact and
`final_human_reviewed/provenance_chain.json` records, for every unit,
`original_label → flagged (reason) → second_review_label → final_label`.

## 2. What the review changed

| | n |
|---|---:|
| Units never flagged — original label kept | 157 |
| Units flagged and re-reviewed | 43 |
| — label changed on review | 36 |
| — original label upheld | 7 |
| Confidence values revised | 17 |

Original vs final human labels: raw agreement 0.8200, Cohen's κ 0.6884. The review
moved a substantial minority of the dataset, so the two passes must not be treated
as interchangeable.

Final label distribution (n = 200): `wrong_retrieval` 136, `ok` 32,
`partial_answer` 22, `answered_when_unanswerable` 9, `incorrect_answer` 1. Three
categories that appeared in the original pass — `no_retrieval` (11),
`ok_abstained` (2), `hallucination` (2) — are absent from the final dataset; the
review reassigned all of them.

## 3. Guideline consistency, before and after

Audited against the decision procedure in `docs/ANNOTATION_GUIDELINES.md`, with the
objective facts recomputed from the package (answerability; whether any retrieved
chunk overlaps a gold span, by character range).

| | Original pass (200) | Flagged units after review (43) |
|---|---:|---:|
| `strongly_supported` | 106 | 29 |
| `plausibly_supported` | 37 | 3 |
| `ambiguous` | 14 | 0 |
| `likely_inconsistent` | 43 | 11 |

The review resolved 32 of the 43. The residual 11 are all the same case: an
answer-quality label (`ok` ×7, `partial_answer` ×4) kept on a unit where no gold
span was retrieved, every one at high confidence, after a deliberate second look at
the full context. That is a considered position, not a slip — and §5 shows the
annotator was very likely right on most of them.

## 4. The central result, against human labels

Both taxonomy variants are computed from the same stored run and differ only in
which signal gates rule R4. Scored against the final human-reviewed labels:

| Variant | Accuracy | 95% CI | Macro F1 | Cohen's κ |
|---|---:|---:|---:|---:|
| Document-gated (`failure_mode_v2`) | 0.6000 | 0.531–0.665 | 0.4764 | 0.3752 |
| **Evidence-gated (`failure_mode_evidence`)** | **0.7000** | 0.633–0.759 | **0.4819** | **0.4371** |

Paired over the same 200 units: 118 both correct, **22 only evidence-gated**,
2 only document-gated, 58 neither. Exact McNemar on the 24 discordant pairs,
**p < 0.0001**.

> **Interpretation.** The directional claim survives human validation, and on a
> paired test it is stronger than it was against the automated reference (22 vs 2,
> where the automated comparison gave 22 vs 9). Evidence-gating agrees with human
> judgement better than document-gating on the same units.

> **What it does not license.** Absolute agreement with humans is much lower than
> with the automated reference (accuracy 0.700 vs 0.805; κ 0.437 vs 0.631). The
> earlier figure overstated how well the taxonomy tracks human judgement, because
> the LLM reference and the rule system share failure directions. Any published
> number should be the human one.

### Per class, evidence-gated against final human labels

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `wrong_retrieval` | 136 | 133 | 0.917 | 0.897 | 0.907 |
| `ok` | 32 | 6 | 0.500 | 0.094 | 0.158 |
| `partial_answer` | 22 | 13 | 0.385 | 0.227 | 0.286 |
| `answered_when_unanswerable` | 9 | 9 | 1.000 | 1.000 | 1.000 |
| `incorrect_answer` | 1 | 33 | 0.030 | 1.000 | 0.059 |
| `hallucination` | 0 | 6 | 0.000 | — | — |

**This is the honest headline: retrieval-side attribution is validated; generation-side
classification is not.** `wrong_retrieval` (F1 0.907) and
`answered_when_unanswerable` (F1 1.000) hold up. Everything else fails. The
taxonomy predicts `incorrect_answer` 33 times where humans used it once, and finds
only 3 of the 32 answers humans called `ok`. The thresholds (`answer_f1_ok = 0.60`,
`key_fact_recall_ok = 1.0`) are far stricter than human judgement of the same
answers, and were never tuned against any evaluation set.

## 5. A threat to validity the human review exposed

The 11 residual units above prompted a check that should have been run earlier: **do
the gold spans actually cover the evidence that answers the question?** For every
answerable unit with zero gold-span coverage, we measure what fraction of the
reference answer's content words appear anywhere in the retrieved context
(`scripts/audit_gold_span_coverage.py`).

| Reference-answer content words present in retrieved context | n | Share |
|---|---:|---:|
| ≥ 0.8 — essentially present | 31 | 23.3% |
| 0.5–0.8 — partially present | 35 | 26.3% |
| < 0.5 — not present | 67 | 50.4% |
| **Total answerable units with zero gold-span coverage** | **133** | |

On roughly a quarter of the units where the span rule declares a retrieval failure,
the reference answer is essentially present in the text the generator received.
QASPER marks the sentences an annotator considered supporting; it does not mark
every passage from which the answer could be derived.

**Consequence for the central claim.** Span-gated attribution over-charges
retrieval, and the human reviewer detected this by reading the text — which is why
those 11 units keep an answer-quality label. The evidence-gating result in §4 is
therefore a comparison between a coarse rule (document-level) and a rule that is
sharper but biased in a known direction. It is not a comparison against a clean
oracle.

> **Superseded (2026-09-07).** A stratified human adjudication of 60 of the 87
> unresolved units now estimates under-coverage at **0.119, 95% CI [0.096, 0.142]**,
> defensibly 4–12% under sensitivity analysis. See `paper.md` §8. The lexical figure
> below is retained as the record of what the proxy alone suggested.

Token presence is not entailment, so 23.3% is an **upper bound** on how often the
span rule is wrong, not a correction factor. A tighter estimate needs entailment
annotation over those 133 units, which is human work that has not been done.

## 6. What the human study establishes, and what it does not

**Supported.**
- Evidence-gated attribution agrees with human judgement significantly better than
  document-gated attribution on the same units (22 vs 2 paired, p < 0.0001).
- The retrieval-side categories are reliably applied: `wrong_retrieval` F1 0.907,
  `answered_when_unanswerable` F1 1.000.
- The document-level/span-level distinction is one humans act on when they read the
  retrieved text, not an artefact of the rule system.

**Not supported.**
- That the taxonomy as a whole matches human judgement — macro F1 is 0.48, and four
  of six observed classes are near-unusable.
- Any generation-side category. `hallucination`, `refusal_when_answerable` and
  `ok_abstained` have zero support in the final human labels; `incorrect_answer`
  has one unit.
- That span-level evidence is a sound gold standard: §5 shows it over-charges
  retrieval on up to a quarter of the affected units.

**Design limitations that remain.**
- **One annotator.** No inter-annotator agreement statistic exists, so annotator
  bias cannot be separated from the construct. The κ figures here are
  human-vs-system, never human-vs-human.
- **Non-independence.** The audit told the annotator which units to re-examine and
  why. The second pass is therefore not independent of the guidelines being tested,
  and the 36 changed labels moved toward what the guidelines prescribe. This is a
  legitimate reading of a review pass, but it is not blind re-annotation, and the
  §4 result must be read with it in mind.
- **22 units** in the original pass came from a pilot conducted on the truncated
  package; 7 of those were re-reviewed on full context, 15 were not flagged and so
  carry pilot-era labels.
- **One corpus, one configuration** — QASPER dev, k = 5, 256-token chunks, MiniLM.

## 7. Reproducing this section

```bash
# audit the original pass
python scripts/audit_human_annotations.py \
    --package reports/annotation/qasper_dev_300_full_context --annotator human \
    --reference reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl \
    --rows reports/experiments/qasper_dev_300/rows.jsonl \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --out reports/annotation/qasper_dev_300_full_context/audit

# build the focused review set, then audit the completed review
python scripts/build_review_subset.py --package reports/annotation/qasper_dev_300_full_context \
    --annotator human --audit .../audit/human_annotation_audit.json \
    --verdict likely_inconsistent --out reports/annotation/review_43_flagged

# derive the final dataset with its provenance chain
python scripts/build_final_human_dataset.py \
    --original reports/annotation/qasper_dev_300_full_context --original-annotator human \
    --review reports/annotation/review_43_flagged --review-annotator review \
    --audit .../audit/human_annotation_audit.json \
    --review-audit reports/annotation/review_43_flagged/audit/human_annotation_audit.json \
    --out reports/annotation/qasper_dev_300_full_context/final_human_reviewed

# the gold-span coverage threat check
python scripts/audit_gold_span_coverage.py \
    --package reports/annotation/qasper_dev_300_full_context \
    --out .../audit/gold_span_coverage.json
```
