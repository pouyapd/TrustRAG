# TrustRAG Failure Taxonomy

Two taxonomies ship in this repository. **v1** (`src/evaluation/failure_modes.py`)
is frozen so previously published numbers stay reproducible. **v2**
(`src/evaluation/taxonomy.py`) is what new analysis should use. Both are
computed for every evaluated row, so any run can be read either way.

This document is also the draft annotation guideline: the categories below are
written so a human annotator can apply them to a row without reading the code.

---

## Why v2 exists

An audit of v1 against the bundled 20-question evaluation found three defects.

### 1. Abstention failures were invisible

A question the corpus cannot answer has an empty `relevant_doc_ids`. In v1 such
a row could not reach `hallucination` (that needs a low faithfulness score) and
could not reach `refusal_when_answerable` (that needs a non-empty relevant set).
A system that confidently answered an unanswerable question was therefore
labelled `partial_answer`.

Both unanswerable questions in the bundled dataset are answered confidently and
wrongly — "What is the capital of France?" is answered with the Starter plan's
storage limits — and v1 reported neither as a safety-relevant failure.

### 2. `partial_answer` absorbed wrong answers

v1 routed everything with low token overlap into one bucket. On the bundled
dataset that bucket mixed genuinely incomplete answers with flatly incorrect
ones: "How much does the Pro plan cost per month?" was answered with the API
rate limit and labelled `partial_answer`. Any conclusion of the form "retrieval
is fine, generation completeness is the bottleneck" is unsupported when the
category conflates *incomplete* with *wrong*.

### 3. Thresholds were undefended constants

`0.30` and `0.60` appeared as literals with no derivation and no sensitivity
analysis, and re-running the classifier with different values required
re-running every LLM call. In v2 thresholds live in a hashable `TaxonomyConfig`
and classification is a pure function of stored features, so
`scripts/reclassify.py` re-scores a finished run with no model calls at all.

---

## Categories

### Non-failures

| Mode | Meaning |
|---|---|
| `ok` | The answer is supported by the retrieved context and matches the reference. |
| `ok_abstained` | The question is unanswerable from the corpus and the system correctly declined. |

`ok_abstained` is a success. Counting a correct refusal as a failure would
penalise exactly the behaviour a trustworthy system should have.

### Retrieval-attributable failures

| Mode | Meaning |
|---|---|
| `no_retrieval` | The retriever returned nothing at all. |
| `wrong_retrieval` | The question is answerable, but no retrieved chunk came from a relevant document. |

### Generation-attributable failures

| Mode | Meaning |
|---|---|
| `hallucination` | The answer is not grounded in the context it was given (faithfulness below threshold). |
| `incorrect_answer` | Relevant context was retrieved, and the answer asserts something other than the reference. |
| `partial_answer` | The answer is on topic and consistent with the reference but omits some of it. |
| `refusal_when_answerable` | Relevant context was retrieved and the system refused anyway. |
| `answered_when_unanswerable` | The corpus cannot answer the question and the system answered regardless. |

---

## Decision rules

Rules are evaluated in order; the first match wins. The id of the rule that
fired is recorded on every row as `failure_rule_v2`, so a label can always be
traced back to the reason it was assigned.

| Rule | Condition | Result |
|---|---|---|
| R1 | nothing retrieved | `no_retrieval` |
| R2 | unanswerable **and** abstained | `ok_abstained` |
| R3 | unanswerable **and** answered | `answered_when_unanswerable` |
| R4 | answerable **and** no relevant document retrieved | `wrong_retrieval` |
| R5 | relevant document retrieved **and** abstained | `refusal_when_answerable` |
| R6 | faithfulness < `faithfulness_threshold` | `hallucination` |
| R8 | key-fact recall ≥ `key_fact_recall_ok` | `ok` |
| R9 | key-fact recall ≤ `key_fact_recall_incorrect` | `incorrect_answer` |
| R7 | reference has no extractable facts **and** F1 ≥ `answer_f1_ok` | `ok` |
| R10 | reference has no extractable facts **and** F1 ≤ `fallback_f1_incorrect` | `incorrect_answer` |
| R11 | otherwise | `partial_answer` |

### Three ordering decisions worth arguing about

**R4 before R5.** When retrieval returned no relevant document, refusing is the
*correct* response. Attributing that row to generation would blame the model for
behaving well, so the causal fault is assigned to retrieval.

**R3 before everything about answer quality.** On an unanswerable question there
is no correct content to match, so answer-similarity rules cannot apply. The
only question that matters is whether the system abstained.

**R8/R9 before R7.** Key facts outrank overall F1 whenever the reference has
any. A fluent answer that drops one of the reference's facts can still score a
high F1 on the words it does share — during development, an answer that omitted
"receive a reset link valid for 30 minutes" scored F1 ≈ 0.66 and would have
passed as `ok` under an F1-first ordering. F1 is used only as the fallback for
references with nothing extractable.

---

## Key-fact recall

The signal that separates `incorrect_answer` from `partial_answer`.

The reference answer's **key facts** are its normalized tokens that are not
function words and are either numeric or at least three characters long. Tokens
are S-stemmed (a trailing `s` is dropped from words longer than three
characters that do not end in `ss`) so a singular/plural difference is not
counted as a missing fact.

Key-fact recall is the fraction of those facts that appear in the prediction.
The intuition: an answer that reproduces most of the reference's facts is
incomplete; one that reproduces almost none while still asserting something is
wrong.

Worked examples from the bundled dataset:

| Question | Reference | Prediction | Recall | Label |
|---|---|---|---|---|
| Refund window for annual subscribers? | 30 days from the date of purchase. | Annual subscribers have a 30-day refund window from the date of purchase. | 1.00 | `ok` |
| How do I reset my password? | Click 'Forgot password' on the login screen to receive a reset link valid for 30 minutes. | If you forget your password, click "Forgot password" on the login screen. | 0.45 | `partial_answer` |
| How much does the Pro plan cost per month? | 29 EUR per month. | The Pro plan allows 1000 API requests per minute. | 0.00 | `incorrect_answer` |

**Correction to an earlier claim.** An audit of the dataset layer asserted that
excluding the QASPER abstract from the document body was silently discarding
abstract-grounded questions. Measured on the real dev split, **zero** evidence
strings match the abstract exactly, so that loss does not occur. Including the
abstract is still correct — it is part of the paper a reader saw — but it is a
robustness improvement, not the bug it was claimed to be. The real unresolvable
-evidence problem is figure and table captions; see
[DATASETS.md](DATASETS.md).

Normalization detail: punctuation is replaced with a space rather than deleted.
Deleting it merges hyphenated compounds — `30-day` becomes the single token
`30day`, matching neither `30` nor `day` — which produced a false
`partial_answer` on a verbatim-correct response during development.

---

## Stage attribution

There are now **two** attribution mechanisms, and they answer different
questions. Both are reported.

### 1. Mode-based mapping (`taxonomy.STAGE_ATTRIBUTION`)

Maps each failure mode to `retrieval`, `generation` or `none`. This is a
**declared mapping, not a causal measurement**: it says where a failure of that
kind belongs by definition, not that fixing that stage would have prevented
this row. The report says so in its own `attribution.note` field.

### 1b. The R4 gate, and an inconsistency it caused

Rule R4 (`wrong_retrieval`) fires on the `retrieval_hit` feature. In
`failure_mode_v2` that feature means **a relevant document was retrieved**.
A row whose document arrived but whose gold span never did therefore *passes*
R4 and falls through to the answer-quality rules, where it is labelled
`incorrect_answer` — a generation failure.

That is the exact error the evidence layer exists to prevent, and it was
present in the taxonomy labels while `attribution_stage` reported the opposite
for the same rows. Measured:

| | rows relabelled generation → retrieval | share |
|---|---|---|
| QASPER n=300 | 46 | 15% |
| NQ n=300 | 66 | 22% |
| HotpotQA n=150 | 62 | 41% |

`failure_mode_evidence` re-runs the identical rules with `retrieval_hit`
meaning **the gold span reached the generator**. The effect on the
distribution is large:

| | `failure_mode_v2` | `failure_mode_evidence` |
|---|---|---|
| QASPER `wrong_retrieval` | 162 | **210** |
| QASPER `incorrect_answer` | 88 | **51** |
| NQ `wrong_retrieval` | **1** | **81** |
| NQ `incorrect_answer` | 174 | **120** |
| HotpotQA `wrong_retrieval` | **1** | **74** |
| HotpotQA `incorrect_answer` | 102 | **45** |

Both are emitted on every row. `failure_mode_v2` is frozen so previously
published v2 distributions stay reproducible; `failure_mode_evidence` is the
one that agrees with `attribution_stage` and the one new analysis should use.

### 2. Evidence-based attribution (`evidence.attribute_stage`)

Introduced with W2 and used whenever gold evidence spans are available. Instead
of reasoning from the label, it reasons from **what the generator was actually
given**, established by exact character-offset overlap between gold spans and
retrieved chunks:

1. Unanswerable question → judged only on abstention.
2. Nothing retrieved → `retrieval`.
3. Required evidence never retrieved → `retrieval`, *regardless of the answer*.
   A correct answer here is not credited as success: without the evidence in
   context it indicates parametric knowledge, not retrieval.
4. Evidence present, answer wrong → `generation`.
5. Evidence present, answer correct → no failure.

This is stronger than the mapping because it can distinguish a generation
failure from a retrieval failure that merely *looks* like one. Measured on real
corpora, the two disagree substantially: on QASPER dev the mode-based view
attributed most failures to generation, while evidence-based attribution
charged the majority to retrieval, because the supporting passage had never
reached the generator. See [EXPERIMENTS.md](EXPERIMENTS.md).

Multi-hop is where the difference bites hardest: under `all_required`,
retrieving one of two required documents is `partial` — a retrieval failure —
whereas the v2 taxonomy's `retrieval_hit` feature counts any single overlap as
a hit and would charge the row to generation.

---

## Limitations

These are the honest caveats. They are the reason the current output is a
diagnostic instrument and not yet a validated one.

1. **The thresholds are tuned, not validated.** Every default in
   `TaxonomyConfig` was chosen by inspecting the bundled 20-question fixture.
   `key_fact_recall_incorrect = 0.20` in particular was raised from `0.0` after
   a single incidental token match ("plan") flipped clearly incorrect answers
   into `partial_answer`. Tuning a classifier on the same data it is evaluated
   on is exactly the practice this project criticises elsewhere, and it is
   recorded here rather than hidden.

2. **No human validation.** Nothing establishes that these labels agree with
   what a person would say. Until an annotation study with at least two
   annotators and a reported agreement statistic exists, the taxonomy is a
   proposal. This is the single largest gap.

   The full validation protocol now exists — package, guidelines and scoring —
   and is described below, and it has been run: 200 units of the full-context
   package are labelled and the taxonomy is scored against them
   (`docs/paper/results.md`). **Those 200 labels were produced by a
   language-model annotator following the guidelines, not by a person**; 22
   units in the earlier package carry human labels. So the gap is narrower than
   it was but not closed: what exists is agreement with an independent reading,
   not with human judgement. The scorer still refuses to run on empty sheets
   rather than emitting a plausible-looking table from no data.

3. **Key-fact recall is lexical.** It cannot recognise a correct paraphrase
   that shares no vocabulary with the reference, and it cannot detect a negation
   flip ("SSO *is* available" vs "SSO is *not* available") when the surrounding
   facts match. Both are plausible sources of misclassification.

4. **Faithfulness comes from the generator itself.** The pipeline scores its own
   output, so `hallucination` inherits that circularity. With the bundled
   extractive mock the score is exactly 1.0 on every row, which means R6 can
   never fire in the offline configuration.

5. **Answerability is inferred from the labels.** A question is treated as
   unanswerable when its `relevant_doc_ids` is empty. That is a property of the
   dataset annotation, not of the corpus, and mislabelled items would propagate
   directly into the abstention metrics.

---

## Validating the taxonomy against humans

The protocol is implemented end to end. What is missing is people, not code.

### 1. Build the package

```bash
python scripts/build_annotation_package.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/annotation/qasper_dev_300 --n-units 200
```

Four properties of the sample matter:

- **Stratified by proposed mode, with a floor.** Every category the classifier
  can emit gets at least `--min-per-mode` units. Without a floor, a category
  occurring in 3% of rows barely appears, and its row of the confusion matrix is
  empty — per-category recall would be undefined for exactly the rare categories
  the taxonomy exists to separate.
- **A quarter of the budget is boundary cases.** Units whose deciding feature
  sits within 0.10 of the threshold that classified them. These are where a
  tuned constant, rather than an obvious fact, chose the label; a uniform sample
  is dominated by rows the rules get trivially right, which inflates agreement
  and teaches nothing.
- **Blinded.** The proposed label, the rule that fired, the metric values and
  the decision features are all withheld from the annotation sheet and written
  to `proposed_labels_key.jsonl` instead. Showing an annotator the system's
  answer invites anchoring.
- **Independently ordered per annotator.** Both annotators receive the same
  units in different shuffles, so two returned sheets cannot be aligned by
  position — only by `annotation_id`.

The sampling weight per category is recorded so population proportions can be
recovered. The sample is deliberately *not* representative.

### 2. Annotate

Two annotators, working independently, following
[ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md), each using the local
annotation tool:

```bash
python scripts/annotate.py --annotator a     # one per annotator
python scripts/annotate.py --annotator a --validate
```

The tool enforces the blinding rather than relying on the annotator to respect
it: it never opens `proposed_labels_key.jsonl`, and the page is built from an
allowlist of fields, so a proposed label cannot reach the screen even if the
package were regenerated with extra keys. The guidelines define
every category in terms of what is visible on the page — question, reference
answer, retrieved context, system answer — and deliberately **not** in the terms
the classifier uses. That distinction is the whole point: if an annotator
reproduces the rule, the study measures only that the rules agree with
themselves.

### 3. Score

```bash
python scripts/score_annotations.py     --package reports/annotation/qasper_dev_300     --annotator a=.../annotator_a/completed.jsonl     --annotator b=.../annotator_b/completed.jsonl
```

Three separate measurements, in this order:

1. **Annotator vs annotator** — Cohen's kappa, per-category kappa, and a
   confusion matrix. This measures whether the *task* is well defined. Kappa
   rather than raw agreement because the label distribution is heavily skewed:
   with 80% of rows in one category, two annotators who both guess that category
   agree 64% of the time knowing nothing. If kappa is low, the categories are at
   fault and nothing downstream is interpretable.
2. **Adjudication** — where both annotators agree, the label stands. Where they
   disagree, the unit is **unresolved** unless a human third pass resolves it. A
   disagreement is never broken using the system's own label, which would
   quietly make the taxonomy its own referee.
3. **Taxonomy vs adjudicated labels** — per-category precision, recall and F1,
   not just accuracy. On a skewed label set, accuracy is dominated by the
   majority class and says nothing about the distinctions that matter.

### Status

| Step | State |
|---|---|
| Sampling, blinding, guidelines, scoring | Implemented and tested |
| Annotation units generated | 200 for `qasper_dev_300` (50 boundary cases) |
| Human labels collected | **0** |
| Cohen's kappa | Not computable — no labels |
| Taxonomy precision/recall against humans | Not computable — no labels |

This remains the largest open gap in the project. The headline retrieval and
evidence results do not depend on the taxonomy's thresholds, but every
generation-side failure label does.

---

## Re-scoring an existing run

```bash
# Score a finished run under different thresholds. No model is called.
python scripts/reclassify.py --records reports/inference.jsonl \
    --out reports/strict --answer-f1-ok 0.8

# Sensitivity of the failure distribution to the faithfulness threshold.
python scripts/reclassify.py --records reports/inference.jsonl \
    --out reports/sweep --sweep-faithfulness 0.3,0.5,0.6,0.8
```
