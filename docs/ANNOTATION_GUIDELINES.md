# Annotation guidelines — validating the failure taxonomy

You are labelling what went wrong (or right) in a retrieval-augmented question
answering system. Your labels are the reference against which the system's own
automatic classifier is measured.

**Read this before starting.** The categories below are defined in terms of what
you can see on the page. They are deliberately *not* phrased as the rules the
system uses. If you reproduce the system's rules, the validation measures
nothing: we would only learn that the rules agree with themselves.

---

## What you are given

Each unit contains:

| Field | Meaning |
|---|---|
| `question` | The question that was asked. |
| `reference_answers` | The dataset's gold answer(s). Treat as correct. |
| `corpus_can_answer` | `true` if the corpus is supposed to contain the answer. |
| `gold_evidence` | The passage(s) the dataset says support the answer. |
| `retrieved_context` | What the system actually retrieved, in rank order. |
| `system_answer` | What the system produced. |

You will **not** see the system's proposed label, its metric values, or its
confidence. That is deliberate.

Fill in three fields:

- `human_label` — exactly one category name from the list below.
- `human_confidence` — `high`, `medium`, or `low`.
- `human_notes` — free text. Please use it whenever you hesitate.

---

## The decision procedure

Work through these in order. The first one that applies is your label.

### Step 1 — Should this question have been answerable?

Look at `corpus_can_answer`.

**If `false`** (the corpus is not supposed to contain the answer):

| Label | When |
|---|---|
| `ok_abstained` | The system declined to answer, or said the context does not contain the answer. **This is a success, not a failure.** |
| `answered_when_unanswerable` | The system produced a substantive answer anyway. |

Stop here. Nothing below applies.

**If `true`**, continue to step 2.

### Step 2 — Did the evidence reach the system?

Compare `gold_evidence` against `retrieved_context`. Read the retrieved text;
do not rely on document identifiers matching.

| Label | When |
|---|---|
| `no_retrieval` | `retrieved_context` is empty. |
| `wrong_retrieval` | Nothing retrieved contains the information needed to answer. Passages from the right document that omit the relevant fact still count as wrong retrieval. |

If the information needed to answer is genuinely absent from everything
retrieved, label it as a retrieval failure **even if the system's answer
happens to be correct**. A correct answer produced without supporting evidence
in context is not a success of this system; the model supplied it from
elsewhere. Note that case in `human_notes` — those units are interesting.

For multi-hop questions (several entries in `gold_evidence`, and the question
requires combining them): if any required piece is missing, that is
`wrong_retrieval`. Retrieving one of two necessary passages is not partial
success at retrieval; the question cannot be answered from what arrived.

If the necessary information **is** present in `retrieved_context`, continue.

### Step 3 — Was the answer right?

The evidence was there. Judge the answer against `reference_answers`.

| Label | When |
|---|---|
| `ok` | The answer conveys the reference answer. Wording may differ; extra correct detail is fine. |
| `partial_answer` | Part of the reference answer is present and correct, but something the reference states is missing. |
| `incorrect_answer` | The answer asserts something different from, or contradictory to, the reference. |
| `refusal_when_answerable` | The system declined although the evidence was present. |
| `hallucination` | The answer contains specific claims — names, numbers, dates, quotations — that appear in neither the retrieved context nor the reference, and are presented as fact. |

---

## The distinctions people get wrong

**`hallucination` vs `incorrect_answer`.** These are not the same and the
difference matters. Use `incorrect_answer` when the answer is simply wrong.
Reserve `hallucination` for an answer that *invents* content not present in the
context — a fabricated citation, an invented statistic, a person who does not
appear anywhere. Wrongness alone is not hallucination. When in doubt, use
`incorrect_answer` and say so in your notes.

**`partial_answer` vs `incorrect_answer`.** Ask whether the answer is *on the
way* to the reference. "30 days" against a reference of "30 days from the date
of purchase, refunded within 5–7 business days" is partial. "90 days" is
incorrect. If it contains nothing the reference contains, it is incorrect.

**`partial_answer` vs `ok`.** A verbose answer that contains everything in the
reference is `ok`. Length is not a defect. Missing a fact the reference states
is `partial_answer`, even if what is present is correct.

**`wrong_retrieval` vs a generation label.** This is the most consequential
distinction in the study, so spend time on it. The question is only: *was the
information needed to answer present in the retrieved text?* If it was not, the
label is `wrong_retrieval` regardless of how good or bad the answer is. Do not
charge the generator for something it was never shown.

**`ok_abstained` vs `refusal_when_answerable`.** The same system behaviour —
declining — is a success in one case and a failure in the other. The difference
is entirely whether the answer was available. Check `corpus_can_answer` and the
retrieved context before choosing.

**Empty answers.** If `system_answer` is blank, label `refusal_when_answerable`
when the evidence was present, `wrong_retrieval` when it was not, and note it.

---

## Difficult and ambiguous cases

Roughly a quarter of these units were selected *because* they sit near a
decision boundary. Finding them hard is expected and is the point — those are
the units where the system's tuned thresholds are doing the work.

- **The reference answer looks wrong to you.** Label against the reference
  anyway, mark `human_confidence: low`, and say so in your notes. Dataset noise
  is a real finding, and we want it separated from system error.
- **The question is ambiguous.** Judge the most natural reading. Note the
  ambiguity.
- **The answer is right but for a different question.** `incorrect_answer`.
- **The evidence is partially present** — some of the supporting passage was
  retrieved, but it is cut off mid-fact. If what arrived is sufficient to answer,
  continue to step 3; if not, `wrong_retrieval`. Note it either way.
- **You genuinely cannot decide.** Use `human_confidence: low`, pick the closer
  category, and explain in notes. Do not leave `human_label` blank — an empty
  label is dropped from the analysis, which silently removes exactly the hard
  cases we most want counted.

---

## Working practice

- **Work independently.** Do not discuss units with the other annotator while
  labelling. The whole value of two annotators is that the labels are
  independent; agreement produced by conversation measures nothing.
- **Do not look at `proposed_labels_key.jsonl`.** It contains the system's
  answers and is in the package only for the scoring step afterwards.
- Your sheet is at `annotator_<your id>/annotation_sheet.jsonl`. Save your
  completed file next to it as `completed.jsonl`, keeping `annotation_id`
  unchanged — that field is what aligns the two sets.
- Expect roughly 1–2 minutes per unit. Take breaks; fatigue shows up as drift
  toward whichever label is easiest.

---

## What happens to your labels

1. **Agreement between the two of you** is measured with Cohen's kappa, plus a
   per-category breakdown and a confusion matrix. This measures whether the task
   is well defined. If it is low, the categories are at fault, not you.
2. **Adjudication.** Where you both chose the same label, it stands. Where you
   differ, the unit is *unresolved* unless a third person adjudicates. A
   disagreement is never broken by using the system's label.
3. **The system is scored** against the adjudicated labels — per-category
   precision, recall and F1, not just accuracy, because the rare categories are
   the ones the taxonomy exists to separate.

Run with:

```bash
python scripts/score_annotations.py \
    --package reports/annotation/<package> \
    --annotator a=<...>/annotator_a/completed.jsonl \
    --annotator b=<...>/annotator_b/completed.jsonl
```

The scorer refuses to run on empty sheets rather than emitting a
plausible-looking table from no data.
