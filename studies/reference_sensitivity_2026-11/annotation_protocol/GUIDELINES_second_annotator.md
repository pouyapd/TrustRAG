# Second, independent adjudication of QASPER gold-span completeness

**Status: prepared, not performed.** No second annotator has been recruited. Nothing in the
revised manuscript reports a second annotator's labels. The manuscript states this
explicitly; the agreement statistics below can only be computed once real labels exist.

## Purpose

The revised paper reports one adjudicator's (the author's) judgement that, on 7 of 133
QASPER units whose retrieved context contained no gold span, the reference answer was
nevertheless derivable from the retrieved text. A single adjudicator gives a count but no
reliability estimate. This protocol obtains an independent second judgement on **all 133
units**, including the 36 that no human has read, so that

1. inter-annotator agreement (raw agreement, Cohen's kappa, Gwet's AC1 and the
   prevalence-adjusted bias-adjusted kappa, PABAK) can be reported on the 97 units both
   annotators judged; and
2. the assumption that the 36 proxy-"absent" units contain no under-coverage can be
   tested instead of assumed.

## Who can annotate

- Someone with enough NLP background to read QASPER papers (graduate level or equivalent).
- Not the author, and not anyone who has seen the author's labels, the proxy partition or
  the manuscript's figures.
- Paid or credited according to the norms of the annotator's institution; recruitment,
  payment and consent are the author's responsibility, not this package's.

## What the annotator sees, per unit

- The question.
- The reference answer(s).
- The five retrieved chunks, **full text**, in retrieval order, with character ranges.
- The paper's gold evidence paragraph(s), shown separately and labelled as the dataset's
  annotation.

The annotator does **not** see: the author's answer, the proxy bucket, the system's
answer, the failure-taxonomy label, or any summary statistic.

## The single question

> Using only the five retrieved chunks, could a careful reader derive the reference answer
> (all of its key facts), without relying on the gold evidence paragraph?

| Answer | Use when |
|---|---|
| `YES` | Every key fact of the reference answer is stated in, or follows directly from, the retrieved text. Paraphrase is fine; background knowledge beyond the text is not. |
| `NO` | At least one key fact of the reference answer is missing from the retrieved text, or would require the gold paragraph or outside knowledge. |
| `CANNOT_TELL` | The reference answer itself is ambiguous or unintelligible, so derivability cannot be judged. Use sparingly and explain in `note`. |

Also record `confidence` (`high`, `medium`, `low`) and a short `note` for every `YES` and
every `CANNOT_TELL`: which chunk carries the answer and why.

Decision rules (identical to the rules the first adjudicator followed):

1. For yes/no questions, `YES` requires that the retrieved text states the fact that
   decides the answer, not merely that the topic is mentioned.
2. For list answers, every list item must be derivable.
3. Numbers must match exactly (rounding as printed in the paper is acceptable).
4. Do not search the paper or the web.
5. Judge each unit independently; do not revisit earlier units after seeing later ones.

## Procedure

1. Build the blind sheet for all 133 units (commands below). Units appear in the fixed
   random order of `unit_order.json` (seed 20261124), not in the original order.
2. Give the annotator this file, the sheet and `annotator_B_answers_TEMPLATE.csv`.
3. Ask for a 10-unit calibration on the QASPER *training* split (not one of the 133);
   discuss the rules, never the 133.
4. The annotator fills the template. Expected effort: 4–6 hours.
5. Score with `score_agreement.py` (below). Report every statistic it prints, including
   the disagreement list, whatever the result.

## Building the sheet

The sheet quotes QASPER text and is therefore not stored in the repository. It is built from
the annotation package deposited at DOI 10.5281/zenodo.22879569 plus QASPER dev v0.3, with
the repository's existing scripts (`scripts/build_goldspan_adjudication.py`,
`scripts/build_goldspan_remainder.py`, `scripts/render_goldspan_review.py`), restricted to
the unit identifiers in `unit_order.json` and with the answer column removed. Those scripts
verify every stored character range against the text before writing.

## Scoring

```bash
python score_agreement.py \
    --a results/goldspan_adjudication/first_pass/answers.csv \
    --a results/goldspan_adjudication/remaining37/answers.csv \
    --b annotator_B_answers.csv
```

It prints raw agreement, Cohen's kappa, PABAK and Gwet's AC1 on the units both annotators
judged; the 2x2 table; under-coverage over 133 units under A, under B, under
"either says YES" and under "both say YES"; and B's rate on the 36 units A never read.
Because `YES` is rare (about 5%), kappa can be low even with high raw agreement; that is
why AC1 and PABAK are reported alongside it.
