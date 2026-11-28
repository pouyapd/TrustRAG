# Blinded repeat adjudication by the original adjudicator (intra-annotator consistency)

**Status: prepared, not performed.** Only the author can do this; no repeat labels exist.

This is not a substitute for an independent second annotator (see
GUIDELINES_second_annotator.md). It measures whether the same person gives the same
judgement twice, which bounds the noise in the existing labels; it says nothing about
whether a different reader would agree.

## Design

- All 133 units, including the 36 never read, in a new random order (`author_repeat_order.json`,
  seed 20261125), different from the original sheets and from the second-annotator order.
- At least two weeks after the author last looked at the adjudication files, and without
  opening `results/goldspan_adjudication/*/answers.csv`, the proxy partition, the NLI audit
  (`results/audit/adjudication_audit.json`) or the manuscript's numbers while labelling.
- Same question, same answers (YES / NO / CANNOT_TELL), same decision rules as
  GUIDELINES_second_annotator.md, same display (question, reference answers, five retrieved
  chunks in full, gold evidence shown separately).
- Fill `author_repeat_answers_TEMPLATE.csv` (save as `author_repeat_answers.csv`).

## Scoring

```bash
python score_agreement.py \
    --a ../../../results/goldspan_adjudication/first_pass/answers.csv \
    --a ../../../results/goldspan_adjudication/remaining37/answers.csv \
    --b author_repeat_answers.csv
```

Report raw agreement, Cohen's kappa, Gwet's AC1 and PABAK on the 97 units judged twice, every
disagreement, and the repeat rate on the 36 previously unread units. In the paper, call this
*intra-annotator* (test–retest) consistency, never inter-annotator agreement.
