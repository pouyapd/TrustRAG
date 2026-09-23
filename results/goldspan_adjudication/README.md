# Gold-span adjudication — labels and estimates

Everything needed to recompute the gold-span completeness figure of the paper (Table 9),
from this repository alone. These files carry **no QASPER text**: they are unit
identifiers, the author's labels, the proxy partition and the computed estimates. The
adjudication sheets themselves quote QASPER passages and are therefore not tracked here —
see [Where the evidence text lives](#where-the-evidence-text-lives).

## The question that was adjudicated

> Is the reference answer derivable from the retrieved text alone, without relying on the
> annotated gold span?

Answers are `YES`, `NO` or `CANNOT_TELL`. A `YES` means the span rule reported a retrieval
failure on a unit whose answer was in fact derivable from what the system retrieved —
under-coverage of the gold-span annotation.

## Population

The 133 answerable units of the 200-unit annotation set whose retrieved context contained
no gold span. Two automated proxies (content-word overlap and MiniLM cosine) partitioned
them; `gold_span_semantic.json` holds that partition.

| Stratum | Units | Adjudicated | YES | NO |
|---|---:|---:|---:|---:|
| Both proxies say "answer absent" | 36 | — | — | — |
| Both proxies say "answer present" | 10 | 10 | 3 | 7 |
| Unresolved, possibly inferable | 8 | 8 | 1 | 7 |
| Unresolved, ambiguous | 58 | 58 | 1 | 57 |
| Unresolved, lexical signal only | 21 | 21 | 2 | 19 |
| **Total** | **133** | **97** | **7** | **90** |

Under-coverage is therefore **7 of 133 = 0.053**, a direct count with no sampling error,
or about **5–7%** once the 36 units nobody read are allowed for. The 36 are the units both
proxies agreed were unsupported, and they are counted as containing no under-coverage; if
they behaved like the 97 that were adjudicated the rate would be 0.072.

## Two rounds, one adjudicator

| | Round 1 | Round 2 |
|---|---|---|
| Units | 60, a proportional stratified sample of the 87 unresolved (seed 20260907) | the remaining 37: 27 unresolved units the sample did not draw, plus the 10 the proxies called "answer present" |
| Answers | 4 YES, 56 NO, 0 CANNOT_TELL | 3 YES, 34 NO, 0 CANNOT_TELL |
| Files | `first_pass/` | `remaining37/` |

Both rounds were judged by the same person, the author. The second round **extends** the
first; it is **not** an independent second annotation, it yields **no** inter-annotator
agreement, and it does not remove the single-annotator limitation. Round 1 was a sample, so
its estimate carried a confidence interval; with round 2 the unresolved stratum is complete
and the finite population correction is zero, so the estimator degenerates to a count.

## Files

| Path | Contents |
|---|---|
| `gold_span_semantic.json` | the proxy partition of the 133 units, with each unit's stratum and proxy scores |
| `first_pass/answers.csv` | the 60 round-1 labels |
| `first_pass/manifest.json` | round-1 sampling design, seed, stratum sizes, blinding |
| `first_pass/estimate.json` | the round-1 stratified estimate, kept as the historical record |
| `remaining37/answers.csv` | the 37 round-2 labels |
| `remaining37/manifest.json` | round-2 selection (a census, no seed), blinding, relation to round 1 |
| `remaining37/estimate_combined.json` | the combined count behind Table 9 |

## Recompute

```bash
python scripts/score_goldspan_adjudication.py \
    --package results/goldspan_adjudication/first_pass \
    --remainder results/goldspan_adjudication/remaining37 \
    --audit results/goldspan_adjudication/gold_span_semantic.json \
    --out /tmp/estimate_combined.json
```

This reproduces `remaining37/estimate_combined.json` exactly. Dropping `--remainder`
reproduces `first_pass/estimate.json` exactly, which is how the round-1 figure was
obtained. `tests/test_goldspan_remainder.py` checks both.

## Where the evidence text lives

To re-read the passages a unit was judged on — rather than only recompute the number —
you need the adjudication sheets, which quote QASPER and so are not redistributed here.
Two routes:

- The annotation package deposited on Zenodo (DOI
  [10.5281/zenodo.22879569](https://doi.org/10.5281/zenodo.22879569), CC BY 4.0) contains
  the 200-unit package with full retrieved context and the round-1 sheet.
- The round-2 sheet is regenerated deterministically from that package plus this
  repository, after fetching QASPER as `docs/DATASETS.md` describes:

  ```bash
  python scripts/build_goldspan_remainder.py \
      --package reports/annotation/qasper_dev_300_full_context \
      --audit  reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json \
      --first-pass reports/annotation/goldspan_adjudication \
      --out    reports/annotation/goldspan_adjudication_remaining37

  python scripts/render_goldspan_review.py \
      --package reports/annotation/goldspan_adjudication_remaining37 \
      --corpus  data/raw/qasper-dev-v0.3.json \
      --out     reports/annotation/goldspan_adjudication_remaining37/HUMAN_REVIEW.md
  ```

  Both scripts verify that every stored character range still indexes the text the package
  recorded, and refuse to write if it does not.
