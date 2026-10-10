# NQ localisation under alternative evidence references

**Exploratory. Not part of the submitted manuscript.** Run on 10 October 2026, after the
manuscript was submitted, to check one design choice. Nothing here changes a file in
`results/localisation/`; the manuscript's NQ numbers are reproduced exactly below, then
re-scored.

## The question

The within-document localisation score counts a localiser as correct when its top-ranked
chunk overlaps the gold evidence. For QASPER the gold evidence is the union of every
annotator's evidence paragraphs (`src/data/loaders/qasper.py`, evidence mode
`any_sufficient`). For Natural Questions it is the long answer of **the first annotator
who marked one** (`_first_long_answer` in `src/data/loaders/nq_parquet.py`), although the
NQ validation split is five-way annotated. How far do the NQ results depend on that?

## How NQ stores several annotations

In the parquet file (`nq-validation-0.parquet`, SHA-256 `d38ab58b…`), `annotations` is a
struct of parallel lists with one element per annotator: `id`, `long_answer`
(`candidate_index`, `start_token`, `end_token`, `start_byte`, `end_byte`; −1 when the
annotator marked no long answer), `short_answers` and `yes_no_answer`. Long-answer
candidates can be nested (`long_answer_candidates.top_level`), so two annotators can pick
the same passage at different granularity.

The loader keeps the first non-null long answer as the only supporting span. If that one
is pure markup, the question is skipped rather than falling back to another annotator.
Reference *answers* do use all annotators.

NQ's own evaluation (`nq_eval.py`, `score_long_answer`; `eval_utils.py`) treats a question
as having a long answer only if at least two annotators marked one
(`long_non_null_threshold = 2`), and counts a prediction as correct if it matches any one
of the non-null annotations.

## What the 300 study questions look like

| | questions |
|---|---:|
| Annotators who marked a long answer: 1 / 2 / 3 / 4 / 5 | 32 / 40 / 38 / 68 / 122 |
| More than one distinct long-answer span | 104 |
| Annotators' spans cover different chunks (the only questions any scheme can change) | 98 |
| Below NQ's official threshold (one annotator only) | 32 |

Compared with the first annotator's span, the other annotators' spans are identical in
607 cases, **disjoint in 185**, and nested in 16 (10 inside it, 6 containing it). By
question: 99 have at least one annotator who marked a passage that does not overlap the
first annotator's, 5 differ only by nesting. So the disagreement is mostly about *which
passage*, not about granularity.

## Reference schemes

Defined and tested in `src/evaluation/references.py`:

| Scheme | Gold evidence | Questions |
|---|---|---:|
| `first` | the first annotator's long answer (the manuscript) | 300 |
| `union` | any annotator's long answer (NQ's matching rule; QASPER's construction) | 300 |
| `official` | NQ's rule in full: questions with ≥ 2 annotators, any of their spans | 268 |
| `plurality` | the span most annotators chose (ties: all tied spans) | 300 |
| `agreement_subset` | questions whose annotators all cover the same chunks | 202 |
| `single` | one annotator drawn at random per question; 2,000 draws, seed 20261010 | 300 |

## Checks before any re-scoring

- Same 300 questions in the same order, same 12,358 chunks, same document sizes and
  first-annotator gold chunk counts as `results/localisation/within_document_nq.json`.
- Recomputed first-annotator ranks equal the stored ranks for every question: BM25 and
  the four bi-encoders on all 300; both cross-encoders on the 98 questions whose
  annotators cover different chunks. On the other 202 every scheme gives the same gold
  chunks, so the stored rank is exact there and the cross-encoders were not re-run.
- Under `first`, hit@1, hit@5, MRR, every McNemar and Holm p-value and the union figure
  (0.780) are identical to `results/localisation/summary_nq.json`.

## Results

hit@1 (hit@5); 95% Wilson intervals, MRR, rank distributions and lifts are in
`summary.json`.

| Localiser | first | union | official | plurality | agreement subset |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.353 (0.690) | 0.393 (0.743) | 0.399 (0.746) | 0.353 (0.707) | 0.391 (0.718) |
| MiniLM | 0.387 (0.773) | 0.463 (0.807) | 0.466 (0.813) | 0.397 (0.773) | 0.386 (0.782) |
| BGE-small | 0.463 (0.817) | 0.547 (0.867) | 0.545 (0.873) | 0.473 (0.827) | 0.480 (0.827) |
| E5-small | 0.457 (0.803) | 0.543 (0.860) | 0.552 (0.869) | 0.473 (0.820) | 0.505 (0.832) |
| MPNet-base | 0.403 (0.793) | 0.487 (0.843) | 0.496 (0.847) | 0.427 (0.797) | 0.421 (0.812) |
| ms-marco cross-encoder | 0.460 (0.813) | 0.550 (0.873) | 0.575 (0.877) | 0.490 (0.833) | 0.500 (0.852) |
| bge-reranker-base | 0.507 (0.837) | 0.577 (0.870) | 0.593 (0.881) | 0.530 (0.837) | 0.525 (0.852) |
| *chance hit@1* | *0.116* | *0.140* | *0.138* | *0.120* | *0.125* |
| *some localiser at rank 1* | *0.780* | *0.837* | *0.847* | *0.797* | *0.802* |

Every localiser gains 4.0–9.0 pp hit@1 under `union`. The union adds gold chunks, so
chance rises too (0.116 → 0.140), and lift over chance barely moves or falls slightly
(bge-reranker-base 4.38 → 4.13, BGE-small 4.01 → 3.91, BM25 3.05 → 2.82). Most of the raw
gain is more targets, not a different picture of localisation skill.

Holm-corrected exact McNemar on hit@1 (21 pairs per scheme), for the claims the
manuscript makes about NQ; the last column is the share of the 2,000 single-annotator
draws that reach the same conclusion as `first`:

| Pair (hit@1) | first | union | official | agreement subset | draws |
|---|---|---|---|---|---:|
| bge-reranker-base best at hit@1 | yes | yes | yes | yes | best in 98.9% |
| bge-reranker-base > MiniLM | p 0.015 | p 0.026 | p 0.017 | p 0.027 | 85.8% |
| bge-reranker-base > MPNet-base | p 0.038 | n.s. (0.121) | n.s. (0.121) | n.s. (0.148) | 34.1% |
| bge-reranker-base vs BGE-small, E5-small | n.s. | n.s. | n.s. | n.s. | 100% |
| ms-marco cross-encoder vs MiniLM | n.s. (0.417) | n.s. (0.121) | **p 0.024** | n.s. (0.130) | 71.2% |
| BGE-small, E5-small, both cross-encoders > BM25 | all p < 0.05 | all p < 0.001 | all p ≤ 0.001 | only bge-reranker-base | 92–100% |

Holm-significant pairs: 6 under `first`, 5 under `union`, 6 under `official` (a different
six), 6 under `plurality` (the same six), 2 in the agreement subset (202 questions, less
power). Under `first`, every localiser's hit@1 lies inside the central 95% of the
single-annotator draws, so the first annotator is a typical draw, not an outlier; the
draws move hit@1 by about ±2 pp, less than the sampling uncertainty (Wilson half-widths
about 5.5 pp).

## What holds and what does not

**Holds under every reference:** bge-reranker-base is the best NQ localiser; its lead over
BGE-small and E5-small is not significant; it is significantly above MiniLM; BM25 is the
weakest localiser on the full question set; rank 1 stays well below rank 5 (0.29–0.40 of
gold chunks sit at ranks 2–5, depending on scheme and localiser); the localisers miss different
questions (some localiser at rank 1 for 0.78–0.85, the best single one 0.51–0.59). The
cross-corpus finding that the reranker effect does not replicate is not an artefact of
NQ's reference: under the union construction, which is how QASPER is scored,
bge-reranker-base is still the best NQ localiser.

**Depends on the reference:** the absolute NQ levels (hit@1 0.353–0.507 becomes
0.393–0.577 under `union` and 0.399–0.593 under `official`; four of seven localisers
exceed 0.5); the median rank (under `first` it is 1 for bge-reranker-base, 2 for five
localisers and 3 for BM25; under `union` it is 1 for four localisers and 2 for the other
three); and the set of significant pairs — bge-reranker-base's significant lead over MPNet-base holds
under `first` and `plurality` only and in a third of single-annotator draws, and the
ms-marco cross-encoder's lead over MiniLM becomes significant under `official`.

## Limitations

- The union and official references add gold chunks, so part of their higher hit@k is
  mechanical; lift over chance corrects for that only against a random ordering.
- No new human judgement. Disjoint annotator spans are all counted as sufficient, as NQ's
  evaluation does; passages no annotator marked still count as misses, so `union` is a
  lower bound on sufficient evidence, not a complete reference.
- One NQ shard, 300 questions, 256/32 chunking, the manuscript's seven localisers.
  Holm correction is applied within each scheme; comparisons across schemes are
  descriptive.
- QASPER was not re-examined per annotator; its loader merges annotators.
- Post hoc: the analysis was designed after the 104/300 disagreement figure was seen,
  and was not pre-registered.
- `score_config.json` records the MiniLM index hash of a scratch copy *after* Chroma had
  opened it (opening rewrites part of `chroma.sqlite3`; the HNSW files were identical to
  `data/build/index_nq_val_300_fixed`, and the MiniLM ranks reproduced 300/300). The
  script now hashes the index before opening it.

## Recompute

From a clone, in seconds (no data or models):

```bash
python scripts/nq_reference_sensitivity.py --summarise \
    --out-dir results/exploratory/nq_reference_sensitivity
```

The per-annotator ranks themselves (needs the raw NQ file, the stored vectors under
`data/build/` — point `--minilm-chroma` at a copy of the index, because Chroma writes to
it on opening — and the cached models; about 2 h 15 min on a 12-thread laptop CPU, almost
all of it the two cross-encoders on 98 questions):

```bash
python scripts/nq_reference_sensitivity.py --score \
    --raw data/raw/nq-validation-0.parquet \
    --load-embeddings data/build/nq_val_300_vectors \
    --minilm-chroma path/to/copy/of/index_nq_val_300_fixed:exp_nq_validation \
    --ce-scope ce_msmarco_minilm=affected,bge_reranker_base=affected \
    --out-dir results/exploratory/nq_reference_sensitivity
```

`tests/test_nq_references.py` checks the schemes, the loader option, and that these
files give the manuscript's ranks back under `first` and recompute `summary.json`.

| File | Contents |
|---|---|
| `per_question.json` | per question: document size, every annotator's long answer as character ranges with its gold chunks, and per localiser the rank of each annotator's span; no dataset text |
| `score_config.json` | inputs and their checksums, model revisions, checks, runtimes, provenance of `--score` |
| `summary.json`, `summary.md` | every scheme's metrics and tests, resampling, the manuscript check, provenance of `--summarise` |
