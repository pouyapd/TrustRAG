# QASPER localisation under alternative evidence references

**Exploratory. Not part of the submitted manuscript.** Run on 10 October 2026, after the
manuscript was submitted, as the counterpart of
[`../nq_reference_sensitivity/`](../nq_reference_sensitivity/README.md). Nothing here
changes a file in `results/localisation/`; the manuscript's QASPER numbers are reproduced
exactly below, then re-scored.

## The question

The two corpora build their gold evidence in opposite ways. QASPER's loader
(`_evidence_spans` in `src/data/loaders/qasper.py`) merges every annotator's evidence
paragraphs, and any one of them counts. NQ's loader keeps only the first annotator's long
answer. The NQ analysis asked what happens when NQ is scored the QASPER way. This one asks
the reverse: how far do the QASPER results depend on merging the annotators? It also
supplies the QASPER half of a like-for-like comparison of the two corpora
([`../reference_sensitivity_claims.md`](../reference_sensitivity_claims.md)).

## How QASPER stores several annotations

In `qasper-dev-v0.3.json` (SHA-256 `2ae7ee62…`) each question carries a list of `answers`,
one per annotator, each with `unanswerable`, the answer fields and `evidence`, a list of
paragraph strings. The loader locates each evidence string in the paper body by exact
match. Figure and table captions (`FLOAT SELECTED: …`) are not in the body and are left
out, and so is any string that matches no paragraph. A question is kept when at least one
evidence string is located.

QASPER's own evaluator (`scripts/evaluator.py` in `allenai/qasper-led-baseline`) scores a
prediction against each annotator's evidence and keeps the best score, and can drop
caption evidence (`--text_evidence_only`). For hit@k that best-of rule is the union, so the
manuscript's QASPER reference is already the dataset's official convention.

## What the 290 study questions look like

| | |
|---|---:|
| Annotator entries: evidence located / marked unanswerable / gave none / gave only captions or unmatched text | 517 / 32 / 7 / 23 |
| Questions with 1 / 2 / 3 annotators whose evidence is located | 65 / 223 / 2 |
| Evidence paragraphs per annotator (located) | 1.64 |
| Questions with two or more such annotators | 225 |
| … of which two annotators marked the same paragraph | 163 |
| … of which the annotators' evidence covers the same chunks | 122 |
| Questions where some reference below gives different gold chunks from the union | 103 |
| Questions whose first annotator's evidence is not in the text (excluded under `first`) | 15 |

## Reference schemes

Defined and tested in `src/evaluation/references.py` (section "evidence sets"):

| Scheme | Gold evidence | Questions |
|---|---|---:|
| `union` | every annotator's located paragraphs (the manuscript; QASPER's best-of rule) | 290 |
| `first` | the first annotator who gave evidence, as the NQ study reference does | 275 |
| `plurality` | the paragraphs most annotators marked; the union when none is shared | 290 |
| `agreement` | paragraphs marked by at least two annotators | 163 |
| `agreement_subset` | questions on which every scheme and every single annotator give the same gold chunks | 187 |
| `single` | one annotator drawn at random per question; 2,000 draws, seed 20261010 | 290 |

Every scheme except the union is narrower than the union, so no localiser can rank better
under it; the analysis measures how much is lost and whether the comparisons change.

## Checks before any re-scoring

- Same 290 questions in the same order, same 2,272 chunks, same document sizes and union
  gold chunk counts as `results/localisation/within_document_qasper.json`; the union of
  the per-annotator evidence equals the loader's evidence for every question; with the new
  option off, the loader's records for the whole dev split are byte-identical to before.
- Recomputed union ranks equal the stored ranks for every question: BM25 and the four
  bi-encoders on all 290, both cross-encoders on the 103 questions where some reference
  differs from the union. On the other 187 every scheme gives the union's gold chunks, so
  the stored rank is exact there and the cross-encoders were not re-run.
- Under `union`, hit@1, hit@5, MRR, every McNemar and Holm p-value and the union figure
  (0.769) are identical to `results/localisation/summary_qasper.json`.
- The four stored Chroma indexes were copied before opening; the copies hashed identical
  to `data/build/index_qasper_dev_300_emb_*`, and the originals were unchanged afterwards.
- Model revisions are those in `results/localisation/model_metadata.json`.

## Results

hit@1 (hit@5); Wilson intervals, MRR, rank distributions and lifts are in `summary.json`.

| Localiser | union | first | plurality | agreement | agreement subset |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.290 (0.755) | 0.269 (0.669) | 0.276 (0.707) | 0.282 (0.644) | 0.299 (0.733) |
| MiniLM | 0.376 (0.845) | 0.331 (0.785) | 0.352 (0.810) | 0.362 (0.797) | 0.353 (0.818) |
| BGE-small | 0.400 (0.845) | 0.360 (0.778) | 0.369 (0.814) | 0.380 (0.785) | 0.401 (0.813) |
| E5-small | 0.359 (0.821) | 0.316 (0.774) | 0.328 (0.793) | 0.325 (0.761) | 0.321 (0.791) |
| MPNet-base | 0.334 (0.797) | 0.284 (0.735) | 0.317 (0.759) | 0.301 (0.712) | 0.294 (0.754) |
| ms-marco cross-encoder | 0.386 (0.855) | 0.338 (0.785) | 0.372 (0.824) | 0.387 (0.779) | 0.412 (0.829) |
| bge-reranker-base | 0.310 (0.797) | 0.262 (0.738) | 0.297 (0.776) | 0.307 (0.749) | 0.316 (0.775) |
| *gold paragraphs per question* | *2.09* | *1.63* | *1.79* | *1.47* | *1.55* |
| *chance hit@1* | *0.157* | *0.127* | *0.141* | *0.112* | *0.124* |
| *some localiser at rank 1* | *0.769* | *0.698* | *0.741* | *0.699* | *0.733* |

On the 275 questions it keeps, `first` costs every localiser 2.9–5.5 pp hit@1 and
4.0–8.4 pp hit@5 against the union. Chance falls
with the gold (0.157 → 0.127), so lift over chance rises (BGE-small 2.54 → 2.84,
bge-reranker-base 1.97 → 2.07). The median rank moves from 2 to 3 for E5-small and both
cross-encoders; it stays 2 for MiniLM and BGE-small and 3 for BM25 and MPNet-base.

Holm-corrected exact McNemar on hit@1 (21 pairs per scheme); the last column is the share
of the 2,000 single-annotator draws that reach the same conclusion as `union`:

| Pair (hit@1) | union | first | plurality | agreement | agreement subset | draws |
|---|---|---|---|---|---|---:|
| BGE-small > BM25 | p 0.023 | n.s. (0.186) | n.s. (0.090) | n.s. (0.657) | n.s. (0.330) | 27.5% |
| ms-marco cross-encoder > BM25 | p 0.026 | n.s. (0.330) | p 0.019 | n.s. (0.160) | n.s. (0.065) | 65.1% |
| BGE-small vs bge-reranker-base | n.s. (0.525) | n.s. (0.265) | n.s. (1.0) | n.s. (1.0) | n.s. (1.0) | 100% |
| BM25 vs bge-reranker-base | n.s. (1.0) | n.s. (1.0) | n.s. (1.0) | n.s. (1.0) | n.s. (1.0) | 100% |

Holm-significant pairs: 2 under `union`, 0 under `first`, 1 under `plurality`, 0 under
`agreement` and in the agreement subset. Best localiser at hit@1: BGE-small under `union`
and `first`, the ms-marco cross-encoder under the other three. Worst neural localiser:
bge-reranker-base under `union`, `first` and `plurality`, MPNet-base under `agreement` and
in the agreement subset; over single-annotator draws bge-reranker-base is the worst neural
localiser in 67.2% and MPNet-base in 32.8%, and the best localiser is BGE-small in 53.9%,
the ms-marco cross-encoder in 40.4% and MiniLM in 5.7%. Under `first`, bge-reranker-base
(0.262) falls just below BM25 (0.269); the difference is not significant.

## What holds and what does not

**Holds under every reference:** rank 1 stays well below rank 5; the localisers miss
different questions (some localiser at rank 1 for 0.70–0.77, the best single one
0.36–0.41); no difference among the four bi-encoders is significant after correction and
MPNet-base stays below BGE-small; bge-reranker-base is never the best QASPER localiser,
never significantly different from BM25, and is the worst or second-worst neural one.

**Depends on the reference:** the absolute QASPER levels, which drop by 2.9–5.5 pp under
`first`; which localiser is best (BGE-small or the ms-marco cross-encoder, never
significantly apart); and both of the manuscript's significant QASPER pairs. BM25's
deficits against BGE-small and the ms-marco cross-encoder survive Holm correction only
under the union (and, for the ms-marco cross-encoder, under `plurality`), and in 27.5% and
65.1% of single-annotator draws. Under `first`, BM25 is not the weakest QASPER localiser.

## Limitations

- No new human judgement. The union counts every annotator's paragraph as sufficient;
  narrower references count a paragraph only one annotator chose as a miss, although it
  was an annotator's choice of evidence. They are sensitivity checks, not better gold.
- Narrower references leave fewer gold chunks, so part of the lower hit@k is mechanical;
  lift over chance corrects for that only against a random ordering.
- `agreement` and the agreement subset keep 163 and 187 questions and select them by
  annotator behaviour, so they are not random subsamples and have less power.
- 290 questions in 111 papers, 256/32 chunking, the manuscript's seven localisers. Holm
  correction is applied within each scheme; comparisons across schemes are descriptive.
- Post hoc: designed after the NQ analysis, not pre-registered.

## Recompute

From a clone, in seconds (no data or models):

```bash
python scripts/qasper_reference_sensitivity.py --summarise \
    --out-dir results/exploratory/qasper_reference_sensitivity
```

The per-paragraph ranks themselves (needs the raw QASPER dev file, copies of the four
stored indexes, because Chroma writes to an index when it opens it, and the cached models;
about 21 min on a 12-thread laptop CPU, almost all of it bge-reranker-base on 103
questions). Scored at commit `86b640c` on a clean tree:

```bash
python scripts/qasper_reference_sensitivity.py --score \
    --raw data/raw/qasper-dev-v0.3.json \
    --chroma-map "minilm=COPY/index_qasper_dev_300_emb_minilm:exp_qasper_dev,mpnet=COPY/index_qasper_dev_300_emb_mpnet:exp_qasper_dev,bge=COPY/index_qasper_dev_300_emb_bge:exp_qasper_dev,e5=COPY/index_qasper_dev_300_emb_e5:exp_qasper_dev" \
    --ce-scope ce_msmarco_minilm=affected,bge_reranker_base=affected \
    --out-dir results/exploratory/qasper_reference_sensitivity
```

`tests/test_qasper_references.py` checks the schemes, the loader option, the summary
arithmetic on a synthetic file, and that these files give the manuscript's ranks back
under `union` and recompute `summary.json`.

| File | Contents |
|---|---|
| `per_question.json` | per question: document size, every annotator's status and located evidence as character ranges, each distinct paragraph with the annotators who chose it and its gold chunks, and per localiser the union rank and each paragraph's rank; no dataset text |
| `score_config.json` | inputs and their checksums, model revisions, checks, runtimes, provenance of `--score` |
| `summary.json`, `summary.md` | every scheme's metrics and tests, resampling, the manuscript check, provenance of `--summarise` |
