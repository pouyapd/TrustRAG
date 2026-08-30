# Evaluation methodology

This document describes *how* TrustRAG measures. For what the measurements
produced, see [EXPERIMENTS.md](EXPERIMENTS.md). For the failure categories, see
[TAXONOMY.md](TAXONOMY.md).

---

## The problem with a document-level answer

The conventional question a RAG evaluation asks is *did we retrieve the right
document?* On a corpus of long documents that question is nearly free. A QASPER
paper is around 22,000 characters and an NQ page around 37,000; a chunk from
anywhere inside one counts as a hit. The metric can therefore report a retrieval
success while the passage that actually supports the answer was never placed in
the generator's context — and every failure that follows is then charged to the
generator.

The narrower question TrustRAG asks is: **was the labelled supporting passage
actually in the context the generator saw?**

---

## Three layers of measurement

Each layer is kept separately and reported separately. Legacy values are never
overwritten, so previously published numbers stay reproducible.

### 1. Legacy (frozen)

`precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `token_overlap`.
Retained unchanged, with their defects documented on each function. `recall_at_k`
returns 0.0 for an unanswerable question, conflating "no answer exists" with
"retrieval failed"; `precision_at_k` divides by *k* and counts repeated document
ids as separate hits.

### 2. Corrected (document and chunk level)

Explicit about their unit — `document_*` deduplicates, `chunk_*` does not — and
returning `None` rather than 0.0 when a question has no relevant document, so an
unanswerable item is excluded from retrieval means instead of depressing them.
Adds nDCG@k, hit-rate@k and first-relevant-rank.

### 3. Evidence level (`src/evaluation/evidence.py`)

Gold spans and retrieved chunks are both half-open character ranges in the same
document, so coverage is arithmetic:

```
overlap = min(span.end, chunk.end) - max(span.start, chunk.start)
covered = overlap >= min_overlap_chars      # default 1, a declared parameter
```

Nothing is located by searching for text. That matters: `str.find()` returns the
first occurrence, so in any document that repeats itself every copy resolves to
the same position. The character offsets come from the chunker itself
(see [ARCHITECTURE.md](ARCHITECTURE.md)), which guarantees
`document[start:end] == chunk.text` by construction.

Reported per question:

| Measure | Meaning |
|---|---|
| `evidence_status` | `complete` / `partial` / `none` / `not_applicable` |
| `evidence_recall` | fraction of gold spans some retrieved chunk covers |
| `evidence_precision` | fraction of retrieved chunks carrying gold evidence — low values mean the context was mostly padding |
| `first_evidence_rank` | rank of the best chunk carrying evidence |
| `missing_evidence_doc_ids` | which required documents never arrived |
| `evidence_degraded` | a chunk lacked offsets, so alignment could not be exact |

**Multi-hop.** Under `evidence_mode = all_required` every gold *document* must
contribute a covered span. Retrieving one of two required documents is
`partial`, which is a retrieval failure — not a generation failure, even though
the retriever looks half right.

---

## Attribution hierarchy

Implemented in `evidence.attribute_stage`. The order is the substance:

1. **Unanswerable question** → judged only on whether the system abstained.
   Answering is charged to `abstention`.
2. **Nothing retrieved** → `retrieval`.
3. **Required evidence never retrieved** → `retrieval`, *whatever the answer
   looks like*. A correct answer here is not credited: without the evidence in
   context it indicates parametric knowledge, not working retrieval. The reason
   string says so explicitly.
4. **Evidence present, answer wrong** → `generation`.
5. **Evidence present, answer correct** → no failure.

`answer_grounded` is true only under (5). The gap between "answer correct" and
"answer correct **and** grounded" is the contamination signal described in
[DATASETS.md](DATASETS.md).

---

## Answer correctness

Scored against **every** acceptable answer, taking the maximum. NQ dev is
five-way annotated and QASPER accepts multiple extractive spans; privileging
`answers[0]` would understate correctness by construction. In a 300-question NQ
sample, 49% of questions carry more than one distinct reference.

Measures: normalised exact match, multiset token F1 (SQuAD-style normalisation,
with punctuation mapped to a space so `30-day` does not become the unmatchable
token `30day`), and key-fact recall — the fraction of the reference's salient
tokens present in the prediction, which is what separates an *incomplete* answer
from a *wrong* one.

For attribution, "correct" is deliberately strict: exact match, or all reference
key facts present, or normalised F1 ≥ 0.6.

---

## Two-phase design

Inference and scoring are separate. `run_inference` is the only phase that calls
a model; `score_records` is a pure function over stored `InferenceRecord`s.

The consequence is that re-scoring is free. `scripts/reclassify.py` re-labels a
finished run under different thresholds with zero model calls, which is what
makes threshold-sensitivity analysis practical rather than theoretical, and
`scripts/run_ablation.py` applies several *methodologies* to one fixed run so
differences are attributable to the measurement rather than to run-to-run noise.

---

## Statistics

| Situation | Test |
|---|---|
| A proportion | Wilson score interval |
| A mean | percentile bootstrap, seeded |
| Two methodologies, same questions, binary outcome | **exact** McNemar |
| Two methodologies, same questions, continuous | paired bootstrap |
| Two failure-mode distributions | permutation test with a chi-square statistic |

Exact McNemar rather than the chi-square approximation, because the discordant
counts are small. Permutation rather than asymptotic chi-square, because
failure-mode tables are sparse and several expected cell counts fall below 5.

Every estimate carries `n`, an interval, a `sufficient` flag and a note.
`MIN_N_FOR_INFERENCE = 30` is a stated convention, not a theorem. The report
also detects and states in its own output when a metric is **saturated** (no
discriminative power), has **zero variance**, or rests on a **rare category**.

Reported comparisons are paired by construction: every condition scores the same
stored records, so McNemar applies to exactly the discordant rows.

---

## Reproducibility

Every report carries a `provenance` block: git commit and dirty flag, dataset
name, raw-file SHA-256, split, sample size, chunk size and overlap, top-k,
embedder identity, generator identity, taxonomy version and threshold
fingerprint, Python version, platform and tracked package versions.

The offline path is fully deterministic — fixed hash embedder, extractive
generator, seeded bootstrap — so repeated runs produce identical numbers. Note
what that does *not* mean: zero run-to-run variance is reproducibility, not
generalisation. Reported intervals describe sampling uncertainty over questions.

---

## What this evaluation cannot currently tell you

- **Generation quality.** No API key was available, so experiments use the
  deterministic extractive control, which copies the best-matching sentence from
  retrieved context. It bounds generation from below and makes runs
  reproducible, but it is not a language model. Retrieval and evidence
  measurements are unaffected, because retrieval is real. No generation-side
  conclusion should be drawn from these runs.
- **Whether the taxonomy agrees with humans.** The annotation package exists
  (`scripts/build_annotation_package.py`); no labels have been collected.
- **Faithfulness.** The pipeline still scores its own output with the same model
  that produced it. With the extractive control this is arithmetically circular
  and faithfulness is constant at 1.0, which the report states in its own
  statistical notes.
