# Experiments

Everything below was produced by the code in this repository against real
third-party corpora. Nothing is estimated, projected or illustrative.

**Read this first — what these runs can and cannot support.**

| Claim type | Status |
|---|---|
| Retrieval and evidence measurement | **Real.** Real corpora, real MiniLM embeddings, real ChromaDB retrieval, real human-annotated gold spans. |
| Failure attribution between retrieval and generation | **Real for the retrieval side.** A row is charged to retrieval on evidence grounds that do not involve the generator. |
| Generation quality, hallucination rates, LLM comparison | **Not supported.** No API key was available, so the generator is a deterministic extractive control, not a language model. |
| Agreement between the taxonomy and human judgement | **Not measured.** The annotation package exists; no labels have been collected. |
| Multi-hop evidence behaviour | **Real.** Demonstrated on HotpotQA distractor: 150 genuine 2-hop questions, all `all_required`. |

---

## Setup

| | |
|---|---|
| Retriever | ChromaDB, cosine similarity |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (real, 384-dim) |
| Chunking | 256 tokens, 32 overlap, tiktoken `cl100k_base` |
| top-k | 5 |
| Generator | `MockExtractiveLLM` — deterministic extractive control |
| Taxonomy | v2.0 |

The generator copies the sentence from retrieved context with the greatest
overlap with the question. It bounds generation quality from below and makes
runs fully deterministic. Crucially, **it does not affect the retrieval or
evidence measurements**, which are what the headline result is about.

---

## Result 1 — Document-level retrieval metrics substantially overstate success

The central measurement. For each question, two definitions of "retrieval
succeeded" are applied to the *same* retrieval output:

- **Document-level** (conventional): did any retrieved chunk come from a
  relevant document?
- **Evidence-level**: did a retrieved chunk actually contain the labelled
  supporting span, by character-offset overlap?

### Three datasets, three evidence structures

| | QASPER dev | NQ validation | HotpotQA distractor |
|---|---|---|---|
| n paired | **290** | 60 | **150** |
| Evidence structure | paragraph | span | **2-hop, all required** |
| Document-level success | 0.441 | **1.000** | **0.993** |
| Evidence-level success | 0.276 | 0.817 | 0.507 |
| **Gap** | **16.6 pp** | **18.3 pp** | **48.7 pp** |
| Discordant: doc yes, evidence no | 48 | 11 | 73 |
| Discordant: evidence yes, doc no | **0** | **0** | **0** |
| Exact McNemar *p* | 7.1 × 10⁻¹⁵ | 9.8 × 10⁻⁴ | 2.1 × 10⁻²² |
| Paired bootstrap CI on the gap | [0.124, 0.207] | — | [0.407, 0.567] |
| n sufficient by project convention | **yes** | no (n=60) | **yes** |

Wilson intervals do not overlap in either sufficiently-powered dataset:
QASPER document-level [0.385, 0.499] vs evidence-level [0.228, 0.330];
HotpotQA [0.963, 0.999] vs [0.427, 0.586].

The effect replicates across three corpora with different document types and
three different evidence granularities, and the discordance is **entirely
one-directional in all three**. That direction is forced by the construction —
evidence-level coverage implies document-level coverage, never the reverse — so
its appearance in the data is a consistency check on the implementation as much
as a result.

### The multi-hop case is the sharpest

HotpotQA questions each require evidence from **two** documents
(`evidence_mode = all_required`). Document-level retrieval reports 0.993 — it
sees essentially no retrieval failures at all. Evidence-level measurement finds:

| Evidence status | count |
|---|---|
| complete (both documents) | 76 |
| **partial (one of two)** | **73** |
| none | 1 |

**Half the dataset received exactly one of the two documents it needed.** A
document-level metric counts every one of those as a retrieval success, because
*a* relevant document was retrieved. Under `all_required` they are retrieval
failures, and the generator could not have answered them however good it was.

Partial evidence is therefore not a theoretical edge case introduced for
completeness — on a real multi-hop corpus it is the single largest category.

### Why the gap is smaller on QASPER at n=300 than at n=60

The QASPER gap moved from 31.0 pp (n=58) to 16.6 pp (n=290), because
document-level success itself fell from 0.707 to 0.441 as the corpus grew from
22 to 111 papers and retrieval got harder. Evidence-level success fell too
(0.397 → 0.276). The *gap* narrowed; the *direction and significance*
strengthened. Quoting a single gap figure as a constant would be wrong: it is a
property of a corpus and a retrieval configuration.

## Result 2 — Failure attribution changes materially

Same runs, same rows; only the attribution rule differs. Document-level
attribution can only reason "the document was retrieved, so what went wrong
must be generation".

**QASPER dev, n = 300**

| Attributed to | Document-level | Evidence-level |
|---|---|---|
| retrieval | 162 | **210** |
| generation | 131 | 74 |
| abstention | — | 10 |
| none | 7 | 6 |

**NQ validation, n = 60**

| Attributed to | Document-level | Evidence-level |
|---|---|---|
| retrieval | **0** | **11** |
| generation | 42 | 35 |
| none | 18 | 14 |

**HotpotQA multi-hop, n = 150**

| Attributed to | Document-level | Evidence-level |
|---|---|---|
| retrieval | **1** | **74** |
| generation | 111 | 49 |
| none | 38 | 27 |

The HotpotQA row is the clearest statement of the problem. A document-level
reading attributes a single failure out of 150 to retrieval and charges 111 to
generation. Evidence-aware attribution charges 74 to retrieval, because in
those rows one of the two required documents never reached the generator.

This is the practical consequence of Result 1. An engineer acting on the
document-level report would tune prompts; the evidence-level report says the
retriever is the binding constraint.

---

## Result 3 — Dataset properties measured, not assumed

| | QASPER dev | NQ validation | HotpotQA |
|---|---|---|---|
| Questions loaded | 400 | 300 | 150 |
| Documents | 128 papers | 297 pages | 1,491 paragraphs |
| Mean document length | 21,831 chars | 37,181 chars | ~1,100 chars |
| Answerable / unanswerable | 386 / 14 | 300 / 0 | 150 / 0 |
| Questions with >1 reference answer | — | **49%** | — |
| Mean gold spans per question | 2.16 | 1.0 | 2.0+ |
| Questions requiring >1 document | 0 | 0 | **150 (all)** |
| Duplicate page content under distinct ids | — | **0** | — |
| Dataset validation failures | **0 / 400** | **0 / 300** | **0 / 150** |
| Corpus offset mismatches at index time | **0** | **0** | **0** |

Every gold span in all three datasets resolves exactly against the document
text the loader built — 0 validation failures across 850 questions — and
`build_corpus` re-verified `document[start:end] == chunk.text` for every chunk
at index time, including 2,272 chunks over 2.4 M characters of QASPER, with 0
mismatches. That is the precondition for every evidence number above.

**NQ loader census (300 questions emitted):** 153 items skipped as page-scoped
nulls — correctly *not* treated as corpus-scoped unanswerables — and 90 skipped
for having no extractable short answer.

**QASPER evidence census (dev split):** 384 of 2,808 evidence strings (13.7%)
have no exact position in the paper body. 253 are `FLOAT SELECTED:` figure and
table captions, which QASPER stores outside `full_text`. A question supported
only by those is not answerable from the text corpus and is excluded rather
than reported as a retrieval failure no retriever could have avoided.

---

## Reproducing

```bash
# See docs/DATASETS.md for the raw-data download commands and checksums.
python scripts/run_experiment.py --dataset qasper \
    --raw data/raw/qasper-dev-v0.3.json --split dev \
    --limit 60 --top-k 5 --embedder minilm \
    --out reports/experiments/pilot_qasper --tag pilot_qasper

python scripts/run_experiment.py --dataset nq \
    --raw data/raw/nq-validation-0.parquet --split validation \
    --limit 60 --top-k 5 --embedder minilm \
    --out reports/experiments/pilot_nq --tag pilot_nq

# The paired methodology comparison (no model calls)
python scripts/run_ablation.py \
    --records reports/experiments/pilot_qasper/inference.jsonl \
    --out reports/experiments/ablation_qasper.json --tag "QASPER dev pilot"
```

Each report carries a `provenance` block with the git commit, the raw file's
SHA-256, the split, chunking and retrieval configuration, embedder and
generator identity, taxonomy version and threshold fingerprint, and package
versions.

---

## Threats to validity

**The generator is not a language model.** Everything about generation quality
in these runs is a property of the extractive control. No hallucination rate,
no faithfulness number and no generation-side comparison should be read from
them. The retrieval and evidence results do not depend on the generator.

**Contamination is mitigated, not eliminated.** NQ is built from Wikipedia,
which is in every current LLM's pretraining data. `answer_grounded` separates
"correct" from "correct **and** supported by retrieved evidence", and a correct
answer without its evidence is charged to retrieval rather than counted as
success. That is a mitigation. With a real LLM the risk would be materially
larger.

**One retrieval configuration.** A single embedder, chunk size and top-k. The
size of the doc/evidence gap certainly depends on all three — a smaller chunk
size would narrow it mechanically. The *direction* cannot reverse, since
evidence-level coverage implies document-level coverage, but the magnitude is
configuration-specific and should not be quoted as a constant.

**Multi-hop is untested empirically.** The `all_required` path has unit tests
but neither corpus produced a question needing two documents. HotpotQA is
loaded by the implementation but has not been run.

**Sample size.** The pilots are n≈60. The McNemar results are significant
because the discordance is large and one-directional, but the *rate* estimates
carry wide Wilson intervals at this n, and they are reported with them.

**The taxonomy is unvalidated against humans.** Its thresholds were tuned by
inspection on a 20-question fixture, which is therefore development data. The
Result 1 and 2 measurements do not depend on those thresholds — evidence
alignment is threshold-free apart from `min_overlap_chars = 1` — but the
failure-mode distributions do.
