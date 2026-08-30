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
| Multi-hop evidence behaviour | **Implemented and unit-tested, not empirically exercised.** Neither NQ nor QASPER produced a question requiring more than one document. |

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

### Pilot, n = 60 per dataset

| | QASPER dev | NQ validation |
|---|---|---|
| n paired | 58 | 60 |
| Document-level success | **0.707** | **1.000** |
| Evidence-level success | **0.397** | **0.817** |
| Gap | **31.0 pp** | **18.3 pp** |
| Discordant: doc says yes, evidence says no | **18** | **11** |
| Discordant: evidence says yes, doc says no | **0** | **0** |
| Exact McNemar *p* | **7.6 × 10⁻⁶** | **9.8 × 10⁻⁴** |

Both are significant, and the discordance is **entirely one-directional** in
both datasets. That is what the construction predicts — evidence-level coverage
implies document-level coverage, never the reverse — and its appearance in the
data is a consistency check on the implementation as much as a result.

The NQ column is the sharper illustration. Document-level retrieval scores
**1.000**: it reports that retrieval never failed, and has no discriminative
power whatsoever on this corpus. Evidence-level measurement finds that 11 of 60
questions never received their supporting passage.

**Why this happens.** NQ documents average ~37,000 characters and QASPER papers
~22,000. Any chunk from anywhere inside one satisfies the document-level test.
Median gold spans are 636 (NQ) and 560 (QASPER) characters — smaller than the
median retrieved chunk (~1,100–1,200 characters), so the evidence test is a
genuinely finer-grained question rather than a degenerate one.

---

## Result 2 — Failure attribution changes materially

Same runs, same rows; only the attribution rule differs. Document-level
attribution can only reason "the document was retrieved, so what went wrong
must be generation".

**QASPER dev, n = 60**

| Attributed to | Document-level | Evidence-level |
|---|---|---|
| retrieval | 17 | **35** |
| generation | 39 | 20 |
| abstention | — | 2 |
| none | 4 | 3 |

**NQ validation, n = 60**

| Attributed to | Document-level | Evidence-level |
|---|---|---|
| retrieval | **0** | **11** |
| generation | 42 | 35 |
| none | 18 | 14 |

On QASPER the majority verdict flips: a document-level reading blames
generation for 39 of 60 rows, while evidence-level attribution charges 35 to
retrieval, because the supporting passage was never in the context. On NQ the
document-level view attributes **zero** failures to retrieval — it structurally
cannot see any.

This is the practical consequence of Result 1. An engineer acting on the
document-level report would tune prompts; the evidence-level report says the
retriever is the binding constraint.

---

## Result 3 — Dataset properties measured, not assumed

| | QASPER dev | NQ validation |
|---|---|---|
| Questions loaded (of 300/400 attempted) | 400 | 300 |
| Documents | 128 papers | 297 pages |
| Mean document length | 21,831 chars | 37,181 chars |
| Answerable / unanswerable | 386 / 14 | 300 / 0 |
| Questions with >1 reference answer | — | **49%** |
| Mean gold spans per question | 2.16 | 1.0 |
| Questions requiring >1 document | **0** | **0** |
| Duplicate page content under distinct ids | — | **0** |
| Dataset validation failures | **0 / 400** | **0 / 300** |

Every gold span in both datasets resolves exactly against the document text the
loader built — 0 validation failures across 700 questions. That is the
precondition for every evidence number above.

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
