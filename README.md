# TrustRAG — Evidence-Aware RAG Evaluation

[![CI](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-308%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-79%25-green)

> **Branch note.** `main` carries the finalized research state. The work was
> developed on **`research/stages-1-4`**, which is preserved at the same commit
> for provenance.

A production RAG service with a research evaluation layer that asks a narrower
question than usual: **not "did we retrieve the right document?" but "did the
passage that actually supports the answer reach the generator?"**

On a corpus of long documents those are very different questions. A Wikipedia
page averages ~37,000 characters and an NLP paper ~22,000, so a chunk from
anywhere inside one satisfies the conventional document-level retrieval test.
Measured on real corpora, that gap is large, one-directional, and it changes
where failures get attributed.

---

## TrustRAG in action

**The running service.** FastAPI with the RAG endpoints and the evaluation
endpoint side by side — captured from the live application, not a mockup:

![TrustRAG OpenAPI interface](docs/screenshots/api-docs.png)

**The study.** `python scripts/reproduce_study.py --all` runs all five
experiments and prints this table. Verbatim output, no API key required:

```
experiment           chunks/doc   A doc  B quant  C span   quant    gran
qasper_dev_300               19   0.441    0.441   0.276    0.0p   16.6p
nq_val_300_fixed             31   0.997    0.997   0.730    0.0p   26.7p
hotpot_150                    2   0.993    0.507   0.507   48.7p    0.0p
qasper_c128                  43   0.445    0.445   0.259    0.0p   18.6p
qasper_c512                   9   0.428    0.428   0.317    0.0p   11.0p
```

`A` is the conventional document-level retrieval metric; `C` asks whether the
gold evidence span actually reached the generator. The two right-hand columns
separate *why* they differ — a multi-hop quantifier effect from a span
granularity effect — and the last two rows show the granularity effect tracking
chunk size exactly as its mechanism predicts.

---

## 1. Project overview

Two things live here:

- **A RAG service** — FastAPI, ChromaDB, pluggable providers, Prometheus,
  Docker. Ordinary, working, and not the interesting part.
- **A research evaluation layer** — position-aware evidence alignment, a
  versioned failure taxonomy, corrected retrieval metrics, an attribution
  hierarchy, and statistics that refuse to overstate small samples.

The research question: *can RAG failures be decomposed into retrieval,
evidence, generation and abstention causes using reproducible,
evidence-grounded evaluation rather than a single aggregate score?*

---

## 2. Engineering features

- Modular pipeline — pluggable LLM, embedder and vector store
- CPU-only container: 9.53 GB → **2.99 GB** by installing the CPU PyTorch
  wheel before the rest, so `sentence-transformers` does not drag the CUDA
  runtime into an image with no GPU
- FastAPI service with structured logging and request tracing
- Prometheus metrics; Docker and docker-compose
- CI on every push: lint, tests, evaluation regression, Docker build
- Deterministic offline mode — runs with no API keys and no network

---

## 3. Research methodology

**Character offsets travel the whole path.** Chunk text is *sliced* from the
source document, never rebuilt by decoding tokens, so
`document[chunk.start_char:chunk.end_char] == chunk.text` holds by
construction. Offsets survive chunker → vector store → retrieval → stored
records.

Offsets are not recovered by searching for the chunk text. `str.find()` returns
the first occurrence, so in any document that repeats itself every copy
resolves to the same wrong position.

**Evidence alignment is arithmetic.** A gold span and a retrieved chunk are
half-open character ranges in one document; overlap decides coverage.

**Attribution refuses to credit ungrounded correctness.** A correct answer
produced without the gold evidence in context is charged to retrieval, not
counted as a success — on a Wikipedia-derived corpus that is the difference
between measuring RAG and measuring memorisation.

**Inference and scoring are separate.** Re-scoring a finished run costs no
model calls, which is what makes threshold-sensitivity analysis and
methodology comparison practical.

See [docs/EVALUATION.md](docs/EVALUATION.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Evaluation protocol

Three measurement layers, reported side by side, none overwriting another:

1. **Legacy** — frozen, with defects documented, so old numbers reproduce.
2. **Corrected** — explicit document vs chunk units; `None` instead of 0.0 for
   unanswerable questions; nDCG, hit-rate, first-relevant-rank.
3. **Evidence-level** — span coverage, evidence recall/precision, first
   evidence rank, multi-hop completeness.

---

## 5. Datasets

Natural Questions (CC BY-SA 3.0), QASPER (CC BY 4.0) and HotpotQA
(CC BY-SA 4.0), chosen because they differ structurally: Wikipedia pages with
span evidence, scientific papers with paragraph evidence, and 2-hop questions
whose evidence is split across two documents. Agreement across three such
different corpora is what makes the result more than a property of one dataset.

**Corpora are not redistributed.** `data/raw/` is git-ignored; loaders,
checksums and licence metadata are committed instead. Download commands,
checksums, split discipline and contamination analysis are in
[docs/DATASETS.md](docs/DATASETS.md).

---

## 6. Failure taxonomy

Nine categories including `incorrect_answer` separated from `partial_answer`,
`answered_when_unanswerable`, and `ok_abstained` as an explicit success.
Thresholds are versioned and hashable; every row records the rule that fired
and the features behind it. See [docs/TAXONOMY.md](docs/TAXONOMY.md).

---

## 7. Evidence-aware evaluation

```
gold span [1200, 1760)  in doc qasper:1901.00001
retrieved chunk [900, 2100) in doc qasper:1901.00001
overlap = 560 chars  ->  covered
```

Multi-hop: under `all_required` every gold document must contribute a covered
span. Retrieving one of two required documents is a *retrieval* failure, not a
generation failure.

---

## 8. Experimental results

Three definitions of "retrieval succeeded", applied to the **same** stored
retrieval output. **A** is the conventional metric (any chunk from any relevant
document). **B** additionally requires every document a multi-hop question
needs. **C** requires that a retrieved chunk actually contained the gold span.
A→B isolates the *quantifier*; B→C isolates the *granularity*.

| | QASPER dev | NQ validation | HotpotQA |
|---|---|---|---|
| n | 290 | 300 | 150 |
| median chunks per gold document | 19 | 31 | **2** |
| A document, ANY | 0.441 | 0.997 | 0.993 |
| B document, quantified | 0.441 | 0.997 | **0.507** |
| C span, quantified | 0.276 | 0.730 | 0.507 |
| **quantifier A→B** | 0.0 pp | 0.0 pp | **48.7 pp** (p=2.1e-22) |
| **granularity B→C** | **16.6 pp** (p=7.1e-15) | **26.7 pp** (p=1.7e-24) | 0.0 pp |

Two distinct blind spots in conventional retrieval metrics, each showing up on
the data where it bites:

- **Granularity blindness** — on long documents, retrieving the document is not
  retrieving the evidence.
- **Quantifier blindness** — on multi-hop questions, retrieving *a* relevant
  document counts as success when the question needs *all* of them.

Discordance is one-directional everywhere (48/0, 80/0, 73/0): span-level
success implies document-level success, never the reverse.

**The granularity effect is mechanistically explained and predicted.** It
scales with how many chunks a gold document spans — varying chunk size on
QASPER, same corpus and questions throughout:

| chunk size | chunks per gold doc | granularity gap |
|---|---|---|
| 128 | 43 | 18.6 pp |
| 256 | 19 | 16.6 pp |
| 512 | 9 | 11.0 pp |
| HotpotQA paragraphs | 2 | 0.0 pp |

So the gap **is** sensitive to chunk size, in a predictable and explained way,
and does not vanish at any realistic setting — at 512 tokens it is still 11 pp
(p = 4.7e-10). It is a property of corpus structure relative to chunk size, not
a universal constant.

Attribution moves accordingly: on NQ a document-level reading charges **1** of
300 failures to retrieval; evidence-aware attribution charges **81**.

Full protocol, dataset census, threats to validity and reproduction commands:
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## 9. Statistical uncertainty

Wilson intervals for proportions, seeded bootstrap for means, **exact** McNemar
for paired binary comparisons, permutation tests for failure-mode
distributions. Every estimate carries `n`, an interval and a `sufficient` flag;
`MIN_N_FOR_INFERENCE = 30` is a stated convention, not a theorem. Reports
detect and state saturated metrics, zero-variance metrics and rare categories
in their own output.

---

## 10. Limitations

Read these before quoting anything above.

- **No language model was used.** No API key was available, so experiments run
  a deterministic extractive control. **No generation-side conclusion — no
  hallucination rate, no faithfulness claim, no model comparison — is supported
  by these runs.** Retrieval and evidence results are unaffected, because
  retrieval is real.
- **The taxonomy is not validated against humans.** Its thresholds were tuned
  by inspection on a 20-question fixture, which is development data. The
  annotation package exists; **no labels have been collected.** The headline
  retrieval result does not depend on those thresholds.
- **Each effect rests on limited data.** The granularity effect is shown on two
  corpora across four chunk sizes; the quantifier effect on one multi-hop
  corpus (HotpotQA), whose crowdworkers wrote questions while looking at the
  paragraphs, so lexical anchoring makes its retrieval easier than natural
  queries.
- **The direction of both gaps is true by construction.** Span-level coverage
  implies document-level coverage. What is measured here is the *magnitude*,
  its dependence on corpus structure, and its consequence for attribution —
  not the existence of an inequality.
- **One embedder and one top-k.** Chunk size is swept; embedder and k are not. The size of the gap depends on chunk size,
  top-k and embedder. Its direction cannot reverse; its magnitude is not a
  constant.
- **Contamination is mitigated, not eliminated.** NQ comes from Wikipedia.
- **Sample sizes are moderate** (n = 290 / 300 / 150). Adequate for the paired
  comparison reported; not a benchmark-scale study.

---

## 11. Reproducibility

`python scripts/reproduce_study.py --all` runs every experiment and prints the
results table above. **No API key is required**: the embedder runs locally and
the generator is a deterministic extractive control, so every retrieval and
evidence measurement reproduces offline once the corpora are downloaded.

Every report carries a provenance block: git commit and dirty flag, raw-file
SHA-256, split, sample size, chunk size and overlap, top-k, embedder and
generator identity, taxonomy version and threshold fingerprint, Python version,
platform and package versions.

Determinism is verified, not assumed: re-running an experiment from scratch —
fresh index, fresh embeddings, fresh retrieval — reproduces every reported
figure exactly. Curated per-run summaries are tracked in `results/`; raw
corpora, vector indices and full reports are git-ignored.

---

## 12. How to run

```bash
pip install -r requirements.txt

# THE STUDY: all five experiments, prints the results table. No API key needed.
# Fetch the three corpora first - see docs/DATASETS.md for commands + checksums.
python scripts/reproduce_study.py --all

# Deterministic offline smoke test, no keys, no network, no corpora
python scripts/run_offline_eval.py

# Re-score a finished run under different thresholds - no model calls
python scripts/reclassify.py --records reports/inference.jsonl \
    --out reports/sweep --sweep-faithfulness 0.3,0.6,0.9

# A real experiment (see docs/DATASETS.md for the data first)
python scripts/run_experiment.py --dataset qasper \
    --raw data/raw/qasper-dev-v0.3.json --split dev \
    --limit 60 --embedder minilm --out reports/experiments/qasper

# Paired methodology comparison
python scripts/run_ablation.py \
    --records reports/experiments/qasper/inference.jsonl \
    --out reports/experiments/ablation_qasper.json

# Human-annotation package (produces empty labels for a person to fill in)
python scripts/build_annotation_package.py \
    --records reports/experiments/qasper/inference.jsonl \
    --out reports/annotation/qasper

pytest tests/ -v --cov=src
```

**308 tests, 79% line coverage**, ruff clean. Nothing is excluded from the
coverage report. The suite includes unit tests, property-style invariants (span
coverage implies document coverage, for every record), end-to-end integration
tests that carry a question from a real dataset file through chunking, a real
vector store and retrieval to a failure label, and regression tests for each
defect found during the work.

The service still runs as a service:

```bash
cp .env.example .env      # set OPENAI_API_KEY
docker-compose up --build # API on :8000, Prometheus on :9090
```

---

## License

MIT (this code). The evaluated corpora carry their own licences — see
[docs/DATASETS.md](docs/DATASETS.md).

## Author

Pouya Bathaei Pourmand — ML Engineer, safe AI and evaluation.
