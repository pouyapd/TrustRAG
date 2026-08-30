# TrustRAG — Evidence-Aware RAG Evaluation

[![CI](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A production RAG service with a research evaluation layer that asks a narrower
question than usual: **not "did we retrieve the right document?" but "did the
passage that actually supports the answer reach the generator?"**

On a corpus of long documents those are very different questions. A Wikipedia
page averages ~37,000 characters and an NLP paper ~22,000, so a chunk from
anywhere inside one satisfies the conventional document-level retrieval test.
Measured on real corpora, that gap is large, one-directional, and it changes
where failures get attributed.

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

Natural Questions (CC BY-SA 3.0) and QASPER (CC BY 4.0), chosen because they
differ structurally — Wikipedia pages with span evidence versus scientific
papers with paragraph evidence. HotpotQA is implemented for multi-hop but not
yet run.

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

Real corpora, real MiniLM embeddings, real retrieval. Two definitions of
"retrieval succeeded" applied to the *same* retrieval output:

| | QASPER dev | NQ validation |
|---|---|---|
| n paired | 58 | 60 |
| Document-level success | 0.707 | **1.000** |
| Evidence-level success | 0.397 | 0.817 |
| Gap | **31.0 pp** | **18.3 pp** |
| Discordant (doc yes / evidence no) | 18 | 11 |
| Discordant (evidence yes / doc no) | **0** | **0** |
| Exact McNemar *p* | 7.6 × 10⁻⁶ | 9.8 × 10⁻⁴ |

Attribution moves accordingly. On QASPER a document-level reading blames
generation for 39 of 60 rows; evidence-level attribution charges 35 to
retrieval. On NQ the document-level view attributes **zero** failures to
retrieval and cannot see the 11 that evidence alignment finds.

Full protocol, dataset census and reproduction commands:
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
- **Multi-hop is untested empirically.** Implemented and unit-tested; neither
  corpus produced a question requiring two documents.
- **One retrieval configuration.** The size of the gap depends on chunk size,
  top-k and embedder. Its direction cannot reverse; its magnitude is not a
  constant.
- **Contamination is mitigated, not eliminated.** NQ comes from Wikipedia.
- **Sample sizes are modest** (n≈60 per dataset in the reported comparison).

---

## 11. Reproducibility

Every report carries a provenance block: git commit and dirty flag, raw-file
SHA-256, split, sample size, chunk size and overlap, top-k, embedder and
generator identity, taxonomy version and threshold fingerprint, Python version,
platform and package versions. The offline path is deterministic.

---

## 12. How to run

```bash
pip install -r requirements.txt

# Deterministic offline evaluation, no keys, no network
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
