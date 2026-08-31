# The bundled smoke-test fixture

**This is not the research study.** It documents the tiny fixture that runs in
CI on every push. For the actual empirical results — three corpora, 740
questions, statistical tests — see [EXPERIMENTS.md](EXPERIMENTS.md).

An earlier version of this file presented the fixture's numbers as evaluation
results and drew a conclusion from them. That conclusion was wrong, and the
correction is instructive enough to keep on the record.

---

## What the fixture is

```bash
python scripts/run_offline_eval.py
```

20 questions over a **3-document, 472-word** synthetic corpus, using
`MockExtractiveLLM` and `HashEmbeddings`. Deterministic, offline, no API key.

Its job is to be a regression gate: fast, stable, and sensitive to breakage. It
is not a benchmark, and its numbers do not characterise retrieval quality.

## Current output

| Metric | Value |
|---|---|
| Total queries | 20 |
| Recall@k (mean) | 0.90 |
| MRR (mean) | 0.833 |
| Faithfulness (mean) | 1.00 |
| Failure rate (v1 taxonomy) | 0.35 |
| Failure rate (v2 taxonomy) | 0.40 |

These values are frozen deliberately: they are what CI asserts against, and the
legacy figures reproduce exactly across every change made to the research
layer, which is how backward compatibility is checked.

## Why these numbers must not be read as evaluation results

Each of the headline figures is an artifact of the fixture's size, and an audit
established why:

- **Recall@k = 0.90 is saturation, not quality.** The corpus has three
  documents and retrieval returns four chunks, so it returns essentially the
  whole corpus on every query. Recall is **1.00 on all 18 answerable
  questions**; the 0.90 comes from the two unanswerable ones scoring 0 by
  definition. A random retriever would score the same here. The metric has no
  discriminative power on this corpus.

- **Faithfulness = 1.00 is circular.** The extractive stand-in copies a
  sentence out of the retrieved context, and the same stand-in then scores what
  fraction of the answer's tokens appear in that context. The value is 1.0 by
  construction, with zero variance across all 20 rows. It measures nothing.

- **The old conclusion was backwards.** This file previously said "the
  retrieval is fine, but the answer generation is the bottleneck". Under the v2
  taxonomy those 7 `partial_answer` rows resolve into 5 genuinely *incorrect*
  answers and 2 failures to abstain — and on the real corpora, evidence-aware
  attribution charges the **majority** of failures to retrieval. The fixture is
  far too small and too saturated to support any claim about where failures
  come from.

The report the run generates says as much in its own `statistical_notes`, which
flag the saturated retrieval metric, the zero-variance faithfulness and the
n=20 sample size.

## What it is good for

- Catching regressions in the pipeline within seconds, offline, in CI.
- Verifying backward compatibility: every legacy metric still reproduces its
  original value after the entire research layer was added.
- Exercising the abstention path — the two unanswerable questions are both
  answered confidently and wrongly, which is what first exposed that the v1
  taxonomy could not represent a failure to abstain.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/run_offline_eval.py
```

Output goes to `reports/`: `summary.json`, `rows.jsonl`, `report.md`, and
`inference.jsonl` (which lets the run be re-scored under different thresholds
with no model calls).

## Where the real results are

| | |
|---|---|
| Empirical study | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Methodology | [EVALUATION.md](EVALUATION.md) |
| Datasets and licences | [DATASETS.md](DATASETS.md) |
| Failure taxonomy | [TAXONOMY.md](TAXONOMY.md) |

```bash
python scripts/reproduce_study.py --all   # the actual study, no API key needed
```
