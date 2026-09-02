# Experimental setup

All values are read from the run artifacts in the working tree. Source file is
given per block. Note that `reports/` is listed in `.gitignore`, so these files
are produced by the commands in `reproducibility.md` rather than shipped with a
clone — see that file's §12.

---

## 1. Corpus and questions

Source: `reports/experiments/qasper_dev_300/summary.json` → `experiment`.

| Field | Value |
|---|---|
| Dataset | QASPER, `dev` split |
| Licence | CC BY 4.0 (`https://allenai.org/data/qasper`) |
| Questions evaluated | 300 |
| Documents | 111 |
| Chunks indexed | 2,272 |
| Total characters | 2,411,721 |
| Offset mismatches | 0 |
| Loader skips | 11 questions whose evidence lives only in figures/tables; 4 whose evidence could not be located |
| Answerable / unanswerable | 290 / 10 |

## 2. Retrieval configuration

Source: same file → `experiment.retrieval`, `experiment.corpus`.

| Field | Value |
|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk size | 256 tokens |
| Chunk overlap | 32 tokens |
| top-k | 5 |
| Vector store | ChromaDB, persistent, offsets carried per chunk |
| Evidence mode | `any_sufficient` (single-hop); `all_required` used for multi-hop corpora |

## 3. Generation configuration

Source: same file → `experiment.generator`.

Generator is `MockExtractiveLLM`, described in the artifact as an *extractive
control condition*: no language model is called; the control copies the sentence
from the retrieved context with the greatest overlap with the question. This
bounds generation quality from below and makes the run fully deterministic.

**Consequence for the annotation study:** the annotated outputs are extractive
spans, not free-form LLM text. `hallucination` therefore has zero support in the
reference set by construction of this run (§`results.md` §5.3).

A separate replay study (`scripts/run_llm_experiment.py`) swaps only the
generator on the same stored retrieval — Qwen2.5-0.5B-Instruct and
SmolLM2-360M-Instruct, greedy, CPU, n=150 each.

## 4. Run-level measurements

Source: `reports/experiments/qasper_dev_300/summary.json`.

| Metric | Value |
|---|---|
| Precision@k (mean) | 0.232 |
| Recall@k (mean) | 0.427 |
| MRR (mean) | 0.337 |
| Document recall@k | 0.441 |
| nDCG@k | 0.252 |
| Evidence complete rate | 0.276 |
| Evidence recall (mean) | 0.222 |
| First evidence rank (mean) | 2.55 |
| Faithfulness (mean) | 0.941 |
| Latency (mean) | 15.4 ms |
| Failure rate | 0.973 (Wilson 95% CI 0.948–0.986) |

Evidence status over the 290 answerable questions: `complete` 80, `none` 210.

## 5. Annotation package

Source: `reports/annotation/qasper_dev_300_full_context/manifest.json`.

| Field | Value |
|---|---|
| Units | 200 (sampled from the 300-question run) |
| Sampling seed | 20260826 |
| Floor per proposed failure mode | 8 |
| Boundary-case budget | 50 units (25%), margin 0.1 from a deciding threshold |
| Annotator sheets | 2, independently shuffled |
| Proposed labels | withheld in `proposed_labels_key.jsonl` |
| Retrieved chunks in the package | 1,000 (5 per unit) |
| Chunks stored complete | 1,000 / 1,000 (`text_complete = true`) |
| Gold spans stored complete | 400 / 400 |
| Longest stored chunk | 1,505 characters |

Population distribution the sample was drawn from (proposed labels over 300
questions): `wrong_retrieval` 162, `incorrect_answer` 88, `partial_answer` 23,
`answered_when_unanswerable` 10, `hallucination` 9, `ok` 8. Per-mode sampling
weights are recorded in the manifest so population proportions can be recovered.

## 6. Reference set

Source: `reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl`
and its `PROVENANCE.md`.

| Field | Value |
|---|---|
| Units labelled | 200 / 200, one label + confidence + free-text note each |
| Produced by | A language model (Claude Opus 5) reading the full retrieved context against `docs/ANNOTATION_GUIDELINES.md` |
| **Not** | a human annotation pass |
| Label distribution | `wrong_retrieval` 130, `incorrect_answer` 42, `ok` 16, `answered_when_unanswerable` 9, `partial_answer` 3 |
| Confidence distribution | high 155, medium 39, low 6 |
| Validation | `scripts/annotate.py --validate` passes all seven checks, including retrieved-context completeness |

Human-labelled units in the repository: **22**, in the earlier package
(`reports/annotation/qasper_dev_300/annotator_a/`), listed by id in that
directory's `PROVENANCE.md`.

## 7. What is being evaluated

The object under evaluation is the *taxonomy*, in two variants computed for
every row of the same stored run:

| Variant | Field | Retrieval gate |
|---|---|---|
| Document-gated | `failure_mode_v2` | rule R4 fires when no retrieved chunk came from a relevant document |
| Evidence-gated | `failure_mode_evidence` | rule R4 fires when no retrieved chunk covered a gold span |

Thresholds: taxonomy version `v2.0`, config fingerprint `4672f4ea2b70`,
`faithfulness_threshold` 0.6, `answer_f1_ok` 0.6, `key_fact_recall_ok` 1.0,
`key_fact_recall_incorrect` 0.2, `fallback_f1_incorrect` 0.1. These were tuned by
inspection on a 20-question development fixture and were **not** re-tuned against
the reference set.

## 8. Metrics

- Per-category precision / recall / F1 and a full confusion matrix
  (`src/evaluation/statistics.py::confusion_matrix`).
- Cohen's kappa with per-category breakdown (`cohens_kappa`).
- Exact McNemar for the paired variant comparison (`mcnemar_exact`).
- Wilson intervals for proportions, seeded bootstrap for means.
- `MIN_N_FOR_INFERENCE = 30` is a stated convention; estimates below it carry a
  `sufficient: false` flag.
