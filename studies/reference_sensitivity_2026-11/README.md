# Reference construction in evidence-level retrieval evaluation (November 2026)

**Status.** These are the code and results for a revised manuscript, *Whose evidence counts?
Reference construction and completeness in evidence-level retrieval evaluation on QASPER and
Natural Questions* (Pouya Bathaei Pourmand). The manuscript is in preparation for journal
submission. It has not been submitted, peer-reviewed or published, and it is not in this
repository. It replaces the September 2026 manuscript (tag `v1.1.0`, described in the
[top-level README](../../README.md)). *Language Resources and Evaluation* desk-rejected that
version on 20 November 2026, without peer review.

This folder adds new files only. It changes no file that tag `v1.1.0` reports.

## Question

Evidence-level retrieval evaluation credits a retriever only when it returns the passage that
supports the answer, judged against human evidence annotations. Many QA datasets have several
annotators per question, so a study must decide how to combine their annotations into one
reference. How much does that decision change scores, model comparisons and significance
conclusions? And how much valid evidence does a single-annotator reference miss?

## Design

- **Within-document localisation.** For every answerable question, every chunk of the gold
  document (256 tokens, overlap 32) is ranked against the question. The full ordering is stored,
  so any reference can be rescored without running a model again. The rank-1 hit rate is compared
  with an exact chance level, `P(rank ≤ k) = 1 − C(D−m, k) / C(D, k)`.
- **References.** First annotator, union, plurality and agreement (QASPER); first, union,
  plurality and the official NQ rule (Natural Questions); and one annotator drawn at random per
  question (1,000 draws).
- **Data.** QASPER dev + test v0.3, 2,197 answerable questions. Natural Questions validation
  shards 0–1, 1,278 questions. 3,475 questions in total. The earlier study's 290 QASPER and 300 NQ
  questions are analysed as subsets.
- **Localisers.**

  | Model | Family | Params | Year | Scope |
  |---|---|---:|---|---|
  | BM25 (Okapi) | lexical | — | — | full samples |
  | all-MiniLM-L6-v2 | bi-encoder | 22M | 2021 | full samples |
  | granite-embedding-30m-english | bi-encoder | 30M | 2024 | full samples |
  | ms-marco-MiniLM-L-6-v2 | cross-encoder | 22M | 2021 | full samples |
  | jina-reranker-v1-turbo-en | cross-encoder | 38M | 2024 | full samples |
  | answerai-colbert-small-v1 | late interaction | 33M | 2024 | full QASPER; NQ 300-question subset |
  | bge-m3 (dense) | bi-encoder | 568M | 2024 | QASPER 290-question subset |
  | bge-small, e5-small, all-mpnet-base, bge-reranker-base | bi-/cross-encoders | 33M–278M | 2021–2023 | 290/300-question subsets (earlier study) |

  Models were chosen from throughput measured on this laptop's CPU
  ([`results/feasibility/benchmark.json`](results/feasibility/benchmark.json)). Larger models were
  not run at full scale; see [Limitations](#limitations).
- **Statistics.** Exact McNemar tests with Holm correction, document-cluster bootstrap
  (4,000 draws), document-level sign-flip permutation tests (20,000), simulated power and minimum
  detectable differences, and Krippendorff's α with question-cluster intervals. Also a random-gold
  control, sample splits (QASPER dev/test, NQ shard 0/1) and a tie audit.

## Main results (full samples)

| | QASPER | Natural Questions |
|---|---|---|
| Questions | 2,197 | 1,278 |
| hit@1 gain, first annotator → union | +4.2 to +7.3 points | +5.6 to +9.1 points |
| Range from redrawing the single annotator | ≤ 2.0 points | ≤ 2.0 points |
| Random-gold control (real gain ÷ random gain) | 1.7–3.9× | 3.8–9.3× |
| Annotator pairs that mark disjoint evidence | 20% | 14% |
| Krippendorff's α (chunk level, 95% CI) | 0.65 (0.63–0.67) | 0.81 (0.78–0.85) |
| Rank-1 misses that another annotator marked as gold | 8–14% | 10–18% |
| Pairwise conclusions whose significance depends on the reference | 1 pair | 1 pair |

- **The reference rule moves scores more than the choice of annotator does.** Moving from the
  first annotator to the union raises rank-1 localisation for every model. Redrawing the
  annotator under a fixed rule moves it by at most 2 points.
- **The gain is mostly real evidence.** Adding the other annotators' evidence raises hit@1
  1.7–9.3 times as much as adding the same number of random chunks.
- **A single-annotator reference compresses differences between models.** Models that differ
  significantly never change order. But the union raises the neural localisers more than BM25,
  so with one annotator their differences shrink. Example: on NQ, MiniLM-L6 vs BM25 is +2.2
  points (p = 0.32) under the first annotator and +5.0 points (p = 0.005) under the union. That
  first-annotator test has about 17% power, so this is an illustration of the mechanism. Which
  pair changes significance differs between sample splits.
- **Missed evidence grows with model strength.** The share of rank-1 misses that another
  annotator marked as gold rises with hit@1 (Spearman 1.00 on QASPER, 0.90 on NQ).
- **Robustness.** Permutation tests agree with McNemar on every comparison. Ties have a
  negligible effect (≤ 0.04 points). The same pattern holds for the models that were too slow
  for the full samples: on the earlier study's 290 QASPER questions, BGE-M3 (568M) gains
  5.1 points from the union reference, and on its 300 NQ questions ColBERT-small gains 8.3
  points. On both subsets, every neural model gains more than BM25.

Per-model scores under every reference, with intervals, are in `results/<corpus>/analysis.json`.

## Verification

- 228 reported metric values were re-derived with separate code; the largest difference is
  3e-17 ([`results/audit/statistical_audit.json`](results/audit/statistical_audit.json)).
- Krippendorff's α matches the `krippendorff` package exactly. McNemar matches `scipy`'s
  binomial test. The exact chance level matches brute-force enumeration.
- The new scorer reproduces the earlier study's stored ranks with 0 mismatches
  (`results/repro_qasper_300/`, `scripts/check_reproduction.py`).
- BM25 orderings recomputed from scores: 2,197/2,197 and 1,278/1,278 identical
  ([`results/audit/tie_audit.json`](results/audit/tie_audit.json)).
- 13 unit tests: `pytest studies/reference_sensitivity_2026-11/tests -q`.
- The analyses in `results/*/analysis.json` were re-run from the published text-free files
  under `results/raw/` and matched the stored outputs exactly (25 November 2026).

## Reproduce

From a clone, without corpora or models (about one minute per corpus):

```bash
pip install -r requirements.txt krippendorff
S=studies/reference_sensitivity_2026-11
python $S/scripts/analyse_v2.py $S/results/raw/nq_full.json.gz     --out $S/results/nq_full
python $S/scripts/analyse_v2.py $S/results/raw/qasper_full.json.gz --out $S/results/qasper_full
git diff --stat $S/results            # only the recorded settings block may differ
pytest $S/tests -q
```

`results/raw/*.json.gz` hold, per question: the full best-first chunk ordering for every model,
every annotator's gold chunk positions, the chunk character ranges, and the global top-20 for
BM25 and the bi-encoders. **They contain no dataset text.** The question text was removed before
release; join on `question_id` with the corpus to recover it.

To re-score from the corpora (CPU hours; fetch the corpora as described in
[docs/DATASETS.md](../../docs/DATASETS.md) first):

```bash
python $S/scripts/score_v2.py --help                  # BM25, MiniLM, ms-marco CE; full orderings
python $S/scripts/add_model.py <scored> --model granite30m   # resumable; also jina_turbo, bge_m3
python $S/scripts/colbert_bridge.py export|score|merge <name>  # ColBERT via PyLate, separate venv
python $S/scripts/stat_audit.py                       # statistical audit
python $S/scripts/tie_audit.py                        # tie audit (needs the corpora)
```

`scripts/gpu_score.py` is a GPU scorer for larger models (BGE-M3, Qwen3-Embedding-0.6B,
bge-reranker-v2-m3 and Qwen3-Reranker-0.6B), with the same protocol as `add_model.py`. **It has
not been run.** No Colab GPU could be assigned on 25 November 2026, and no result from it is
reported anywhere.

Environment: Python 3.12.7, Windows 11, CPU only (`environment_freeze.txt`). Model revisions are
the Hugging Face commits recorded in each scored file's `config.models`. Seeds are fixed in the
scripts (bootstrap 0, single-annotator draws 20261123, permutation 1, power 2, random-gold
control 3).

## Limitations

- **Small models at full scale.** The full samples use models of 22M–38M parameters. BGE-M3 ran
  only on 290 QASPER questions, and ColBERT only on 300 NQ questions. gte-modernbert and the
  ModernBERT/LLM rerankers were infeasible on this CPU (measured, see the benchmark file).
- **One chunk size** (256 tokens) and one overlap rule.
- **Single adjudicator.** The small derivability adjudication (7 of 133 units, 5.3%) was done by
  the author alone. An automated NLI cross-check contradicted none of the NO labels but could not
  confirm the YES labels, so the rate lies between 0% and 5.3%. No second human annotator has been
  recruited; inter-annotator reliability of that judgement is unmeasured.
- **Two datasets.** Datasets with one annotator per question cannot be analysed this way.
- **Size of the effect varies by split.** QASPER dev and test differ (dev has fewer second
  annotators), so pooled estimates are not split-independent.
- **No answer-quality measurement.** Nothing here is a claim about generation.

## Files

```
scripts/   score_v2.py, add_model.py, add_cross_encoder.py, colbert_bridge.py, gpu_score.py (not run),
           build_subsets.py, analyse_v2.py, stat_audit.py, tie_audit.py, adjudication_audit.py,
           benchmark_models.py, check_reproduction.py
results/   raw/ (text-free per-question orderings), <corpus>/analysis.json, audit/, feasibility/
tests/     test_revision.py (13 tests)
```

The scripts import the repository's loaders, chunker and statistics (`src/`, `scripts/`), so run
them from the repository root.
