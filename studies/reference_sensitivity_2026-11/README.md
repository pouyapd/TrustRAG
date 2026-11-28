# Reference construction in evidence-level retrieval evaluation (study of November 2026)

Code, text-free per-question data and results of the current TrustRAG study. The study is reported in
a single-author journal manuscript, which has not been published and is not in this repository.
Overview, figures and main findings: [top-level README](../../README.md).

## What is in this folder

| Path | Content |
|---|---|
| `results/raw/` | Per question and model: the complete best-first chunk ordering, every annotator's gold chunk positions, the chunk character ranges and the global top-20. Full samples (QASPER 2,197, NQ 1,278), the earlier study's subsets (290, 300) and the 300-question reproduction runs. **No dataset text**: the question text was removed; join on `question_id` with the corpus to recover it |
| `results/chunk_sensitivity/` | Text-free orderings at 128, 256 and 512 tokens on the subsets (BM25, MiniLM-L6, ms-marco CE) |
| `results/<corpus>/analysis.json` | Per-corpus analysis: scores under every reference, intervals, pairwise tests, single-annotator draws, agreement, penalised evidence, reach |
| `results/audit/` | Cluster-level inference, seven-model inference, overlap-rule and chunk-size sensitivity, statistical audit (including the uniform random-gold control), position-matched control, tie audit, adjudication audit, numeric consistency |
| `results/feasibility/`, `FEASIBILITY_REPORT.md` | CPU throughput of the candidate models |
| `data/colbert/` | ColBERT-small orderings on full NQ, used to build the NQ subset file |
| `annotation_protocol/` | Protocol, guidelines and templates for an independent second derivability adjudication and a blinded repeat (not yet carried out) |
| `scripts/`, `tests/` | Analysis pipeline and tests |

**Relation to the manuscript's supplementary material.** The manuscript's supplementary material is an
anonymized, self-contained copy of these scripts and files, made for double-anonymized review.
Before this update (see the [CHANGELOG](../../CHANGELOG.md)), this folder held an earlier version of the per-corpus analysis (without the
cluster-level primary test, NQ page clustering or the McNemar fields), and not the cluster-level
inference, sensitivity, position-matched control or verification scripts. The orderings and gold
chunks in `results/raw/` were already identical to the analysed ones.

## Reproduce

Run every command from this folder. Steps 1–10 need no corpus, model or internet (about 30 minutes
on a laptop CPU); outputs can be compared with the committed files (`git diff --stat results`).

| Step | Command | Output | Time |
|---|---|---|---|
| 1 | `python scripts/build_subsets.py` | `results/raw/nq_sub300.json.gz` (rebuilt); `qasper_sub290.json.gz` updated in place | seconds |
| 2 | `python scripts/analyse_v2.py results/raw/<f>.json.gz --out results/<f>` for `qasper_full`, `nq_full`, `qasper_sub290`, `nq_sub300`, `repro_qasper_300` | `results/<f>/analysis.json` | 12–20 min |
| 3 | `python scripts/cluster_inference.py` | `results/audit/cluster_inference.json` | about 1 min |
| 4 | `python scripts/seven_inference.py` | `results/audit/seven_inference.json` | seconds |
| 5 | `python scripts/overlap_sensitivity.py` | `results/audit/overlap_sensitivity.json` | about 1 min |
| 6 | `python scripts/chunk_analysis.py` | `results/audit/chunk_sensitivity.json` | seconds |
| 7 | `python scripts/stat_audit.py` | `results/audit/statistical_audit.json` (metric re-derivation, α, McNemar against SciPy, power, uniform random-gold control) | 7–15 min |
| 8 | `python scripts/position_control.py` | `results/audit/position_control.json` (checks first that it reproduces the uniform control of step 7) | about 2 min |
| 9 | `python scripts/make_tables.py` | `manuscript/generated/*.tex` (tables, number macros) and `figures/*.pdf`, written locally and not tracked (`.gitignore`) | seconds |
| 10 | `python scripts/verify_numbers.py` | `results/audit/numeric_consistency.json`: independent re-implementation, compared with the analyses, macros and tables | seconds |
| 11 | `pytest tests -q` | unit tests | seconds |

Dependencies: `pip install -r ../../requirements.txt krippendorff matplotlib` (Python 3.11 or newer;
the results were produced with Python 3.12.7, see `environment_freeze.txt`).

**Steps that need the corpora, the models or the internet.** Fetch the corpora as described in
[docs/DATASETS.md](../../docs/DATASETS.md).

| Script | Needs | Purpose |
|---|---|---|
| `score_v2.py` | corpora, models | scores BM25, MiniLM and the ms-marco cross-encoder and stores full orderings |
| `add_model.py`, `add_cross_encoder.py` | corpora, models | adds Granite-30M, Jina-turbo (ONNX), BGE-M3 or a cross-encoder to a scored file; resumable |
| `colbert_bridge.py` | corpora, PyLate (separate environment) | ColBERT-small orderings |
| `tie_audit.py` | corpora (question text) | recomputes BM25 scores and the effect of ties |
| `adjudication_audit.py` | annotation package, NLI model | NLI cross-check of the derivability labels |
| `benchmark_models.py` | corpora, models | CPU throughput (machine-specific) |
| `check_reproduction.py` | earlier-study ranks | compares the scorer with the earlier study's stored ranks |
| `gpu_score.py` | GPU | GPU scorer for larger models. **Not run**; no result from it is reported |

The per-question files record the corpus files they were scored from (`config.raw_files`: path relative to the
repository root and SHA-256); fetch them to `data/raw/` as described in `docs/DATASETS.md`.

Seeds: cluster bootstrap 0, permutations 1, power 2, random-gold controls 3, subsamples 7,
single-annotator draws 20261123. Model revisions are the Hugging Face commits recorded in each
scored file's `config.models`.

## Position-matched random-gold control

The uniform control (`stat_audit.py`, control G) adds, on every question where the union adds chunks
to the first annotator's gold, the same number of chunks drawn uniformly from the document's
non-gold chunks. `position_control.py` replaces each added chunk instead by a random non-gold chunk at
the same distance, in chunk positions, from the first annotator's nearest gold chunk, taking the
nearest available distance when no such chunk is left (exact match in 61% of QASPER and 46% of NQ
draws; within one chunk in 91% and 72%). Real gain ÷ matched gain: 1.7–3.3 on QASPER and 3.0–5.3 on
NQ, against 1.7–3.9 and 3.8–9.3 for the uniform control; for every model, the cluster bootstrap 95%
interval of the real minus matched gain excludes zero.

## Limitations

- Small models at full scale (22M–38M parameters); BGE-M3 only on 290 QASPER questions, ColBERT-small
  only on 300 NQ questions; no LLM reranker.
- Chunk-size sensitivity only on the subsets and with three models; the compression of model
  differences is established at 256-token chunks.
- The derivability adjudication (7 of 133 units) has one adjudicator and no reliability estimate; an
  NLI cross-check bounds the rate between 0% and 5.3%.
- Holm correction within each corpus and reference, not across all analyses.
- No measurement of answer quality.
