# Reproducibility map

Every analysis reported in the README and in the manuscript (prepared for submission to
*Language Resources and Evaluation*; not in this repository), mapped to the command that
produces it and the file it reads. Run from the repository root after
`pip install -r requirements.txt`. The within-document localisation study has its own
command list in [results/localisation/README.md](../../results/localisation/README.md)
(section "Reproducing"); the human study has its own in
[human_validation_final.md](human_validation_final.md) §7.

Corpora are not redistributed; see `docs/DATASETS.md` for download commands and
checksums. `reports/` is gitignored, so annotation and experiment artifacts are
produced locally rather than shipped with a clone. `results/` is tracked.

## Environment

| Item | Value |
|---|---|
| Python | 3.12.7 |
| Platform | Windows 11 (paths in commands are POSIX-style; both shells work) |
| Key packages | chromadb 1.5.8, numpy 2.4.4, sentence-transformers 5.4.1, tiktoken 0.12.0 |
| Seeds | annotation sampling 20260826; threshold split 20260906 |
| Cost | zero — no API key; all generators run locally |

## Analysis → command → output

| Analysis | What | Command | Output |
|---|---|---|---|
| Decomposition | A/B/C decomposition, 4 corpora | `python scripts/reproduce_study.py --all` | `results/decomp_*.json` |
| Decomposition, robustness | embedder / depth / chunk sweeps | `python scripts/reproduce_study.py --embedder-sweep --topk-sweep --multihop` | `results/`, `reports/experiments/*` |
| BM25 vs dense | BM25 baseline under the same definitions (the withdrawn-inversion check) | `python scripts/run_bm25_baseline.py --dataset qasper --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300 --dense-rows reports/experiments/qasper_dev_300/rows.jsonl --dense-records reports/experiments/qasper_dev_300/inference.jsonl --out results/bm25_qasper_dev_300.json` | `results/bm25_*.json` |
| Oracle evidence | paired oracle-evidence control (replication) | `python scripts/run_oracle_evidence.py --records reports/experiments/qasper_dev_300/inference.jsonl --generator qwen0.5b --limit 150 --out reports/experiments/oracle_qasper_qwen` | `.../oracle_qasper_qwen/summary.json` |
| Human study | both gates against the final human-reviewed labels | `python scripts/build_final_human_dataset.py …` then the scoring snippet in `human_validation_final.md` §7 | `final_human_reviewed/headline_vs_final_human.json` |
| Human study, provenance | original → flag → review → final | `python scripts/build_final_human_dataset.py --original … --review … --audit … --out …` | `final_human_reviewed/provenance_chain.json` |
| Human study, audit | guideline-consistency verdicts | `python scripts/audit_human_annotations.py --package … --annotator human --reference … --rows … --records … --out …` | `audit/human_annotation_audit.{json,md}` |
| Threshold ablation | held-out ablation, 144 configurations, seed 20260906 | `python scripts/threshold_ablation.py --package … --labels …/final_human_reviewed/completed.jsonl --rows … --records … --out audit/threshold_ablation.json` | `audit/threshold_ablation.json` |
| Gold spans, lexical proxy | gold-span coverage, lexical | `python scripts/audit_gold_span_coverage.py --package … --out audit/gold_span_coverage.json` | `audit/gold_span_coverage.json` |
| Gold spans, both proxies | gold-span coverage, lexical + semantic (partition of the 133 zero-coverage units) | `python scripts/audit_gold_span_semantic.py --package … --out audit/gold_span_semantic.json` | `audit/gold_span_semantic.json` |
| Gold spans, adjudication sample | stratified 60-unit sample of the 87 unresolved units, blind sheet | `python scripts/build_goldspan_adjudication.py --package … --audit audit/gold_span_semantic.json --n 60 --seed 20260907 --out reports/annotation/goldspan_adjudication` | adjudication package |
| Gold spans, estimate | under-coverage estimate from the completed sheet (census part + stratified sample, fpc) | `python scripts/score_goldspan_adjudication.py …` | `goldspan_adjudication/estimate.json` |
| Gold spans, remaining units | the 37 units the first pass left open (27 unsampled + 10 proxy-resolved "present"), same protocol, no sampling | `python scripts/build_goldspan_remainder.py --package … --audit audit/gold_span_semantic.json --first-pass reports/annotation/goldspan_adjudication --out reports/annotation/goldspan_adjudication_remaining37` | second adjudication package |
| Gold spans, review sheet | the package rendered for reading, with the gold span located in its paper | `python scripts/render_goldspan_review.py --package … --corpus data/raw/qasper-dev-v0.3.json --out …/HUMAN_REVIEW.md` | `HUMAN_REVIEW.md` |
| Gold spans, combined count | both rounds together; the unresolved stratum becomes a census, so the fpc is zero and the estimate degenerates to a count | `python scripts/score_goldspan_adjudication.py --package reports/annotation/goldspan_adjudication --remainder reports/annotation/goldspan_adjudication_remaining37 --audit … --out …/estimate_combined.json` | `goldspan_adjudication_remaining37/estimate_combined.json` |
| Localisation | within-document ranks, seven localisers, two corpora, tests, bootstrap, sweeps, contrasts | see `results/localisation/README.md` § Reproducing | `results/localisation/*.json`, `rank_distribution.png` |
| Annotation package | blinded, stratified package build | `python scripts/build_annotation_package.py --records … --out reports/annotation/qasper_dev_300_full_context --n-units 200` | package + `manifest.json` |
| Figures | all four paper figures | `pip install -r requirements-research.txt && python scripts/make_paper_figures.py --all` | `results/figures/*.png` |
| Figures | earlier study figures | `python scripts/make_figures.py --all` | `results/figures/*.png` |
| Tests | full suite | `pytest tests/ -q` | 501 tests |
| Lint | ruff | `ruff check scripts/ src/ tests/` | clean |

## Provenance recorded in every report

UTC timestamp, git commit and dirty flag, raw-file SHA-256, split, sample size, chunk
size, top-k, embedder and generator identity, taxonomy version and threshold
fingerprint, Python version, platform, package versions.

## Known reproducibility limits

- **Annotation data is not in the git tree.** `reports/` is gitignored, so the human
  study and the gold-span adjudication cannot be re-derived from a clone alone; the labels
  are annotation data, not a computation. The annotation package (200 units with retrieved
  context, original/review/final labels with provenance, audits, adjudication sheet and
  answers, threshold-ablation and oracle-evidence rows) is deposited as a separate
  checksummed archive in the Zenodo record of release v1.0.0 (version DOI
  10.5281/zenodo.22879569; concept DOI 10.5281/zenodo.22879357).
- **Approximate nearest-neighbour search.** Fine-grained aggregates on long-document
  corpora move by ≤0.001 between independently built indices. Headline A/B/C figures
  reproduce exactly; no reported gap or test changes.
- **Local model downloads.** `qwen0.5b` pulls from Hugging Face on first use.
