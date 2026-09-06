# Reproducibility map

Every table and figure in `paper.md`, mapped to the command that produces it and the
file it reads. Run from the repository root after `pip install -r requirements.txt`.

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

## Paper section → command → output

| Paper | What | Command | Output |
|---|---|---|---|
| §5 Table | A/B/C decomposition, 4 corpora | `python scripts/reproduce_study.py --all` | `results/decomp_*.json` |
| §5 robustness | embedder / depth / chunk sweeps | `python scripts/reproduce_study.py --embedder-sweep --topk-sweep --multihop` | `results/`, `reports/experiments/*` |
| §5.1 Table, Fig. bm25 | BM25 vs dense | `python scripts/run_bm25_baseline.py --dataset qasper --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300 --dense-rows reports/experiments/qasper_dev_300/rows.jsonl --dense-records reports/experiments/qasper_dev_300/inference.jsonl --out results/bm25_qasper_dev_300.json` | `results/bm25_*.json` |
| §6 Table, Fig. oracle | paired oracle-evidence control | `python scripts/run_oracle_evidence.py --records reports/experiments/qasper_dev_300/inference.jsonl --generator qwen0.5b --limit 150 --out reports/experiments/oracle_qasper_qwen` | `.../oracle_qasper_qwen/summary.json` |
| §7 Table, Fig. human | gates vs final human labels | `python scripts/build_final_human_dataset.py …` then the scoring snippet in `human_validation_final.md` §7 | `final_human_reviewed/headline_vs_final_human.json` |
| §7 provenance | original → flag → review → final | `python scripts/build_final_human_dataset.py --original … --review … --audit … --out …` | `final_human_reviewed/provenance_chain.json` |
| §7 audit | guideline-consistency verdicts | `python scripts/audit_human_annotations.py --package … --annotator human --reference … --rows … --records … --out …` | `audit/human_annotation_audit.{json,md}` |
| §7.1 Table | threshold ablation, held-out | `python scripts/threshold_ablation.py --package … --labels …/final_human_reviewed/completed.jsonl --rows … --records … --out audit/threshold_ablation.json` | `audit/threshold_ablation.json` |
| §8 Table, Fig. goldspan | gold-span coverage, lexical | `python scripts/audit_gold_span_coverage.py --package … --out audit/gold_span_coverage.json` | `audit/gold_span_coverage.json` |
| §8 Table | gold-span coverage, lexical + semantic | `python scripts/audit_gold_span_semantic.py --package … --out audit/gold_span_semantic.json` | `audit/gold_span_semantic.json` |
| §4 method | annotation package build | `python scripts/build_annotation_package.py --records … --out reports/annotation/qasper_dev_300_full_context --n-units 200` | package + `manifest.json` |
| Figures | all four paper figures | `pip install -r requirements-research.txt && python scripts/make_paper_figures.py --all` | `results/figures/*.png` |
| Figures | earlier study figures | `python scripts/make_figures.py --all` | `results/figures/*.png` |
| Tests | full suite | `pytest tests/ -q` | 477 tests |
| Lint | ruff | `ruff check scripts/ src/ tests/` | clean |

## Provenance recorded in every report

UTC timestamp, git commit and dirty flag, raw-file SHA-256, split, sample size, chunk
size, top-k, embedder and generator identity, taxonomy version and threshold
fingerprint, Python version, platform, package versions.

## Known reproducibility limits

- **Annotation data is not distributed.** `reports/` is gitignored, so §7 and §8
  cannot be re-derived from a fresh clone; the labels are annotation data, not a
  computation. Tracking the 7.5 MB package would fix this and is an open decision.
- **Approximate nearest-neighbour search.** Fine-grained aggregates on long-document
  corpora move by ≤0.001 between independently built indices. Headline A/B/C figures
  reproduce exactly; no reported gap or test changes.
- **Local model downloads.** `qwen0.5b` pulls from Hugging Face on first use.
