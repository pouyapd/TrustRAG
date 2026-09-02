# Reproducibility

Exact commands for every number in `results.md`. Run from the repository root.
Corpora are not redistributed — download commands and checksums are in
`docs/DATASETS.md`.

```bash
pip install -r requirements.txt
```

---

## 1. The source run (retrieval + extractive generation)

```bash
python scripts/run_experiment.py --dataset qasper \
    --raw data/raw/qasper-dev-v0.3.json --split dev \
    --limit 300 --embedder minilm \
    --out reports/experiments/qasper_dev_300
```

Writes `inference.jsonl` (raw records), `rows.jsonl` (scored rows, both taxonomy
variants), `summary.json`, `report.md`. No API key: the embedder runs locally and
the generator is a deterministic extractive control.

## 2. Retrieval decomposition and robustness

```bash
python scripts/reproduce_study.py --all            # A/B/C on 5 experiments
python scripts/reproduce_study.py --embedder-sweep # 4 models
python scripts/reproduce_study.py --topk-sweep     # k = 1,3,5,10,20
python scripts/reproduce_study.py --multihop       # 2WikiMultihopQA
```

## 3. Annotation package construction

```bash
python scripts/build_annotation_package.py \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --out reports/annotation/qasper_dev_300_full_context --n-units 200
```

Deterministic under the recorded seed (`20260826`). The build aborts if any
retrieved chunk or gold span holds less text than its `char_range` covers;
`manifest.json` records `context_integrity` counts for both.

## 4. Annotation and validation

```bash
# serve the offline annotation UI (one unit at a time, withheld key unreadable)
python scripts/annotate.py --annotator a \
    --package reports/annotation/qasper_dev_300_full_context

# validate a completed file: rows, ids, labels, confidences, sheet checksum,
# unit-content preservation, retrieved-context completeness
python scripts/annotate.py --annotator a \
    --package reports/annotation/qasper_dev_300_full_context --validate
```

Expected on the committed reference set: seven `ok` lines including
`retrieved context is complete for all 1000 chunk(s)`, exit code 0.

## 5. Truncation audit

```bash
python scripts/audit_annotation_truncation.py \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --old-package reports/annotation/qasper_dev_300 \
    --new-package reports/annotation/qasper_dev_300_full_context \
    --out reports/annotation/qasper_dev_300_full_context/TRUNCATION_AUDIT.json
```

Reproduces `results.md` §2: 1000 chunks, 941 cut at 600, 941 recovered,
1000/1000 complete, 0 unreconstructable, 588,671 → 1,163,638 characters.

## 6. Two-annotator agreement and adjudication (truncated package)

```bash
python scripts/score_annotations.py \
    --package reports/annotation/qasper_dev_300 \
    --annotator a=reports/annotation/qasper_dev_300/annotator_a/completed.jsonl \
    --annotator b=reports/annotation/qasper_dev_300/annotator_b/completed.jsonl \
    --adjudicated reports/annotation/qasper_dev_300/final_adjudicated_labels.jsonl \
    --out reports/annotation/qasper_dev_300/final_agreement_report.json
```

Reproduces `results.md` §4 row 1: kappa 0.8365, observed 0.925, 200 resolved.

## 7. The headline evaluation

```bash
python scripts/score_against_reference.py \
    --package reports/annotation/qasper_dev_300_full_context \
    --reference reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl \
    --rows reports/experiments/qasper_dev_300/rows.jsonl \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --compare truncated_pass_a=reports/annotation/qasper_dev_300/annotator_a/completed.jsonl \
    --compare truncated_pass_b=reports/annotation/qasper_dev_300/annotator_b/completed.jsonl \
    --compare truncated_adjudicated=reports/annotation/qasper_dev_300/final_adjudicated_labels.jsonl \
    --out reports/annotation/qasper_dev_300_full_context/final_evaluation.json
```

Reproduces `results.md` §1 and §4 in one file: document-gated accuracy 0.7400 /
macro F1 0.6223 / kappa 0.5728; evidence-gated 0.8050 / 0.6295 / 0.6305; paired
McNemar p = 0.0294 (22 vs 9 discordant); agreement with the three earlier passes.

`--rows` / `--records` are what add the evidence-gated variant: the labels are
read from the stored run, not recomputed, so the comparison costs no model calls.

## 8. Threshold re-scoring (no model calls)

```bash
python scripts/reclassify.py \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --out reports/sweep --sweep-faithfulness 0.3,0.6,0.9
```

## 9. Figures

```bash
pip install -r requirements-research.txt

# retrieval-study figures (A/B/C, embedders, top-k, attribution)
python scripts/make_figures.py --all

# the pipeline + evaluation figure used in the README, drawn from
# final_evaluation.json and TRUNCATION_AUDIT.json
python scripts/make_pipeline_figure.py \
    --package reports/annotation/qasper_dev_300_full_context \
    --out docs/figures/pipeline_evaluation.png
```

## 10. Tests

```bash
pytest tests/ -q                 # 466 tests
pytest tests/ -q --cov=src       # 80% line coverage
ruff check scripts/ src/ tests/
```

Relevant to the annotation pipeline specifically:

```bash
pytest tests/test_annotation_package_no_truncation.py \
       tests/test_annotation_tool.py \
       tests/test_annotation_validation.py \
       tests/test_taxonomy.py -q     # 114 tests
```

`tests/test_annotation_package_no_truncation.py` fails if a fixed-size slice is
reintroduced into the package builder — the regression guard for the defect in
`results.md` §2.

## 11. Provenance

Every report embeds `collect_provenance(...)`: UTC timestamp, git commit and
dirty flag, Python version, platform, and versions of chromadb, numpy, openai,
anthropic, sentence-transformers, tiktoken, pydantic and fastapi. The annotation
package additionally records the sampling seed, per-mode sampling weights, the
guidelines path and the `context_integrity` counts; each completed file records
the sheet's SHA-256 in `.sheet_integrity.json`.

## 12. Artifact availability — the one gap in this file

`results/` is tracked by git; **`reports/` is listed in `.gitignore`**. Every
command above regenerates its own inputs *except* one: the reference labels in
`reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl` are
annotation data, not a computation, so a fresh clone cannot reproduce §7 — the
headline evaluation — from source. §1–§5 and §8–§10 reproduce from the corpora
alone.

Nothing in the repository currently distributes that file. The fix is to track
the annotation package (7.5 MB; 15 MB with the earlier truncated package, against
131 MB for all of `reports/`), or to publish it separately and record the
checksum. This is a repository-policy decision and has not been made here; it is
stated so that no reader assumes §7 runs after a clone.

## 13. Determinism

Retrieval and evidence measurements reproduce exactly on re-run. Fine-grained
aggregates on long-document corpora (chunk precision/recall, nDCG, faithfulness
means) move by ≤0.001 because approximate nearest-neighbour search can rank one
borderline question differently between two independently built indices. No
reported gap, significance test or conclusion changes.
