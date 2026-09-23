# Changelog

## v1.1.0 — 22 September 2026 — completed gold-span adjudication

Tag `v1.1.0` is the code and result state the manuscript reports. It is a GitHub tag, not a
new Zenodo deposit: everything it adds is tracked in the repository, so no archive is needed.
The Zenodo record is unchanged — concept DOI 10.5281/zenodo.22879357, code archive
10.5281/zenodo.22879358 (v1.0.0), annotation package 10.5281/zenodo.22879569 — and the earlier
versions remain published.

- **Gold-span completeness audit completed.** The first pass had adjudicated a stratified sample
  of 60 of the 87 proxy-unresolved zero-coverage units and counted the 10 units both proxies
  called "answer present" as under-coverage without anyone reading them. The author has now
  adjudicated the remaining 37 units — the 27 unresolved units the sample had not drawn and
  those 10 — under the same protocol, the same answers and the same blind sheet: **3 YES, 34 NO,
  0 CANNOT_TELL**. The 87 unresolved units are therefore a census, the finite population
  correction is zero and the sampling error disappears; the proxy-resolved stratum is now
  observed rather than assumed (3 of 10 confirmed).
  - Under-coverage over the 133 zero-coverage units: **0.053** (7 units, counted) in place of
    0.119 with a 95% CI of [0.096, 0.142]; the range once the 36 unadjudicated "answer absent"
    units are allowed for is about **5–7%**, in place of the earlier 4–12%.
  - The same person judged both rounds. This adds coverage, not agreement: there is still one
    annotator and no inter-annotator agreement, and 36 of the 133 units remain unread.
  - New: `scripts/build_goldspan_remainder.py` (builds the second package),
    `scripts/render_goldspan_review.py` (renders a package for reading, locating each gold span
    in its paper), `tests/test_goldspan_remainder.py` (13 tests).
    `scripts/score_goldspan_adjudication.py` gained `--remainder`; without it the original
    estimate is reproduced unchanged.
  - The first-pass package (`reports/annotation/goldspan_adjudication/`) is untouched.
  - New: [`results/goldspan_adjudication/`](results/goldspan_adjudication/) tracks the proxy
    partition of the 133 units, both rounds of labels and both estimates. These files carry no
    QASPER text, so Table 9 now recomputes from a clone with one command. The adjudication
    sheets, which do quote QASPER, stay out of the repository as before.
  - Reference identifiers in the manuscript build are now clickable: apacite hyperlinks a
    reference's `doi` field but printed the `url` field as plain text, which left 12 of 16
    entries unlinked.
- **Tests**: 504 pass (`pytest tests/ -q`); `ruff check src/ tests/ scripts/` is clean.

## v1.0.0 — 21 September 2026 — research and reproducibility release

This is the software and artefact release corresponding to the manuscript *Right Document, Wrong
Passage: Evidence Localisation in Retrieval-Augmented Generation Evaluation on QASPER and Natural
Questions* (Pouya Bathaei Pourmand, 2026), prepared for submission to *Language Resources and
Evaluation*. The manuscript has not been submitted, accepted or published at the time of this
release, and the release makes no claim to the contrary. The tag `v1.0.0` fixes the code, result
files and documentation the manuscript reports. The release is archived on Zenodo through the
GitHub integration: concept DOI 10.5281/zenodo.22879357, version DOI 10.5281/zenodo.22879358
(code archive, 201 files).

### What the release contains

- **Corpus loaders and chunking.** QASPER (dev), Natural Questions (validation), HotpotQA and
  2WikiMultihopQA loaders (`src/data/loaders/`); corpora are fetched from their original sources
  and are not redistributed (`docs/DATASETS.md` gives commands, checksums and licences). The
  256-token chunker carries character offsets, and the offset identity
  `document[start:end] == chunk.text` is property-tested.
- **Within-document evidence-localisation protocol** with its analytic, length-aware chance level
  (`scripts/localisation_probe.py`), the seven-localiser comparison — BM25, all-MiniLM-L6-v2,
  bge-small-en-v1.5, e5-small-v2, all-mpnet-base-v2, ms-marco-MiniLM-L-6-v2, bge-reranker-base —
  with Holm-corrected exact McNemar tests, sign tests, Wilson intervals, Spearman and union
  analyses (`localisation_report.py`, `localisation_analysis.py`), the document-level cluster
  bootstrap, chunk audit, model metadata, hit@k recomputed from stored ranks and
  reach × localisation Fisher tests (`localisation_robustness.py`), admission-budget and chunk-size
  sweeps (`localisation_admission.py`) and the corpus contrasts (`localisation_contrast.py`).
- **All per-question localisation rows and summaries** under `results/localisation/` (29 files;
  the file table and reproduction commands are in its README), including the new
  `hit_at_k_qasper.json`, `hit_at_k_nq.json` and `reach_association_qasper.json`, so that every
  deeper cut-off and every reach test quoted in the manuscript traces to a committed file.
- **A/B/C retrieval decomposition** on four corpora, embedder/depth/chunk-size sweeps, the BM25
  baseline scored under the same definitions, and the withdrawn-inversion record
  (`scripts/reproduce_study.py`, `run_bm25_baseline.py`, `results/*.json`, `docs/EXPERIMENTS.md`).
- **Annotation tooling** for the human-reviewed attribution study: blinded stratified package
  build, offline annotation server, validation, guideline audit, review subset, final dataset with
  a per-unit provenance chain, scoring, truncation audit and held-out threshold ablation
  (`build_annotation_package.py`, `annotate.py`, `audit_human_annotations.py`,
  `build_review_subset.py`, `build_final_human_dataset.py`, `score_annotations.py`,
  `audit_annotation_truncation.py`, `threshold_ablation.py`).
- **Gold-span completeness audit**: lexical and semantic proxies, the blind stratified
  adjudication sheet and the stratified estimator (`audit_gold_span_coverage.py`,
  `audit_gold_span_semantic.py`, `build_goldspan_adjudication.py`,
  `score_goldspan_adjudication.py`).
- **Oracle-evidence experiment** (`run_oracle_evidence.py`) and the failure taxonomy with its two
  retrieval gates (`src/evaluation/taxonomy.py`, `docs/TAXONOMY.md`).
- **Documentation**: the root README (project, paper, findings, reproducibility, limitations),
  `results/localisation/README.md`, `docs/paper/` with a stage index (`docs/paper/README.md`),
  `docs/paper/human_validation_final.md`, `docs/paper/reproducibility.md`.
- **Metadata**: `LICENSE` (MIT), `CITATION.cff`, `.zenodo.json`, `pyproject.toml` version 1.0.0.
- **Tests**: 488 tests pass (`pytest tests/ -q`); `ruff check src/ tests/ scripts/` is clean.

### Not in the git tree

The raw corpora, model weights (pulled from Hugging Face on first use; the revisions actually run
are listed in `results/localisation/model_metadata.json`), and `reports/` (run records and the
annotation package). The human annotation package — 200 QASPER units with full retrieved context,
original/review/final labels with provenance, guideline and truncation audits, the gold-span
adjudication sheet and answers, threshold-ablation and oracle-evidence rows; 68 files, SHA-256
`d6d84ecb155853939be2b6d482caedb808915e29aa54498fb78c7d09502ea44d`) is deposited as the second version of the
Zenodo record of this release (version DOI 10.5281/zenodo.22879569), under QASPER's CC BY 4.0
terms for the excerpted text.

### Changes since the previous commit on `main` (9a7ab20)

- Added `LICENSE`, `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`; version 1.0.0 in
  `pyproject.toml` and `src/__init__.py`; `scipy` listed explicitly in `requirements.txt`.
- `scripts/localisation_robustness.py`: `--hit-at-k` and `--reach-association` modes;
  `scripts/localisation_contrast.py`: the corpus-contrast computation as a script (reproduces
  `results/localisation/corpus_contrast.json` exactly; the file now also records that one NQ
  question is excluded because its gold document has no non-gold chunk).
- Documentation corrections found by the pre-submission audit, none of which changes a stored
  result: four table cells that had been re-rounded from four-decimal summaries (MPNet hit@1 0.334,
  BM25-with-document-local-IDF hit@5 0.734, bge-reranker-base MRR 0.514, one Wilson interval
  [0.404, 0.517]); κ 0.630 for agreement with the automated reference pass; and the provenance of
  the 22 pilot-era labels in the human study (none of the 22 was among the 43 reviewed units, so
  all 22 keep their pilot label: 135 original + 43 review decisions + 22 pilot-era).
- `docs/paper/`: stage index added; stale internal planning notes removed (`venue_fit.md`,
  `paper_outline.md`, `reviewer_simulation.md`, `RESEARCH_SUMMARY.md`); stage banners on the
  stage-2 documents; `reproducibility.md` updated.
- Root README rewritten around the manuscript; author affiliation is "independent researcher,
  Genoa, Italy".

### Licence

MIT for the code and result files. QASPER excerpts in the annotation archive: CC BY 4.0.
