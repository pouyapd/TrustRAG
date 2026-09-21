# docs/paper — supporting documents for the manuscript

The manuscript *Right document, wrong passage: evidence localisation in
retrieval-augmented generation evaluation on QASPER and Natural Questions* is prepared
for submission to *Language Resources and Evaluation* (not yet submitted, not accepted,
not published). The manuscript itself is not in this repository. The documents in this
folder record, with their source files, the measurements it draws on. They were written
at different stages of the project and are kept as written; the stage of each is stated
here so that nothing is read as more current than it is.

| Document | Stage | What it holds |
|---|---|---|
| [human_validation_final.md](human_validation_final.md) | 3 — current | The 200-unit human annotation study: provenance of every label, the guided review of 43 flagged units, both retrieval gates against the final human labels, the truncation threat, what the study does and does not establish |
| [literature_review.md](literature_review.md) | 3 — current | Novelty audit and comparison table against prior work on evidence-aware RAG evaluation |
| [reproducibility.md](reproducibility.md) | 3 — current | Analysis → command → output map, environment, known limits |
| [results.md](results.md) | 2 | Failure taxonomy scored against the *automated* reference set; context-integrity audit; effect of restoring full context; retrieval decomposition; generation replay |
| [TABLES.md](TABLES.md), [FIGURES.md](FIGURES.md) | 2 | Stage-2 tables and figures with the source file and command for each |
| [experimental_setup.md](experimental_setup.md) | 2 | Corpus, retrieval and generation configuration of the annotated run; annotation package; reference set |
| [limitations.md](limitations.md) | 2 | Limitations as they stood before the human study; see the status of its "what is missing" table below |

**Stages.** Stage 1 (August 2026): the A/B/C retrieval decomposition on four corpora
([docs/EXPERIMENTS.md](../EXPERIMENTS.md), `results/*.json`). Stage 2 (early September
2026): the failure taxonomy scored against an automated (language-model) reference set.
Stage 3 (September 2026): the human annotation study and guided review, the gold-span
completeness adjudication, the within-document localisation study on QASPER and Natural
Questions ([results/localisation/README.md](../../results/localisation/README.md)) and
the oracle-evidence replication. The root [README](../../README.md) summarises stage 3,
which is what the manuscript reports.

**Reading the stage-2 documents.** What they call the "reference set" is the
language-model annotation pass, not human annotation. The human-reviewed labels in
`human_validation_final.md` supersede it: agreement with the automated pass (accuracy
0.805, κ 0.630) is reported only as context, and the human number (accuracy 0.700
against 0.600 for the document gate) is the one the manuscript reports.

**Status of the "what is missing" table in `limitations.md`** (written 2 September 2026):

| # | Item | Status |
|---|---|---|
| 1 | Human annotation pass over the 200 full-context units | Done — `human_validation_final.md` (one annotator, the author; guided review, not independent validation) |
| 2 | Second independent pass, inter-annotator agreement | Open — stated as a limitation |
| 3 | Related-work section | Done — `literature_review.md` and the manuscript |
| 4 | Second corpus for the taxonomy validation | Open — the human study is QASPER only; the localisation study, not the taxonomy validation, covers NQ |
| 5 | Real generative model behind the annotated package | Open — the annotated run uses the extractive control; the oracle-evidence experiment uses Qwen2.5-0.5B on 150 questions but was not annotated |
| 6 | Threshold-tuning ablation, held-out | Done — `scripts/threshold_ablation.py`, 144 configurations, 50/50 split, seed 20260906; the output rows are in the annotation package |
| 7 | Oracle-context ablation | Done — `scripts/run_oracle_evidence.py`, reported in the root README as a replication |
| 8 | Distribution of the annotation package | Decided — deposited as a separate checksummed archive in the Zenodo record of release v1.0.0 (DOI inserted once the record exists) |
