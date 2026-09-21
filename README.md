# TrustRAG — evidence localisation and failure attribution for RAG evaluation

[![CI](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release v1.0.0](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/pouyapd/TrustRAG/releases/tag/v1.0.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22879357.svg)](https://doi.org/10.5281/zenodo.22879357)

**TrustRAG is a research framework for evaluating retrieval-augmented generation (RAG) at the
level of evidence rather than documents.** A document-level retrieval metric counts any retrieved
chunk of a relevant document as a success. TrustRAG measures what that metric cannot see: the gap
between retrieving the correct document and localising the passage that actually supports the
answer — the passage a generator needs, and the passage an evaluation decision about *which stage
failed* depends on. It does this with an offset-carrying pipeline (every chunk keeps its character
range in the source document, so evidence coverage is interval arithmetic, not string search), a
document-restricted localisation protocol with an analytic chance level, a failure taxonomy whose
retrieval rule can be gated on document-level or span-level evidence, and an audit of the evidence
annotations that span-level scoring relies on.

TrustRAG proposes no retrieval method and claims no state-of-the-art result. It is a measurement
study: what happens inside the right document, on two public corpora, with the reference itself
audited.

## Research paper

**Right Document, Wrong Passage: Evidence Localisation in Retrieval-Augmented Generation Evaluation
on QASPER and Natural Questions.** Pouya Bathaei Pourmand, 2026.

*Status: manuscript prepared for submission to* Language Resources and Evaluation *(Springer).* It
has not been submitted, accepted or published; this line will be updated when that changes. The
manuscript is not in the repository.

Release **[v1.0.0](https://github.com/pouyapd/TrustRAG/releases/tag/v1.0.0)** of this repository
is the code and result state the manuscript reports. It is archived on Zenodo: concept DOI
[10.5281/zenodo.22879357](https://doi.org/10.5281/zenodo.22879357) (always the latest version),
version DOI [10.5281/zenodo.22879358](https://doi.org/10.5281/zenodo.22879358) for the v1.0.0
code archive. The human annotation package (SHA-256 `d6d84ecb155853939be2b6d482caedb808915e29aa54498fb78c7d09502ea44d`)
is deposited as a separate archive in a new version of the same record; until that version is
published the record holds the code archive only (see [Reproducibility](#reproducibility)).

**In one paragraph.** Every chunk of the document that holds the gold evidence is ranked against
the question by seven localisers — BM25, four dense bi-encoders and two cross-encoder rerankers,
22M–278M parameters — on QASPER (290 questions, NLP papers) and Natural Questions (300 questions,
Wikipedia), under one protocol and a length-aware analytic chance level. Rank-1 localisation stays
at or below about half the questions for every model and corpus (hit@1 0.29–0.51) while hit@5
reaches 0.69–0.86; the median rank is 2 for almost every model, and the models miss different
questions. BM25 is the weakest localiser on both corpora, differences among the bi-encoders are
not significant after Holm correction, and the cross-encoder reranker effect does not replicate
across corpora. An author-conducted annotation study of 200 QASPER units finds that gating failure
attribution on span coverage agrees with the human labels more often than gating on document
coverage (accuracy 0.700 against 0.600), and a stratified, blind adjudication of 60 units estimates
the incompleteness of the QASPER span annotations at 0.119 (95% CI [0.096, 0.142]; 4–12% under
sensitivity analysis). The results argue for reporting localisation against chance and for auditing
evidence annotations before they serve as a scoring reference.

## Contents

[The problem](#the-problem-right-document-wrong-passage) · [What is evaluated](#what-the-framework-evaluates) ·
[Main findings](#main-findings) · [Human-reviewed attribution study](#human-reviewed-attribution-study) ·
[Gold-span completeness audit](#gold-span-completeness-audit) · [Oracle-evidence control](#oracle-evidence-control-a-replication) ·
[Decomposition and a withdrawn result](#retrieval-decomposition-and-a-withdrawn-result) ·
[Reproducibility](#reproducibility) · [Repository structure](#repository-structure) · [Installation](#installation) ·
[Tests](#tests) · [Limitations](#limitations) · [Version](#version-and-release) · [Citation](#citation) · [License](#license)

## The problem: right document, wrong passage

A gold span and a retrieved chunk are half-open character intervals in the same document.
Positive overlap is evidence coverage:

```
document  qasper:1901.00001
gold span        [1200, 1760)
retrieved chunk  [ 900, 2100)   overlap = 560 chars  ->  evidence covered
retrieved chunk  [8300, 9500)   overlap =   0 chars  ->  same document, no evidence
```

Both chunks satisfy a document-level metric; only the first makes the question answerable from
context. Three definitions of retrieval success follow — **A**: some chunk from a relevant document
(the conventional metric); **B**: every document a multi-hop question requires; **C**: a retrieved
chunk contains the gold span — with `C ≤ B ≤ A` by construction. The A→C gap is large on
long-document corpora (16.6 pp on QASPER, 26.7 pp on NQ at *k* = 5; see
[below](#retrieval-decomposition-and-a-withdrawn-result)). That the gap exists is
[prior work](docs/paper/literature_review.md). TrustRAG asks what remains once the right document
has been reached: does the retriever rank the evidence passage first inside it, does that depend on
the model or the corpus, and how far can the span annotations used to score it be trusted?

## What the framework evaluates

**Within-document localisation protocol.** For every answerable question, *all* chunks of the gold
document (256 tokens, overlap 32, character offsets verified `document[start:end] == chunk.text`)
are ranked against the question by a localiser; the rank of the first chunk overlapping a gold span
is recorded. No other document competes, so the measurement is about localisation only. The
expected hit@*k* and MRR under a random ordering of that document's *D* chunks with *m* gold chunks
are computed analytically, which makes corpora with different document lengths comparable and gives
a lift over chance for every model. Statistics: exact McNemar on paired hit@1 with Holm correction
over all 21 model pairs per corpus, exact sign tests on reciprocal ranks, Wilson intervals,
Spearman rank correlation, union analysis, and a document-level cluster bootstrap because QASPER
questions cluster in papers. Also measured: hit@*k* for *k* ∈ {1, 3, 5, 10, 20}, near-miss mass at
ranks 2–5, the trade-off between admitting more chunks of the reached document and reach, chunk
size (128/256/512), and descriptive corpus contrasts (offered as hypotheses, not tested causes).

**Seven localisers, identical protocol on both corpora.**

| Localiser | Family | Parameters |
|---|---|---|
| BM25 (Okapi, k1 = 1.5, b = 0.75, corpus-level IDF) | lexical | — |
| `sentence-transformers/all-MiniLM-L6-v2` | dense bi-encoder | 22M |
| `BAAI/bge-small-en-v1.5` | dense bi-encoder | 33M |
| `intfloat/e5-small-v2` | dense bi-encoder | 33M |
| `sentence-transformers/all-mpnet-base-v2` | dense bi-encoder | 110M |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder reranker | 22M |
| `BAAI/bge-reranker-base` | cross-encoder reranker | 278M |

Model revisions actually run, input limits and documented training data:
[results/localisation/model_metadata.json](results/localisation/model_metadata.json).

**Datasets.** Corpora are not redistributed; loaders, download commands, checksums and licence
metadata are committed ([docs/DATASETS.md](docs/DATASETS.md)).

| Dataset | Licence | Structure | Used for |
|---|---|---|---|
| QASPER (dev) | CC BY 4.0 | NLP papers, ~20 chunks per gold document, evidence paragraphs (2.09 spans per question) | localisation, human study, gold-span audit, oracle control, decomposition |
| Natural Questions (validation) | CC BY-SA 3.0 | Wikipedia pages, ~42 chunks per gold document, one long answer | localisation, decomposition |
| HotpotQA, 2WikiMultihopQA | CC BY-SA 4.0, Apache-2.0 | 10 paragraphs, 2–4 gold | the A→B quantifier effect in the decomposition only |

**Failure taxonomy with two retrieval gates.** Nine categories, one rule (R4) that decides whether
a row is charged to retrieval. The document-gated variant binds R4 to definition A, the
evidence-gated variant to definition C; everything else is identical and both labels are written to
every row, so the comparison runs on the same retrieval output at no extra inference cost
([docs/TAXONOMY.md](docs/TAXONOMY.md)).

## Main findings

![Rank of the gold chunk inside its document, per model and corpus](results/localisation/rank_distribution.png)

| | QASPER (n = 290) | NQ (n = 300) |
|---|---|---|
| Chance hit@1 (random order of the document's chunks) | 0.157 | 0.116 |
| hit@1, seven localisers | 0.290 – 0.400 | 0.353 – 0.507 |
| hit@5, seven localisers | 0.755 – 0.855 | 0.690 – 0.837 |
| Best single localiser at rank 1 | BGE-small 33M, 0.400 | bge-reranker-base 278M, 0.507 |
| Weakest neural localiser at rank 1 | bge-reranker-base 278M, 0.310 | MiniLM 22M, 0.387 |
| Some localiser ranks the gold chunk first | 0.769 | 0.780 |
| Pairs significant after Holm (of 21) | 2 (both BM25 deficits) | 6 |

Per localiser, hit@1 / hit@5 (from [`hit_at_k_qasper.json`](results/localisation/hit_at_k_qasper.json)
and [`hit_at_k_nq.json`](results/localisation/hit_at_k_nq.json), recomputed from the stored ranks):

| Localiser | QASPER hit@1 | QASPER hit@5 | NQ hit@1 | NQ hit@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.290 | 0.755 | 0.353 | 0.690 |
| MiniLM-L6 (22M) | 0.376 | 0.845 | 0.387 | 0.773 |
| BGE-small (33M) | 0.400 | 0.845 | 0.463 | 0.817 |
| E5-small (33M) | 0.359 | 0.821 | 0.457 | 0.803 |
| MPNet-base (110M) | 0.334 | 0.797 | 0.403 | 0.793 |
| ms-marco-MiniLM cross-encoder (22M) | 0.386 | 0.855 | 0.460 | 0.813 |
| bge-reranker-base (278M) | 0.310 | 0.797 | 0.507 | 0.837 |

**What replicates on both corpora.** Every model places the evidence near the top of its document
but rarely first: the median rank is 2 for almost every model, a third to a half of gold chunks sit
at ranks 2–5, and the misses are only partly shared (union of seven at rank 1 0.77–0.78 against a
best single model of 0.40–0.51). BM25 is the weakest localiser on both corpora and the only model
whose deficit survives Holm correction on both; it localises at 0.84–0.89 when the gold chunk is
the lexically most question-like chunk of its document and at 0.04–0.06 otherwise. The 110M
bi-encoder trails the 33M ones on both corpora, though no difference among the bi-encoders is
significant after correction. No association between reaching the document and localising inside
it was detected on QASPER (Fisher exact *p* 0.31–0.63; power is limited), and the pairwise
conclusions hold under the document-level cluster bootstrap.

**What does not replicate.** The 278M cross-encoder reranker is the best localiser on NQ
(+4 to +12 pp hit@1 over the bi-encoders, significantly above MiniLM and MPNet after correction,
though its lead over BGE-small is not significant, *p* = 0.19) and the worst neural localiser on
QASPER, below every bi-encoder and indistinguishable from BM25. The corpora differ in ways
consistent with this — QASPER papers are more topically homogeneous chunk-to-chunk (mean
intra-document cosine 0.58 vs 0.46), their gold evidence shares no content word with the question
about four times as often (8.3% vs 2.0%), and web QA is in the documented training data of every
neural model tested while scientific-paper QA is in none — but these contrasts are descriptive;
they suggest hypotheses and do not establish causes.

**The claim that survives both corpora.** Increasing retriever or reranker capacity does not
produce a consistent improvement in rank-1 evidence localisation, and the localisation term is
closed only by admitting more of the reached document (hit@1 0.29–0.40 → hit@5 0.76–0.86 →
hit@10 0.93–0.98 on QASPER), at the price of reach; no fixed document-first allocation beats flat
top-*k* at equal budget. Conditional span coverage at a fixed cut-off is therefore not a measure of
ranking skill: two retrievers can cover the span at similar rates while ranking it very
differently inside the document, if one admits more chunks from it. Full tables, pairwise tests,
correlations, sweeps, corpus contrasts and limitations:
[results/localisation/README.md](results/localisation/README.md).

## Human-reviewed attribution study

![Both retrieval gates scored against the final human-reviewed labels](results/figures/human_validation.png)

200 QASPER units were annotated by the author under written guidelines
([docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md)) in a blinded package built by
[`scripts/build_annotation_package.py`](scripts/build_annotation_package.py). Provenance is
stated exactly because it matters: an audit against the guidelines flagged 43 of the 200 labels as
conflicting with an explicit rule; the annotator re-reviewed those 43 on full context (36 changed,
7 upheld); 22 labels come from a pilot on a truncated display and were never re-judged (none of the
22 was among the 43). The final 200 labels are therefore 135 original full-context labels, 43
review decisions and 22 pilot-era labels, with a per-unit chain `original → flag reason → review →
final`.

| Variant | Accuracy | 95% CI | Macro F1 | κ |
|---|---:|---:|---:|---:|
| Document-gated | 0.600 | 0.531–0.665 | 0.476 | 0.375 |
| **Evidence-gated** | **0.700** | 0.633–0.759 | **0.482** | **0.437** |

Paired over the same units: 118 both correct, **22 only evidence-gated**, 2 only document-gated,
58 neither; exact McNemar *p* = 3.6 × 10⁻⁵. Per class (evidence-gated): `wrong_retrieval` F1 0.907
(support 136), `answered_when_unanswerable` 1.000 (9), `partial_answer` 0.286 (22), `ok` 0.158
(32), `incorrect_answer` 0.059 (1). Retrieval-side attribution is where the gate helps;
generation-side classes are not reliable, and a held-out threshold ablation (144 configurations,
50/50 split, [`scripts/threshold_ablation.py`](scripts/threshold_ablation.py)) moves the evidence
gate only from 0.730 to 0.750 on the held-out half — the rules, not the thresholds, are the larger
problem.

**This is agreement with a guided expert reading, not independent validation.** There is one
annotator (the author), no inter-annotator agreement, and the second pass was directed by an audit
of the same guidelines under test. Agreement with an earlier *automated* (language-model) reference
pass is higher (0.805, κ 0.630) than with the human labels; the human number is the one reported.
Full account: [docs/paper/human_validation_final.md](docs/paper/human_validation_final.md).

## Gold-span completeness audit

![Proxy partition of the 133 zero-coverage units, before human adjudication](results/figures/gold_span_validity.png)

Span-level scoring assumes the annotated spans are the only places the answer can be derived from.
QASPER marks supporting paragraphs, not every sufficient passage, so a retriever that returns a
different but sufficient passage is scored as a failure. The audit asks, for the 133 answerable
units with zero gold-span coverage, whether the reference answer is derivable from the retrieved
text alone. Two proxies (content-word overlap and MiniLM cosine) resolved 46 units where they agreed
(36 answer absent, 10 answer present); the 87 they could not resolve were stratified, and 60 were
sampled (seed 20260907) and adjudicated by hand, blind to every previous label and score
([`scripts/build_goldspan_adjudication.py`](scripts/build_goldspan_adjudication.py),
[`scripts/score_goldspan_adjudication.py`](scripts/score_goldspan_adjudication.py)):
**4 YES, 56 NO, 0 CANNOT_TELL**.

| | Estimate |
|---|---|
| Gold-span under-coverage among the 133 zero-coverage units | **0.119**, 95% CI **[0.096, 0.142]** |
| Defensible range under sensitivity analysis | **4% – 12%** |

The interval covers sampling error only, and the point estimate leans on the 10 units both proxies
called "answer present" that no human checked; the adjudicated sample agreed with "answer present"
on only 6.7% of unresolved units, and if those 10 behave like the adjudicated ones the rate is
0.049. The estimate is not definitive: it is one annotator, one corpus, and a proxy-resolved census
part. What it supports is that the span reference is a substantially sound instrument for
retrieval attribution — the bias runs in the expected direction and is an order of magnitude
smaller than the effects measured — and that annotation completeness can be measured at low cost
before annotations are used as a scoring reference.

## Oracle-evidence control (a replication)

![Oracle-evidence control](results/figures/oracle_evidence.png)

150 QASPER questions answered twice by the same generator (Qwen2.5-0.5B-Instruct), same prompt,
same decoding; only the context differs (retrieved chunks vs the gold spans verbatim).

| Stratum | n | retrieved | oracle | difference | p |
|---|---:|---:|---:|---:|---:|
| Evidence complete under retrieval | 46 | 0.065 | 0.174 | +10.9 pp | 0.125 (7 discordant; indicative) |
| **Document retrieved, span missing** | 26 | **0.000** | **0.231** | **+23.1 pp** | 0.031 (6 discordant; indicative) |
| Nothing from any gold document | 78 | 0.000 | 0.321 | +32.1 pp | 6.0 × 10⁻⁸ |

The middle row is the argument: those 26 questions count as retrieval *successes* under a
document-level metric, the model got none right, and supplying the actual span repairs 23% of them.
The experiment replicates [arXiv:2608.08944](https://doi.org/10.48550/arXiv.2608.08944) (32.8%
repair over 11,105 failures) at n = 150 with one small reader and no sham control; the two smaller
strata have too few discordant pairs for their *p*-values to carry weight.

## Retrieval decomposition and a withdrawn result

The A/B/C decomposition on four corpora (dense retrieval, `all-MiniLM-L6-v2`, *k* = 5, 256-token
chunks) isolates a *quantifier* effect (A→B, multi-hop completeness) from a *granularity* effect
(B→C, span coverage); the two are near-orthogonal, each null on the corpus where the other
dominates. Full protocol and threats to validity: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| Corpus | n | A | B | C | quantifier A→B | granularity B→C |
|---|---:|---:|---:|---:|---:|---:|
| QASPER dev | 290 | 0.441 | 0.441 | 0.276 | 0.0 pp | **16.6 pp** (p = 7.1e-15) |
| Natural Questions | 300 | 0.997 | 0.997 | 0.730 | 0.0 pp | **26.7 pp** (p = 1.7e-24) |
| HotpotQA | 150 | 0.993 | 0.507 | 0.507 | **48.7 pp** (p = 2.1e-22) | 0.0 pp |
| 2WikiMultihopQA | 150 | — | — | — | **64.7 pp** (p = 1.3e-29) | 1.3 pp (n.s.) |

**A result we reported and then withdrew.** An earlier version of this repository claimed that the
document/span choice reverses the BM25-vs-dense comparison. It does not: the finding was an
evidence-mode defect in our own BM25 baseline (QASPER declares `any_sufficient` evidence, the
baseline hard-coded `all_required`, and 51% of QASPER questions carry more than one span, so BM25's
span coverage was under-reported at 0.183 instead of 0.321). Corrected, **no inversion occurs on
any corpus**: BM25 leads at both granularities on QASPER (0.528/0.321 vs 0.441/0.276; paired 40
vs 27, p = 0.142, n.s.) and dense leads at both on NQ (53 vs 27, p = 0.0049) and HotpotQA, stable
across *k* = 1…20 and chunk sizes 128/256/512. Conditional on reaching a gold document at *k* = 5
the two cover the span at similar rates (60.8% vs 62.5%) — which the localisation study shows is a
coincidence of the top-*k* cut, not equal ranking skill: BM25 ranks worse inside the document and
compensates by admitting more chunks from it (3.2 vs 2.7). The withdrawal is kept on record here
and in the manuscript.

## Reproducibility

Everything the manuscript reports traces to a file in this repository or to the annotation package
deposited with the Zenodo record:

| Available in the repository | Where |
|---|---|
| Localisation protocol, seven-localiser runs, cross-encoders and variants | [`scripts/localisation_probe.py`](scripts/localisation_probe.py), [`localisation_extra.py`](scripts/localisation_extra.py) |
| Consolidated metrics, Holm-corrected pairwise tests, sign tests, Wilson intervals, Spearman, union analysis, figure | [`localisation_report.py`](scripts/localisation_report.py), [`localisation_analysis.py`](scripts/localisation_analysis.py), [`localisation_figure.py`](scripts/localisation_figure.py) |
| Robustness: document-level cluster bootstrap, chunk audit, model metadata, hit@k from stored ranks, reach × localisation Fisher tests | [`localisation_robustness.py`](scripts/localisation_robustness.py) |
| Admission-budget and chunk-size sweeps; corpus contrasts | [`localisation_admission.py`](scripts/localisation_admission.py), [`localisation_contrast.py`](scripts/localisation_contrast.py) |
| Stored per-question rows and summaries for every localisation table | [`results/localisation/`](results/localisation/README.md) (29 files; file table in its README) |
| A/B/C decomposition, sweeps, BM25 baseline, oracle-evidence control | [`reproduce_study.py`](scripts/reproduce_study.py), [`run_bm25_baseline.py`](scripts/run_bm25_baseline.py), [`run_oracle_evidence.py`](scripts/run_oracle_evidence.py); outputs in [`results/`](results/) |
| Annotation tooling: blinded package build, offline annotation server, validation, guideline audit, review subset, final dataset with provenance, scoring | [`build_annotation_package.py`](scripts/build_annotation_package.py), [`annotate.py`](scripts/annotate.py), [`audit_human_annotations.py`](scripts/audit_human_annotations.py), [`build_review_subset.py`](scripts/build_review_subset.py), [`build_final_human_dataset.py`](scripts/build_final_human_dataset.py), [`score_annotations.py`](scripts/score_annotations.py) |
| Gold-span audit: lexical/semantic proxies, blind adjudication sheet, stratified estimator | [`audit_gold_span_coverage.py`](scripts/audit_gold_span_coverage.py), [`audit_gold_span_semantic.py`](scripts/audit_gold_span_semantic.py), [`build_goldspan_adjudication.py`](scripts/build_goldspan_adjudication.py), [`score_goldspan_adjudication.py`](scripts/score_goldspan_adjudication.py) |
| Threshold ablation, truncation audit | [`threshold_ablation.py`](scripts/threshold_ablation.py), [`audit_annotation_truncation.py`](scripts/audit_annotation_truncation.py) |
| Statistics (Wilson, exact McNemar, bootstrap, Holm) | [`src/evaluation/statistics.py`](src/evaluation/statistics.py) |
| Human study and gold-span documentation | [`docs/paper/human_validation_final.md`](docs/paper/human_validation_final.md), [`docs/paper/`](docs/paper/README.md) |
| Analysis → command → output map | [`docs/paper/reproducibility.md`](docs/paper/reproducibility.md) |

**Not in the git tree.** The raw corpora (fetched from their original sources with the commands and
checksums in [docs/DATASETS.md](docs/DATASETS.md)); model weights (pulled from Hugging Face on
first use); and `reports/`, which holds the run records and the annotation package — annotation
labels are data, not a computation, so the package (200 units with full retrieved context,
original/review/final labels with provenance, guideline and truncation audits, gold-span
adjudication sheet and answers, threshold-ablation and oracle-evidence rows; QASPER excerpts under
CC BY 4.0) is deposited as a separate checksummed archive (SHA-256 `d6d84ecb15585393…`) in a new
version of the Zenodo record of release v1.0.0 (concept DOI 10.5281/zenodo.22879357).
Every report embeds the git commit, raw-file SHA-256, configuration, threshold fingerprint and
package versions. Approximate nearest-neighbour search moves fine-grained decomposition aggregates
by ≤ 0.001 between independently built indices; the localisation study has no randomness (stable
argsort of deterministic scores).

**Reproducing the analyses.**

```bash
pip install -r requirements.txt            # everything except plotting
python scripts/reproduce_study.py --all    # A/B/C decomposition on four corpora, no API key
# within-document localisation: exact commands and runtimes in results/localisation/README.md,
# e.g. hit@k and the reach tests from the committed rows (seconds, no model download):
python scripts/localisation_robustness.py --hit-at-k qasper --out results/localisation/hit_at_k_qasper.json
python scripts/localisation_robustness.py --reach-association qasper --out results/localisation/reach_association_qasper.json
# human study and gold-span audit: commands in docs/paper/human_validation_final.md §7 and
# docs/paper/reproducibility.md (need the annotation package under reports/)
pip install -r requirements-research.txt && python scripts/make_paper_figures.py --all   # figures
```

## Repository structure

```
src/
  data/loaders/      QASPER, Natural Questions, HotpotQA, 2WikiMultihopQA loaders (+ parquet variants)
  data/              corpus chunking with character offsets, identity checks, licence metadata
  rag/               chunker, embedders (query prefixes per model), vector store, providers, local LLM
  evaluation/        evidence coverage, metrics, statistics, failure taxonomy, records, provenance, runner
  api/, monitoring/  the FastAPI service and Prometheus metrics of the pipeline (not used by the study)
scripts/             36 scripts: localisation_*.py, run_*.py, build_*/audit_*/score_*.py, reproduce_study.py,
                     threshold_ablation.py, make_*figures.py, report_tables.py, curate_results.py
results/
  localisation/      per-question rows, summaries, tests, bootstrap, sweeps, contrasts, hit@k, figure, README
  figures/           decomposition, oracle, human-validation and gold-span figures
  *.json             A/B/C decomposition, BM25 baseline and sweep outputs
docs/
  EXPERIMENTS.md     decomposition protocol, all results, threats to validity
  TAXONOMY.md, EVALUATION.md, ANNOTATION_GUIDELINES.md, DATASETS.md, ARCHITECTURE.md, QUICKSTART.md
  paper/             supporting documents for the manuscript, with the stage of each (docs/paper/README.md)
tests/               23 test modules, 488 tests
data/documents/, data/eval/   the small sample corpus used by the offline pipeline smoke test
CITATION.cff, .zenodo.json, LICENSE, pyproject.toml, CHANGELOG.md
```

## Installation

Python 3.11 or newer (CI runs 3.12).

```bash
git clone https://github.com/pouyapd/TrustRAG.git && cd TrustRAG
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # service, evaluation, tests
pip install -r requirements-research.txt             # + matplotlib, torch/transformers for figures and local generators
python scripts/run_offline_eval.py                   # end-to-end smoke run on the sample corpus, ~30 s, no key
```

No API key is needed for anything reported here; all models run locally on CPU. The FastAPI
service (`uvicorn src.api.main:app`) and the Docker setup are the pipeline the evaluation wraps;
[docs/QUICKSTART.md](docs/QUICKSTART.md) covers them.

## Tests

```bash
pytest tests/ -q                     # 488 tests
ruff check src/ tests/ scripts/      # clean
```

Unit tests, property-style invariants (span coverage implies document coverage for every record;
the offset identity for every chunk), end-to-end integration from a dataset file through chunking
and retrieval to a failure label, statistical helpers against known values, a test that recomputes
hit@k and the reach tests from the committed localisation rows, and a regression test for every
defect found during the work — including the 600-character annotation-display truncation that hid
retrieved evidence from the annotator and biased labels toward blaming retrieval. CI runs lint,
tests, an evaluation regression and a Docker build on every push.

## Limitations

Read before quoting any number above.

- **One annotator, and a guided review.** The human study is author-conducted; there is no
  inter-annotator agreement, and the second pass was directed by an audit of the guidelines being
  tested. 22 of the 200 labels retain pilot-era provenance.
- **The gold-span estimate is not definitive.** 0.119 [0.096, 0.142], defensibly 4–12%; the
  interval covers sampling error only, the census part rests on proxies, and it is measured on
  QASPER only.
- **Generation-side categories are unvalidated**; several have near-zero support in the human
  labels, and the annotated run uses an extractive control that cannot hallucinate.
- **Two corpora, one chunking, small models.** Seven localisers with one reranker per size class,
  revisions recorded after the fact; no claim is made about other corpora, chunkers or larger
  models, and the corpus contrasts are descriptive, not causal.
- **Reach × localisation** was tested on QASPER only (NQ reach is saturated) with limited power.
- **The oracle control** has n = 150, one small reader, no sham control, and two strata with too
  few discordant pairs.
- **The core premise is prior art.** This is a measurement study, not a new evaluation paradigm
  ([docs/paper/literature_review.md](docs/paper/literature_review.md)).
- **Not a deployed system** — containerised and CI-tested, never run at production scale.

## Version and release

`v1.0.0` (21 September 2026) is the research and reproducibility release: the state of the code,
result files and documentation that the manuscript reports. Later commits may change documentation
or add analyses; anything the manuscript cites is fixed at this tag, and a change to a cited result
file would be released under a new tag. Release notes: [CHANGELOG.md](CHANGELOG.md).

## Citation

`CITATION.cff` is the source of truth (GitHub's "Cite this repository" reads it). Until the article
exists, cite the software release:

```bibtex
@software{bathaeipourmand_trustrag_2026,
  author  = {Bathaei Pourmand, Pouya},
  title   = {{TrustRAG}: evidence localisation and failure attribution for retrieval-augmented generation evaluation},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/pouyapd/TrustRAG},
  doi     = {10.5281/zenodo.22879357}
}
```

The manuscript *Right document, wrong passage: evidence localisation in retrieval-augmented
generation evaluation on QASPER and Natural Questions* is prepared for submission to *Language
Resources and Evaluation*; a citation to the article will replace the software citation when it
exists.

## License

MIT for the code and the result files ([LICENSE](LICENSE)). The evaluated corpora carry their own
licences ([docs/DATASETS.md](docs/DATASETS.md)); the annotation package excerpts QASPER text and is
distributed under CC BY 4.0.

## Author

Pouya Bathaei Pourmand — independent researcher, Genoa, Italy.
