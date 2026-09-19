# TrustRAG — Evidence-Aware RAG Evaluation

[![CI](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-486%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-80%25-green)

**TrustRAG is a research codebase for evaluating retrieval-augmented generation at the level of
evidence, not documents.** Chunks carry character offsets from the source document into the vector
store and the stored run, so "did the passage that supports the answer arrive?" is decided by
interval overlap after the fact. On top of that it holds a nine-category failure taxonomy with two
interchangeable retrieval gates, blinded annotation tooling, and a within-document localisation
study. That document-level retrieval metrics overstate success on long documents is
[established prior work](docs/paper/literature_review.md); this repository measures what happens
after the right document is reached, and what span-level evaluation buys and costs.

> **Research questions.**
> 1. Inside the document that holds the evidence, how often does a retriever rank the evidence
>    chunk first, and does that depend on the model family, model size, or the corpus?
> 2. When RAG failure attribution is gated on span-level evidence rather than document-level
>    retrieval, does it agree better with human judgement — and how far can the span-based gold
>    standard itself be trusted?

**What was tested.** Seven localisers (BM25, four dense bi-encoders, two cross-encoders, 22M–278M)
ranking every chunk of the gold document on QASPER (290 questions) and Natural Questions (300),
under one protocol with paired tests and an analytic chance level; a document/span retrieval
decomposition on four corpora; a 200-unit human annotation study of the taxonomy with a guided
second review; a 60-unit human adjudication of gold-span completeness; and a paired oracle-evidence
replication with a 0.5B reader.

![Rank of the gold chunk inside its document, per model and corpus](results/localisation/rank_distribution.png)

**Strongest current result.** Inside the right document, no retriever ranks the evidence first
reliably, and capacity does not fix it consistently:

| | QASPER | NQ |
|---|---|---|
| Chance hit@1 (random order of the document's chunks) | 0.157 | 0.116 |
| hit@1, all seven models | 0.290 – 0.400 | 0.353 – 0.507 |
| hit@5, all seven models | 0.755 – 0.855 | 0.690 – 0.837 |
| Best single model | BGE-small 33M, 0.400 | bge-reranker-base 278M, 0.507 |
| Worst neural model | bge-reranker-base 278M, 0.310 | MiniLM 22M, 0.387 |
| Some model ranks the gold first | 0.769 | 0.780 |
| Pairs significant after Holm (of 21) | 2 | 6 |

The median rank is 2 on both corpora and a third to a half of gold chunks sit at ranks 2–5. BM25 is
the weakest localiser on both, the 110M bi-encoder trails 33M models on both, and no association
between reach and localisation was detected. The cross-encoder result does *not* replicate: the 278M reranker is
the best model on NQ and the worst neural model on QASPER, and its NQ lead over BGE-small is not
significant (*p* = 0.19). Full study, raw rows, tests and the corpus contrast:
[results/localisation/README.md](results/localisation/README.md).

**Where to find things.** Results and analysis → [results/localisation/](results/localisation/README.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/paper/](docs/paper/) · Reproduction → [Reproducibility](#reproducibility)
and [docs/paper/reproducibility.md](docs/paper/reproducibility.md) · What is and is not claimed →
[literature_review.md](docs/paper/literature_review.md), [Limitations](#limitations).

**All findings, including one we withdrew.**

| | Result |
|---|---|
| **Inside the right document, no retriever ranks the evidence first reliably — and capacity does not fix it consistently** | Seven localisers rank the gold chunk first for **0.29–0.40** of QASPER questions and **0.35–0.51** of NQ questions, against top-5 rates of 0.69–0.86; the median rank is 2. The 278M reranker is the best model on NQ and the worst neural model on QASPER; the 110M bi-encoder trails 33M models on both. [Full study](results/localisation/README.md). |
| **Evidence-gating agrees better with humans — for retrieval attribution only** | Against 200 human-reviewed labels: accuracy **0.700 vs 0.600**, κ **0.437 vs 0.375**, paired **22 vs 2**, exact McNemar *p* < 0.0001. But only the retrieval classes are reliable: `wrong_retrieval` F1 0.907 against `ok` recall 0.094. |
| **The span-based gold standard is incomplete, and now measured** | Human adjudication of 60 sampled units puts gold-span under-coverage at **0.119, 95% CI [0.096, 0.142]** — the span rule calls a retrieval failure where the answer was in fact derivable. A sensitivity analysis places the defensible range at **4–12%**. Modest, quantified, and an order of magnitude smaller than the effects measured. |
| **A result we reported and then withdrew** | We claimed the document/span choice inverts the BM25-vs-dense ranking. It does not — the finding was an evidence-mode bug in our own baseline. Corrected, BM25 leads at *both* granularities on QASPER (0.528/0.321 vs 0.441/0.276) and dense leads at both on NQ and HotpotQA, across 5 depths and 3 chunk sizes. Reported in full [below](#a-withdrawn-result-no-retriever-ranking-inversion). |

**What this repository does not claim.** Evidence-aware RAG evaluation is not new here;
neither is the failure taxonomy, nor the oracle-evidence experiment (a
[replication](#oracle-evidence-control-a-replication) — 32.1% repair
against 32.8% published). See [literature_review.md](docs/paper/literature_review.md) for
what is and is not novel. The manuscript is kept private and is not published.

📋 **[Reviewer simulation](docs/paper/reviewer_simulation.md)** · 🎯 **[Venue fit](docs/paper/venue_fit.md)** · 🧪 **[Full experiment log](docs/EXPERIMENTS.md)**

---

## Contents

[Core insight](#core-insight) · [Methodology](#methodology) · [Results](#results) ·
[Human validation](#human-validation) · [Gold-span limits](#how-far-the-gold-standard-can-be-trusted) ·
[Datasets & setup](#datasets-and-experimental-setup) · [Reproducibility](#reproducibility) ·
[Install](#install-and-use) · [Tests](#tests) · [Limitations](#limitations)

---

## Core insight

A gold span and a retrieved chunk are half-open character intervals in the same
document. Overlap decides coverage — arithmetic, not string search:

```
document  qasper:1901.00001
gold span        [1200, 1760)
retrieved chunk  [ 900, 2100)   overlap = 560 chars  ->  evidence covered
retrieved chunk  [8300, 9500)   overlap =   0 chars  ->  same document, no evidence
```

Both chunks satisfy a document-level metric. Only the first makes the question
answerable from context. Three definitions of retrieval success follow:

- **A** — some chunk from a relevant document (the conventional metric)
- **B** — every document a multi-hop question requires
- **C** — a retrieved chunk actually contains the gold span

`C ≤ B ≤ A` by construction. A→B isolates a **quantifier** effect, B→C a **granularity**
effect, and the two are near-orthogonal: each is null on the corpus where the other
dominates.

## Methodology

Character offsets travel chunker → vector store → retrieval → stored record, so
`document[chunk.start:chunk.end] == chunk.text` holds by construction and is
property-tested. That is what makes span coverage computable after the fact.

The failure taxonomy has nine categories and one rule — R4 — that decides whether a row
is charged to retrieval. It reads a single boolean. **The document-gated variant binds it
to A; the evidence-gated variant binds it to C. Everything else is identical**, and both
labels are written to every row, so the comparison runs on identical retrieval output at
zero extra inference cost.

Design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Taxonomy: [docs/TAXONOMY.md](docs/TAXONOMY.md)

## Results

### Retrieval decomposition

| Corpus | n | A | B | C | quantifier A→B | granularity B→C |
|---|---:|---:|---:|---:|---:|---:|
| QASPER dev | 290 | 0.441 | 0.441 | 0.276 | 0.0 pp | **16.6 pp** (p=7.1e-15) |
| Natural Questions | 300 | 0.997 | 0.997 | 0.730 | 0.0 pp | **26.7 pp** (p=1.7e-24) |
| HotpotQA | 150 | 0.993 | 0.507 | 0.507 | **48.7 pp** (p=2.1e-22) | 0.0 pp |
| 2WikiMultihopQA | 150 | — | — | — | **64.7 pp** (p=1.3e-29) | 1.3 pp (n.s.) |

![A/B/C decomposition across corpora](results/figures/abc_decomposition.png)

### A withdrawn result: no retriever-ranking inversion

![Dense vs BM25 under both definitions](results/figures/bm25_vs_dense.png)

An earlier version of this README claimed the document/span choice reverses the
BM25-vs-dense comparison. **It was an artefact of our own bug**: QASPER declares
`any_sufficient` evidence mode, the BM25 baseline hard-coded `all_required`, and 51% of
QASPER questions carry more than one span — so BM25's span coverage was under-reported
(0.183 instead of 0.321) against a dense pipeline using the correct mode.

Corrected, **no inversion occurs on any corpus**: BM25 leads at both granularities on
QASPER (paired 40 vs 27, p = 0.142, n.s.), dense leads at both on NQ (53 vs 27,
p = 0.0049) and HotpotQA. Stable across k = 1…20 and chunk sizes 128/256/512.
Conditional on reaching a gold document at k = 5, the two retrievers cover the span at
similar rates (60.8% vs 62.5%) — which is why an inversion was implausible. The section
below shows that this similarity is a coincidence of the top-k cut, not equal ranking
skill: BM25 ranks worse inside the document and compensates by admitting more chunks
from it (3.2 vs 2.7).

### Inside the right document: who finds the passage?

The granularity gap says retrievers reach the right document and miss the passage. This
study asks whose fault that is. For every question, *all* chunks of the gold document are
ranked against the question by each of seven localisers, so no other document competes,
and the rank of the first gold-overlapping chunk is recorded against an analytic chance
level. Same chunking, same evidence definition, same statistics on both corpora.

The headline table is at the top of this README; the full tables are in [results/localisation/README.md](results/localisation/README.md).

**What replicates.** Every model places the evidence near the top of its document but
rarely first — median rank 2, a third to a half of gold chunks at ranks 2–5 — and the
misses are only partly shared (union of seven 0.77–0.78 vs best single 0.40–0.51). BM25 is
the weakest localiser on both corpora and the only one whose deficit survives Holm
correction on both; it succeeds at 0.84–0.89 when the gold chunk is the lexically most
question-like chunk of its document and at 0.04–0.06 otherwise. The 110M bi-encoder trails
the 33M ones on both corpora. No association between reach and localisation was detected
(Fisher exact *p* = 0.31–0.63, QASPER; power is limited), and the pairwise conclusions hold
under a document-level cluster bootstrap.

**What does not.** Cross-encoder capacity helps on Wikipedia and not on scientific papers:
the 278M reranker is the best model on NQ (+4 to +12 pp over the bi-encoders) and the
worst neural model on QASPER (below every bi-encoder, indistinguishable from BM25). The
corpora differ in ways consistent with this — QASPER papers are far more topically
homogeneous chunk-to-chunk (mean intra-document cosine 0.58 vs 0.46), their evidence is
lexically hidden from the question four times as often, and NQ-style web QA is in the
documented training data of every neural model tested while scientific-paper QA is in
none — but this is a hypothesis, not a finding.

**The claim that survives both corpora:** increasing retriever or reranker capacity does
not produce a consistent improvement in rank-1 evidence localisation, and the localisation
term is closed only by admitting more of the reached document (top-1 chunk 0.31–0.39 →
top-5 0.77–0.85 → top-10 0.95 on QASPER), at the price of reach; no fixed document-first
allocation beats flat top-*k* at equal budget. Full tables, pairwise tests, correlations,
the corpus contrast and limitations: [results/localisation/README.md](results/localisation/README.md).

### Oracle-evidence control (a replication)

Every question answered twice by the same generator, same prompt, same decoding — only
the context differs (retrieved chunks vs the gold spans verbatim). Within-question
pairing removes question difficulty and generator identity as confounds.

| Stratum | n | retrieved | oracle | difference | p |
|---|---:|---:|---:|---:|---:|
| Evidence complete under retrieval | 46 | 0.065 | 0.174 | +10.9 pp | 0.125 (n.s.) |
| **Document retrieved, span missing** | 26 | **0.000** | **0.231** | **+23.1 pp** | 0.031 |
| Nothing from any gold document | 78 | 0.000 | 0.321 | +32.1 pp | 6.0e-08 |

![Oracle-evidence control](results/figures/oracle_evidence.png)

The middle row is the argument: those 26 questions are scored as retrieval *successes* by
a document-level metric, the model got none right, and supplying the actual span repairs
23% of them. This replicates [arXiv:2608.08944](https://arxiv.org/html/2608.08944)
(32.8% repair over 11,105 failures) at n=150 with one reader and no sham control.

## Human validation

![Both retrieval gates scored against the final human-reviewed labels](results/figures/human_validation.png)

**Provenance matters here, so it is stated precisely.**

| Artifact | What it is |
|---|---|
| `annotator_human/completed.jsonl` | **Human**, original pass, 200 units |
| `review_43_flagged/annotator_review/` | **Human**, second review of 43 audit-flagged units |
| `final_human_reviewed/completed.jsonl` | **Derived**: original label where unflagged, review decision where flagged |
| `annotator_a/completed.jsonl` | **Automated** — a language-model annotator. Not ground truth. |

An audit against the written guidelines flagged 43 of 200 labels as conflicting with an
explicit rule. The annotator re-reviewed those 43 on full context: **36 changed, 7
upheld**. A per-unit provenance chain records `original → flag reason → review → final`.

**This is agreement with a guided expert reading, not independent validation.** The
annotator was told which units to re-examine and why, and the changes moved toward what
the guidelines prescribe. Reported honestly rather than as validation.

| Variant | Accuracy | 95% CI | Macro F1 | κ |
|---|---:|---:|---:|---:|
| Document-gated | 0.6000 | 0.531–0.665 | 0.4764 | 0.3752 |
| **Evidence-gated** | **0.7000** | 0.633–0.759 | **0.4819** | **0.4371** |

Per class, evidence-gated: `wrong_retrieval` F1 **0.907** (support 136),
`answered_when_unanswerable` F1 **1.000** (9), `partial_answer` 0.286 (22), `ok` 0.158
(32), `incorrect_answer` 0.059 (1).

> **Retrieval-side attribution is validated. Generation-side classification is not.** A
> [held-out threshold ablation](docs/paper/results.md) (144
> configurations, 50/50 split) improves the evidence gate from 0.730 to 0.750 accuracy
> and does not rescue the generation classes — the rules, not the thresholds, are the
> larger problem.

Agreement with the *automated* reference pass is markedly higher (0.805, κ 0.631) than
agreement with humans. Two automated readings share failure directions; **the human
number is the one reported.**

Full detail: [human_validation_final.md](docs/paper/human_validation_final.md)

## How far the gold standard can be trusted

![Proxy partition of the 133 zero-coverage units, before human adjudication](results/figures/gold_span_validity.png)

*The proxy partition that preceded adjudication. The 87 unresolved units in the three
right-hand bands are what the human study below settled.*

**Question put to a human annotator, blind to every previous label:** *is the reference
answer derivable from the retrieved text alone, without relying on the annotated gold
span?*

All 133 answerable units with zero gold-span coverage were partitioned by two automated
proxies. Where both agreed, the unit was counted directly (36 answer absent, 10 answer
present). The 87 they could not resolve were sampled — 60 units, stratified,
seed `20260907` — and adjudicated by hand: **4 YES, 56 NO, 0 CANNOT_TELL**.

| | Estimate |
|---|---|
| **Gold-span under-coverage** | **0.119**, 95% CI **[0.096, 0.142]** (≈16 of 133 units) |
| Defensible range under sensitivity analysis | **4% – 12%** |
| Share of the full 200-unit annotation set | ≈ 8% |

The confidence interval covers sampling error only. The point estimate leans on 10
units both proxies called under-coverage that **no human ever checked** — and the
adjudication showed the human agreeing with "answer present" on only 6.7% of unresolved
units, well below the 100% the proxies asserted there. If those 10 behave like the
adjudicated ones, the rate is 0.049. Adjudicating them is the cheapest remaining
improvement in the project.

**What this means:** span-level evidence is a substantially sound instrument for
retrieval attribution. The bias runs in the expected direction — QASPER marks supporting
sentences, not every passage an answer can be derived from — but it is modest and does
not undermine the retrieval-side conclusions above.

## Datasets and experimental setup

| Dataset | Licence | Structure | Role |
|---|---|---|---|
| QASPER | CC BY 4.0 | NLP papers, ~22k chars | granularity |
| Natural Questions | CC BY-SA 3.0 | Wikipedia, ~37k chars | granularity |
| HotpotQA | CC BY-SA 4.0 | 10 paragraphs, 2 gold | quantifier |
| 2WikiMultihopQA | Apache-2.0 | 10 paragraphs, 2–4 gold | quantifier (replication) |

Chunk size 256, overlap 32, top-k 5, `all-MiniLM-L6-v2`; BM25 (Okapi, k1=1.5, b=0.75)
over identical chunks. Robustness: 4 embedders, depths k=1…20, chunk sizes 128/256/512.
Generators: deterministic extractive control; Qwen2.5-0.5B-Instruct and
SmolLM2-360M-Instruct locally. **Corpora are not redistributed** — loaders, checksums and
licences are committed ([docs/DATASETS.md](docs/DATASETS.md)).

Statistics: Wilson intervals, seeded bootstrap, **exact** McNemar for paired binary
comparisons, `MIN_N_FOR_INFERENCE = 30` as a stated convention. No multiple-comparison
correction is applied; the headline results survive one, the marginal ones would not.

## Reproducibility

```bash
python scripts/reproduce_study.py --all     # every retrieval experiment, no API key
```

Every table and figure maps to a command and an output file in
**[docs/paper/reproducibility.md](docs/paper/reproducibility.md)**. Reports embed git
commit, raw-file SHA-256, configuration, threshold fingerprint and package versions.

Two honest limits: `reports/` is gitignored, so the annotation artifacts behind the human
validation are produced locally rather than shipped; and approximate nearest-neighbour
search moves fine-grained aggregates by ≤0.001 between independently built indices
(headline figures reproduce exactly).

## Install and use

```bash
git clone https://github.com/pouyapd/TrustRAG.git && cd TrustRAG
pip install -r requirements.txt
python scripts/run_offline_eval.py      # end-to-end evaluation, ~30s, no key
pytest tests/ -q                        # 486 tests
```

<details>
<summary><b>All research commands</b></summary>

```bash
# retrieval study and robustness sweeps
python scripts/reproduce_study.py --all
python scripts/reproduce_study.py --embedder-sweep --topk-sweep --multihop

# BM25 baseline, scored under the same definitions
python scripts/run_bm25_baseline.py --dataset qasper \
    --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300 \
    --dense-rows reports/experiments/qasper_dev_300/rows.jsonl \
    --dense-records reports/experiments/qasper_dev_300/inference.jsonl \
    --out results/bm25_qasper_dev_300.json

# paired oracle-evidence control
python scripts/run_oracle_evidence.py \
    --records reports/experiments/qasper_dev_300/inference.jsonl \
    --generator qwen0.5b --limit 150 --out reports/experiments/oracle_qasper_qwen

# annotation: build a blinded package, annotate locally, validate
python scripts/build_annotation_package.py --records ... --out ... --n-units 200
python scripts/annotate.py --annotator human --package ...
python scripts/annotate.py --annotator human --package ... --validate

# audit the labels, derive the final reviewed dataset, check the gold standard
python scripts/audit_human_annotations.py --package ... --annotator human --out ...
python scripts/build_final_human_dataset.py --original ... --review ... --out ...
python scripts/audit_gold_span_semantic.py --package ... --out ...
python scripts/threshold_ablation.py --package ... --labels ... --out ...

# within-document localisation study (commands and runtimes in results/localisation/README.md)
python scripts/localisation_probe.py --dataset qasper --raw ... --embedders minilm,mpnet,bge,e5 --out ...
python scripts/localisation_extra.py --dataset qasper --raw ... --cross-encoder BAAI/bge-reranker-base --out ...
python scripts/localisation_report.py --dataset qasper --probe ... --extra ... --reranker ... --out ...

# figures
pip install -r requirements-research.txt
python scripts/make_figures.py --all && python scripts/make_paper_figures.py --all
```
</details>

## Tests

```bash
pytest tests/ -v --cov=src      # 486 tests, 80% line coverage, ruff clean
```

Unit tests, property-style invariants (span coverage implies document coverage, for every
record), end-to-end integration from a real dataset file through chunking and retrieval to
a failure label, and a regression test for every defect found during the work — including
a 600-character annotation truncation defect that hid 49% of retrieved evidence from
annotators and biased labels toward blaming retrieval.

CI runs lint, tests, an evaluation regression and a Docker build on every push.

## Limitations

Read before quoting anything above. Full list in
[docs/paper/limitations.md](docs/paper/limitations.md).

- **One annotator, and a guided review.** No inter-annotator agreement exists. The second
  pass was directed by an audit of the same guidelines being tested.
- **The span gold standard is incomplete** — estimated 0.119 [0.096, 0.142], defensibly 4–12%; the interval covers sampling error only and the adjudication had one annotator.
- **Generation-side categories are unvalidated**; three have zero support in the human
  labels, and the annotated run uses an extractive control that cannot hallucinate.
- **The core premise is prior art.** This is a measurement-validity study, not a new
  evaluation paradigm.
- **Retrieval breadth.** The decomposition uses two retrievers (dense + BM25); the localisation study adds four bi-encoders and two cross-encoders but one reranker per size class, on two corpora. Model revisions were not pinned at run time; the revisions actually run are recorded after the fact in `results/localisation/model_metadata.json`.
- **One corpus and one configuration** for the human study; small generators (0.5B, 0.36B). Gold-span under-coverage is measured on QASPER only, not on NQ.
- **Targeted, not systematic, literature review.**
- **Not a deployed system** — containerised and CI-tested, never run at production scale.

## Documentation

| Document | Contents |
|---|---|
| [results/localisation/README.md](results/localisation/README.md) | Within-document evidence localisation study, QASPER and NQ |
| [docs/paper/literature_review.md](docs/paper/literature_review.md) | Novelty audit and comparison table |
| [docs/paper/human_validation_final.md](docs/paper/human_validation_final.md) | The complete human study |
| [docs/paper/reviewer_simulation.md](docs/paper/reviewer_simulation.md) | Three adversarial reviews and the fixes |
| [docs/paper/venue_fit.md](docs/paper/venue_fit.md) | Where this can realistically be submitted |
| [docs/paper/reproducibility.md](docs/paper/reproducibility.md) | Command → output map |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Full protocol and threats to validity |
| [docs/TAXONOMY.md](docs/TAXONOMY.md) · [docs/EVALUATION.md](docs/EVALUATION.md) | Categories, rules, metric definitions |
| [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) | What annotators are asked to judge |
| [docs/DATASETS.md](docs/DATASETS.md) · [docs/QUICKSTART.md](docs/QUICKSTART.md) | Data provenance; install and run |

## Citation

The manuscript is kept private and is **not published**. Cite the repository:

```bibtex
@software{bathaeipourmand_trustrag_2026,
  author = {Bathaei Pourmand, Pouya},
  title  = {TrustRAG: Evidence-Aware RAG Evaluation},
  year   = {2026},
  url    = {https://github.com/pouyapd/TrustRAG}
}
```

## License

MIT for this code. Evaluated corpora carry their own licences — see
[docs/DATASETS.md](docs/DATASETS.md).

## Author

Pouya Bathaei Pourmand — MSc researcher, Computer Engineering (AI), University of Genoa.
