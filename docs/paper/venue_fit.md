# Venue fit

Written after the literature review and the reviewer simulation, both of which
constrain the realistic target. Scope descriptions below were checked against the
venues' own current calls; rankings (Q1/Q2, CORE) shift year to year and are given as
indicative rather than verified for the current cycle.

## What we are placing

A measurement-validity study, not a new method. Its assets are a retriever-ranking
inversion, a human-validated attribution comparison, an honest replication, and two
negative results about the instrument itself. Its liabilities are that the core
premise is prior art, the human study has one annotator and no inter-annotator
agreement, and the retrieval baselines are two (dense + BM25) with no reranker or
hybrid.

The simulated reviews returned weak reject / reject / borderline for top venues. Any
plan that ignores that is wishful.

## Candidates

| Venue | Type | Scope fit | Novelty bar | Human-eval expectation | Risk |
|---|---|---|---|---|---|
| **SIGIR (full)** | Q1 conf | High — retrieval evaluation is core | Very high | Moderate | **Reject likely.** Reviewer B's baseline objection is disqualifying here |
| **ACL/EMNLP (main)** | Q1 conf | High | Very high | High | **Reject likely.** Premise is prior art |
| **TOIS / IP&M** | Q1 journal | High | High, but rewards thorough evaluation studies | High | Moderate-high; needs the second annotator and more retrievers |
| **ECIR (short)** | Q2 conf | High | Moderate — reproducibility and evaluation papers welcome | Low-moderate | **Realistic** |
| **SIGIR Resource & Reproducibility track** | Q1 conf, distinct track | Very high — replication and measurement validity are the point | Moderate; replication is a virtue | Low | **Realistic and the best strategic fit** |
| **EMNLP/ACL Findings** | Q1-adjacent | High | Moderate | Moderate | Realistic |
| **TrustNLP / GEM / workshop** | Workshop | Very high | Low-moderate | Low | **Safe** |

## Recommendation

**Ambitious — SIGIR Reproducibility track (or TOIS).** The reproducibility track is
the one place where "we replicated a published result at small scale and found where
the measurement breaks" is a feature rather than a weakness. To be credible there,
add: a second annotator, a reranker or hybrid baseline, and the entailment
adjudication of the 87 unresolved gold-span units.

**Realistic — ECIR short paper or ACL/EMNLP Findings.** The retriever-ranking
inversion plus the gold-span under-coverage analysis is enough for a 4–6 page
contribution as it stands. This is where the work is publishable *today*, with the
limitations section intact.

**Safe — a trustworthy-NLP or evaluation workshop.** Accepts the current empirical
base without modification, and would generate the reviewer feedback needed to decide
whether to invest in the ambitious route.

## Honest assessment

The project is **not ready for a Q1 main-track submission** and would likely be
desk-weakened or rejected on novelty. It **is ready** for a workshop or short-paper
submission now, and could reach the SIGIR Reproducibility track or a journal with
roughly three additions: a second annotator (human time), stronger retrieval baselines
(compute only, feasible locally), and human entailment adjudication of the 87
unresolved units (human time).

The two human-time items are the binding constraint. No amount of further automated
work substitutes for them.
