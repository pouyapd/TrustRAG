# Paper outline

Target: a short empirical paper (8–9 pages) on RAG evaluation methodology.
Every section below lists the repository artifact that supports it. Sections
marked **[GAP]** are not yet supported by the repository and must either be run
or dropped — they are listed in `limitations.md` §"What is missing for a paper".

---

## 1. Introduction

**Claim.** Conventional RAG evaluation reports whether a chunk from a relevant
*document* was retrieved. On long-document corpora that is not the same question
as whether the passage supporting the answer reached the generator, and the
difference changes which pipeline stage a failure is charged to.

**Opening evidence** (`docs/EXPERIMENTS.md`, `results/`): on Natural Questions a
document-level reading charges 1 of 300 failures to retrieval; the evidence-level
reading charges 81. Same stored retrieval output, opposite engineering
conclusion.

**Contributions** — as listed in `RESEARCH_SUMMARY.md` §4.

## 2. Related work

**[GAP]** No literature review exists in the repository. This section must be
written from scratch; no citations are fabricated here.

## 3. Problem formulation

- RAG pipeline and where each failure can originate.
- Definition of document-level vs span-level retrieval success.
- Why a single aggregate score cannot separate them.
- Formal statement of the three retrieval definitions A / B / C
  (`src/evaluation/metrics.py`, `src/evaluation/evidence.py`).

## 4. Method

### 4.1 Offset-carrying pipeline
Chunks are sliced, never re-decoded; the invariant
`document[start:end] == chunk.text` is property-tested
(`tests/test_chunk_offsets.py`).

### 4.2 Evidence alignment
Half-open interval overlap; `any_sufficient` vs `all_required` for multi-hop.

### 4.3 Failure taxonomy v2
Nine categories, versioned thresholds, one rule recorded per row, stage
attribution declared rather than inferred (`src/evaluation/taxonomy.py`,
`docs/TAXONOMY.md`).

### 4.4 Two retrieval gates
`failure_mode_v2` gates rule R4 on document-level retrieval;
`failure_mode_evidence` gates it on gold-span coverage. Both are computed for
every row from the same stored inference.

### 4.5 Annotation protocol
Blinded stratified package, offline annotation server, withheld key, validation
checks (`scripts/build_annotation_package.py`, `scripts/annotate.py`,
`docs/ANNOTATION_GUIDELINES.md`).

### 4.6 Context integrity
The 600-character display truncation, why it made step 2 of the guidelines
unanswerable, the fix, the audit, and the three guards that prevent recurrence
(builder abort, `--validate` report, regression test).

## 5. Experimental setup

See `experimental_setup.md`. Corpus, chunking, retrieval, generator, sampling,
metrics, and the exact commands.

## 6. Results

See `results.md` and `TABLES.md`.

1. Retrieval decomposition A/B/C across four corpora, with robustness sweeps
   (4 embedders, 5 depths, 4 chunk sizes, second multi-hop corpus).
2. Attribution shift per corpus.
3. Taxonomy vs the reference set: document-gated vs evidence-gated, per-category
   precision/recall/F1, paired McNemar.
4. Context-integrity audit.
5. Annotation agreement.
6. Generation replay: what a small LLM does when the evidence is missing.

## 7. Error analysis

See `results.md` §5. Four patterns, all from the confusion matrices:
misattributed retrieval failures, `partial_answer` over-prediction,
`hallucination` prediction against zero support, and the labels that moved when
full context was restored.

## 8. Discussion

- What an evaluation designer should change: report span-level coverage beside
  document-level recall; gate attribution on evidence.
- What the reference-set experiment does and does not license as a claim.

## 9. Limitations

See `limitations.md`. Must appear before any headline number is quoted.

## 10. Conclusion

Evidence gating is a small change to the attribution rule with a measurable
effect on agreement with independent judgement, and it is cheap: it reuses
stored retrieval output and costs no model calls.

---

## Figure and table plan

See `FIGURES.md` and `TABLES.md`.

## Section-by-section support status

| Section | Supported by repository | Notes |
|---|---|---|
| 1 Introduction | Yes | NQ 1 vs 81, QASPER/Hotpot/2Wiki gaps |
| 2 Related work | **No** | must be written; no citations exist here |
| 3 Problem formulation | Yes | metric definitions in code + `docs/EVALUATION.md` |
| 4 Method | Yes | all components implemented and tested |
| 5 Setup | Yes | `summary.json`, `manifest.json` |
| 6 Results | Yes | `final_evaluation.json`, `TRUNCATION_AUDIT.json`, `results/` |
| 7 Error analysis | Yes | confusion matrices + per-unit joins |
| 8 Discussion | Interpretation | must be marked as interpretation, not measurement |
| 9 Limitations | Yes | provenance files state what is model-generated |
| 10 Conclusion | Yes | follows from §6 |
