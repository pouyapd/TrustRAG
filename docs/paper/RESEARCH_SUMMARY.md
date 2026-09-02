# TrustRAG — research summary

Every number in this file is read from a run artifact in this working tree and
carries its source file. Nothing here is estimated, rounded up, or carried over
from an earlier draft. `reports/` is gitignored, so the annotation package and
evaluation JSON are not distributed with a clone; `docs/paper/reproducibility.md`
§12 states what that costs and what would fix it.

**Provenance convention used throughout `docs/paper/`:**

| Term | Meaning in this repository |
|---|---|
| **reference set** | `reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl` — 200 labels, produced by a language model (Claude Opus 5) reading the full retrieved context against the written guidelines. **Not a human annotation pass.** |
| **human-labelled subset** | 22 units in `reports/annotation/qasper_dev_300/annotator_a/completed.jsonl`, listed in that file's `PROVENANCE.md`, labelled by the project owner through the annotation UI. |
| **taxonomy labels** | Labels the *system* assigns automatically (`failure_mode_v2` / `failure_mode_evidence`), the object under evaluation. |

---

## 1. Title candidates

1. *Did the Evidence Reach the Generator? Span-Level Attribution of RAG Failures*
2. *Document-Level Retrieval Metrics Misattribute RAG Failures: Evidence from Span-Aware Evaluation*
3. *TrustRAG: An Evidence-Gated Failure Taxonomy for Retrieval-Augmented Generation*
4. *Retrieval Succeeded, Evidence Did Not: Re-attributing Failure in RAG Pipelines*
5. *From Aggregate Scores to Causes: A Reproducible Failure Taxonomy for RAG Evaluation*

## 2. Research question

Can RAG failures be attributed to retrieval, evidence, generation or abstention
using reproducible, evidence-grounded evaluation — and does gating that
attribution on **span-level evidence availability** rather than **document-level
retrieval** produce attributions that match independent judgement better?

## 3. Central claim

Document-level retrieval success is a poor proxy for *the generator having what
it needed*. On long-document corpora the two come apart systematically, and the
gap changes which pipeline stage a failure is charged to. A taxonomy whose
retrieval rule is gated on gold-span coverage agrees with an independent
annotation of the same units significantly better than the same taxonomy gated
on document retrieval.

## 4. Contributions

1. **Evidence-level retrieval measurement.** Character offsets are carried
   chunker → vector store → retrieval → stored records, so
   `document[chunk.start_char:chunk.end_char] == chunk.text` holds by
   construction and gold-span coverage is computed by interval arithmetic, not
   string search (`src/evaluation/evidence.py`, `tests/test_chunk_offsets.py`).
2. **A three-way decomposition of "retrieval succeeded"** — A (any chunk from a
   relevant document), B (A plus every document a multi-hop question requires),
   C (a retrieved chunk actually contained the gold span) — isolating a
   *quantifier* effect (A→B) from a *granularity* effect (B→C).
3. **A 9-category, versioned failure taxonomy** with hashable thresholds, one
   fired rule recorded per row, and two retrieval gates (document-level and
   evidence-level) computed side by side.
4. **An annotation protocol and tooling** for validating the taxonomy: blinded
   stratified packages, an offline annotation server that cannot read the
   withheld key, kappa/confusion/adjudication scoring, and per-file provenance.
5. **A measurement-integrity result and fix.** The first annotation package
   stored only the first 600 characters of each retrieved chunk while recording
   the full offsets, which made the central annotation question unanswerable;
   the audit quantifies exactly what was hidden and the rebuilt package restores
   it (941/1000 chunks recovered).
6. **External evidence for the central claim.** Scored against the same 200-unit
   reference set, the evidence-gated taxonomy beats the document-gated one:
   accuracy 0.805 vs 0.740, kappa 0.6305 vs 0.5728, exact McNemar p = 0.0294.

## 5. Methodology

- **Pipeline.** FastAPI service; ChromaDB persistent store; chunks sliced from
  source documents with `start_char`/`end_char`; retrieval top-k; generation via
  a pluggable provider or a deterministic extractive control.
- **Retrieval evaluation.** Legacy (frozen, defects documented), corrected
  (document vs chunk units, `None` for unanswerable, nDCG/hit-rate/first-rank),
  and evidence-level (span coverage, evidence recall/precision, first evidence
  rank, multi-hop completeness) reported side by side.
- **Evidence availability.** A gold span and a retrieved chunk are half-open
  character ranges in one document; positive overlap is coverage. Multi-hop
  `all_required`: any missing required span is a retrieval failure.
- **Generation evaluation.** Answer F1/EM, key-fact recall, faithfulness,
  abstention; a separate replay study swaps only the generator on a stored run.
- **Failure taxonomy.** See §7 and `docs/TAXONOMY.md`.
- **Annotation protocol.** See §6 and `docs/ANNOTATION_GUIDELINES.md`.
- **Reproducibility.** Inference and scoring are separate; a finished run is
  re-scorable with zero model calls; every report embeds git commit + dirty
  flag, file checksums, package versions, thresholds fingerprint.

## 6. Annotation protocol (as implemented)

`scripts/build_annotation_package.py` — 200 units sampled from a finished
300-question run under seed `20260826`: a floor of 8 units per proposed failure
mode, 25% of the budget (50 units) reserved for rows within 0.1 of a deciding
threshold, the remainder proportional to the population; per-annotator shuffles;
the system's proposed label written to a withheld key file. Sampling weights per
mode are recorded in `manifest.json` so population estimates can be recovered.

`scripts/annotate.py` — local offline server, one unit at a time, atomic
full-file flush after every change, an explicit field allowlist, and a guard that
refuses to open the withheld key. `--validate` checks row count, id uniqueness,
label/confidence validity, sheet checksum, unit-content preservation and
retrieved-context completeness.

Labels are chosen by the three-step decision procedure in
`docs/ANNOTATION_GUIDELINES.md`: (1) answerability, (2) did the evidence reach
the system, (3) answer quality — with the explicit rule that a correct answer
produced without the gold evidence in context is a retrieval failure.

## 7. Failure taxonomy — 9 categories (`src/evaluation/taxonomy.py`)

| Category | Stage attribution | When it applies |
|---|---|---|
| `ok` | none | Answer conveys the reference answer |
| `ok_abstained` | none | Declined on a question the corpus cannot answer — a success |
| `no_retrieval` | retrieval | Nothing was retrieved |
| `wrong_retrieval` | retrieval | Nothing retrieved contains the information needed |
| `hallucination` | generation | Specific invented content present in neither context nor reference |
| `incorrect_answer` | generation | Asserts something different from the reference |
| `partial_answer` | generation | Part of the reference present, something the reference states missing |
| `refusal_when_answerable` | generation | Declined although the evidence was present |
| `answered_when_unanswerable` | generation | Substantive answer to a question the corpus cannot answer |

Thresholds are versioned (`v2.0`, fingerprint `4672f4ea2b70`) and every row
records the rule that fired (`R3`, `R4`, `R6`, `R8`–`R11`) and the decision
features behind it.

## 8. Datasets

Annotation study: **QASPER dev** (CC BY 4.0), 300 questions over 111 documents,
2,272 chunks; 200 units sampled for annotation.

Retrieval study: QASPER, Natural Questions (CC BY-SA 3.0), HotpotQA (CC BY-SA
4.0), 2WikiMultihopQA (Apache-2.0). Corpora are not redistributed; loaders,
checksums and licence metadata are committed.

## 9. Experimental protocol

See `docs/paper/experimental_setup.md`. In brief: chunk size 256 / overlap 32,
top-k 5, embedder `sentence-transformers/all-MiniLM-L6-v2`, generator
`MockExtractiveLLM` (deterministic extractive control, no LLM call), evidence
mode `any_sufficient` for single-hop and `all_required` for multi-hop.

## 10. Final results (verified against source files)

**Taxonomy vs the 200-unit reference set** (`final_evaluation.json`):

| Variant | Accuracy | Macro F1 | Cohen's kappa |
|---|---|---|---|
| Document-gated (`failure_mode_v2`) | 0.7400 | 0.6223 | 0.5728 |
| **Evidence-gated (`failure_mode_evidence`)** | **0.8050** | **0.6295** | **0.6305** |

Paired over the same 200 units: 22 units only the evidence-gated variant labels
correctly, 9 only the document-gated one, 139 both, 30 neither; **exact McNemar
p = 0.0294** (31 discordant pairs).

**Context integrity** (`TRUNCATION_AUDIT.json`): 1000 retrieved chunks audited,
941 cut at the 600-character display limit, 941 recovered, 1000/1000 complete
after rebuild, 0 unreconstructable; characters visible to the annotator
588,671 → 1,163,638.

**Annotation agreement**: two independent passes over the *truncated* package
reached kappa 0.8365 (92.5% raw, 15 disagreements adjudicated). The full-context
reference set agrees with those passes at kappa 0.7766 / 0.8100 and with the
adjudicated set at 0.8710; with the 22 human-labelled units it agrees on 20/22
(90.9%, kappa 0.7412 — n=22, below the repository's `MIN_N_FOR_INFERENCE = 30`).

**Retrieval decomposition** (`docs/EXPERIMENTS.md`, `results/`): QASPER
A 0.441 / B 0.441 / C 0.276 (granularity 16.6 pp, p = 7.1e-15); NQ 0.997 / 0.997
/ 0.730 (26.7 pp, p = 1.7e-24); HotpotQA 0.993 / 0.507 / 0.507 (quantifier
48.7 pp, p = 2.1e-22); 2WikiMultihopQA quantifier 64.7 pp (p = 1.3e-29),
granularity 1.3 pp (p = 0.5, not significant).

Full tables: `docs/paper/results.md` and `docs/paper/TABLES.md`.

## 11. Main findings

1. **Evidence gating matches independent judgement better.** +6.5 points of
   accuracy, +0.058 kappa, significant under an exact paired test — measured, not
   argued.
2. **The misattribution is concentrated and directional.** Of 30 units the
   reference calls `wrong_retrieval` but the document-gated taxonomy charges to
   generation, **22 had no gold evidence retrieved at all** (`evidence_status =
   none`). No unit moves the other way for that reason.
3. **Annotating on truncated context biases labels toward retrieval.** Restoring
   the full chunks moved 13 of 200 labels, 10 of them `wrong_retrieval` →
   `incorrect_answer`; none moved in the opposite direction.
4. **The rare generation categories are the taxonomy's weak point.**
   `partial_answer` is predicted 18 times against a support of 3; `hallucination`
   is predicted 8 times against a support of 0.
5. **Abstention detection is exact.** All 9 unanswerable units are labelled
   `answered_when_unanswerable` by both the reference and the taxonomy (P = R =
   F1 = 1.00).

## 12. Limitations

See `docs/paper/limitations.md` for the full list. The three that most constrain
what can be claimed:

- The 200-unit reference set is **model-generated**, not a human annotation
  pass; only 22 units in the repository carry human labels, and no
  inter-annotator agreement exists *for the full-context package*.
- The annotated run uses a **deterministic extractive control**, not a language
  model, so `hallucination` has zero support by construction of that run.
- The annotation study covers **one corpus** (QASPER) and **one configuration**
  (k=5, 256-token chunks, MiniLM).

## 13. Reproducibility

Every number above is regenerated by the commands in
`docs/paper/reproducibility.md`. Test suite: **466 tests**, 80% line coverage,
`ruff` clean.
