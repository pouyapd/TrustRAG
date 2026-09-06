# What Span-Level Evidence Evaluation Buys, and What It Costs: A Measurement-Validity Study of RAG Failure Attribution

**Pouya Bathaei Pourmand** · University of Genoa
Repository: https://github.com/pouyapd/TrustRAG

---

## Abstract

Retrieval-augmented generation is usually evaluated with document-level retrieval
metrics, and recent work has shown these overstate success on long documents, where
the answer-bearing passage inside a retrieved document is often missed. We take that
finding as given and ask the follow-up question: what does moving to span-level
evidence evaluation actually buy, what does it cost, and how far can its own gold
standard be trusted? Using four public QA corpora, a dense and a lexical retriever,
a nine-category failure taxonomy computed under two interchangeable retrieval gates,
and a 200-unit human annotation study with a second review pass, we report three
results. First, a retriever-ranking inversion we initially reported does not exist: it
was an artefact of an evidence-mode defect in our own BM25 baseline, and with the
defect fixed BM25 leads the dense retriever at *both* granularities on QASPER
(0.528/0.321 vs 0.441/0.276), with no significant inversion on any corpus, retrieval
depth or chunk size we tested. Second, gating failure attribution on
span-level evidence agrees with human judgement better than gating on document-level
retrieval (accuracy 0.700 vs 0.600, κ 0.437 vs 0.375; paired 22 vs 2,
p < 0.0001), but only the retrieval-side categories are reliable — generation-side
classification fails (`ok` recall 0.094, `incorrect_answer` precision 0.030). Third,
the span-based gold standard is itself incomplete: on units the span rule calls
retrieval failures, two independent proxies agree the answer is present in the
retrieved text in 7.5% of cases and disagree on a further 65%, which no automated
signal resolves. We conclude that span-level evidence evaluation is worth adopting
for retrieval-stage attribution and is not yet trustworthy as a gold standard for
generation-stage attribution.

## 1. Introduction

A RAG system fails for distinct reasons that call for distinct fixes: the retriever
returns nothing useful; it returns the right document but not the passage carrying
the answer; the generator is given adequate evidence and answers wrongly anyway. An
aggregate score compresses these into one number.

Prior work has established that document-level retrieval metrics miss the second case.
*Decomposing Retrieval Failures in RAG for Long-Document Financial QA*
(arXiv:2602.17981) evaluates retrieval at document, page and chunk level and names the
failure mode directly. We do not re-claim that finding. Instead we ask what follows
from it, and we report where the span-level alternative itself breaks down.

**Contributions.**

1. A negative result: the document/span choice does **not** reverse retriever
   comparisons in any configuration we tested, contradicting our own earlier
   report and correcting it (§5.1).
2. A human-validated measurement of how the choice of retrieval gate re-assigns
   failures across a taxonomy, with an explicit account of which categories the
   validation covers and which it does not (§7).
3. A quantified account of **gold-span under-coverage** — the extent to which the
   span-based gold standard reports retrieval failures that the retrieved text does
   not support (§8).
4. A replication, at n = 150 and one reader, of the counterfactual evidence-repair
   result of arXiv:2608.08944 (§6).

We claim no novelty for evidence-aware evaluation as an idea, for the failure
taxonomy, or for the oracle-evidence experiment; see `literature_review.md`.

## 2. Related work

Covered in full in `docs/paper/literature_review.md`, which includes a nine-row
comparison table. In brief: multi-granularity retrieval evaluation is established
(arXiv:2602.17981; Dense X Retrieval, EMNLP 2024); RAG failure taxonomies exist at
greater scope than ours (TrustNLP 2026, 33 modes across 7 stages); counterfactual
evidence intervention has been run at 70× our scale with matched sham controls
(arXiv:2608.08944, 32.8% repair — our replication finds 32.1%); and evidence
sufficiency is used elsewhere for compression (ECoRAG) and abstention (SURE-RAG)
rather than for attribution. The gap we address is not the existence of the
document/span distinction but its consequences and its limits.

## 3. Problem formulation

Let a question *q* have gold supporting spans *G* = {*g*₁…*g*ₙ}, each a half-open
character interval in a source document. A retriever returns chunks *R*, each also
carrying a character interval. Three definitions of retrieval success:

- **A (document, any):** some *r* ∈ *R* comes from a document containing some *g*.
- **B (document, quantified):** every document required by *q* is represented in *R*.
- **C (span, quantified):** for every required *g*, some *r* ∈ *R* overlaps it:
  `min(g_end, r_end) − max(g_start, r_start) > 0`.

`C ≤ B ≤ A` holds by construction. A→B isolates a **quantifier** effect (multi-hop
questions needing all documents); B→C isolates a **granularity** effect (long
documents where the span is a small part). The decomposition is our analytical
contribution; the underlying observation is not.

## 4. Methodology

**Offset-carrying pipeline.** Chunks are sliced from source documents, never
re-decoded, so `document[start:end] == chunk.text` holds by construction and is
property-tested. Offsets survive chunker → vector store → retrieval → stored record,
which is what makes span coverage computable after the fact by interval arithmetic
rather than string search.

**Taxonomy and the two gates.** Nine categories (`ok`, `ok_abstained`,
`no_retrieval`, `wrong_retrieval`, `hallucination`, `incorrect_answer`,
`partial_answer`, `refusal_when_answerable`, `answered_when_unanswerable`), assigned
by ordered rules with versioned thresholds. Exactly one rule, R4, decides whether a
row is charged to retrieval, and it reads one boolean. The document-gated variant
binds that boolean to A; the evidence-gated variant binds it to C. Everything else is
identical, and both labels are written to every row, so the comparison is on
identical retrieval output at zero additional inference cost.

**Annotation protocol.** 200 units sampled from a finished 300-question run under a
recorded seed, stratified with a floor per proposed failure mode and 25% of the budget
on rows near a deciding threshold. Annotators work from the question, gold evidence,
full retrieved context and system answer, blind to the system's proposed label,
through a three-step procedure: answerability → did the evidence reach the system →
answer quality.

## 5. Experimental setup and retrieval results

QASPER dev (290 answerable questions, 111 documents, 2,272 chunks), Natural Questions
validation (300), HotpotQA distractor (150), 2WikiMultihopQA (150). Chunk size 256,
overlap 32, top-k 5, `all-MiniLM-L6-v2`; BM25 (Okapi, k1 = 1.5, b = 0.75) over
identical chunks. Generator for the annotated run is a deterministic extractive
control; §6 uses Qwen2.5-0.5B-Instruct.

| Corpus | n | A | B | C | quantifier A→B | granularity B→C |
|---|---:|---:|---:|---:|---:|---:|
| QASPER dev | 290 | 0.441 | 0.441 | 0.276 | 0.0 pp | 16.6 pp (p = 7.1e-15) |
| Natural Questions | 300 | 0.997 | 0.997 | 0.730 | 0.0 pp | 26.7 pp (p = 1.7e-24) |
| HotpotQA | 150 | 0.993 | 0.507 | 0.507 | 48.7 pp (p = 2.1e-22) | 0.0 pp |
| 2WikiMultihopQA | 150 | — | — | — | 64.7 pp (p = 1.3e-29) | 1.3 pp (n.s.) |

The two effects are near-orthogonal: each is null on the corpus where the other
dominates.

### 5.1 A retriever-ranking inversion that turned out not to exist

An earlier version of this work reported that the document/span choice reverses the
BM25-versus-dense comparison on QASPER, and made that the headline contribution. **It
was an artefact of a defect in our own BM25 baseline**, found during a later audit and
corrected here.

The defect: QASPER and Natural Questions declare `any_sufficient` evidence mode — one
covered gold span suffices — but the BM25 script hard-coded `all_required`, demanding
every span. On QASPER 51% of questions carry more than one span, so BM25's span
coverage was systematically under-reported (0.183 instead of 0.321) while the dense
pipeline used the correct mode. The comparison was not like-for-like.

With the evidence mode applied consistently:

| Corpus | A dense | A BM25 | C dense | C BM25 | paired at span level |
|---|---:|---:|---:|---:|---|
| QASPER dev | 0.441 | **0.528** | 0.276 | **0.321** | BM25 40 vs dense 27, p = 0.142 (n.s.) |
| Natural Questions | **0.997** | 0.977 | **0.730** | 0.643 | dense 53 vs BM25 27, p = 0.0049 |
| HotpotQA | **0.993** | 0.927 | **0.507** | 0.420 | dense 36 vs BM25 23, p = 0.118 (n.s.) |

**No inversion occurs on any corpus.** On QASPER BM25 leads at both granularities; on
NQ and HotpotQA the dense retriever leads at both. We further checked five retrieval
depths (k = 1, 3, 5, 10, 20) and three chunk sizes (128/256/512) on QASPER: the
ordering is stable in all eight configurations, with BM25 ahead at both levels every
time.

A within-document diagnostic explains why an inversion was implausible to begin with.
Conditional on reaching a gold document, BM25 covers the span 60.8% of the time
(93/153) and the dense retriever 62.5% (80/128) — statistically indistinguishable
localisation. The retrievers differ in how often they reach the right document, not in
where they land inside it, so the two granularities rank them the same way.

We report this at length because the erroneous version was published to the repository
and because the failure mode is instructive: an evidence-mode mismatch between a
baseline and the system it is compared against is invisible in aggregate numbers and
changes the headline conclusion. The regression guard is now in the script's own
docstring and the evidence-mode distribution is recorded in every output file.

## 6. Oracle-evidence control (replication)

Every question answered twice by the same generator with the same prompt and
decoding; only the context differs (retrieved chunks vs the gold spans verbatim).
Within-question pairing removes question difficulty and generator identity as
confounds. n = 150, Qwen2.5-0.5B-Instruct, correctness = all reference key facts
present.

| Stratum | n | retrieved | oracle | difference | p |
|---|---:|---:|---:|---:|---:|
| Evidence complete under retrieval | 46 | 0.065 | 0.174 | +10.9 pp | 0.125 (n.s.) |
| Document retrieved, span missing | 26 | 0.000 | 0.231 | +23.1 pp | 0.031 |
| Nothing from any gold document | 78 | 0.000 | 0.321 | +32.1 pp | 6.0e-08 |
| **Overall** | **150** | **0.020** | **0.260** | **+24.0 pp** | **2.8e-10** |

The middle row is the argument: those 26 questions are scored as retrieval *successes*
by a document-level metric, the model answered none correctly, and supplying the
actual span repairs 23%.

**This is a replication.** arXiv:2608.08944 reports 32.8% repair from support addition
over 11,105 failures with four readers and matched sham controls; we find 32.1% on the
comparable stratum from n = 78 with one reader and no sham control. Our contribution
here is corroboration at small scale, not a new finding. The internal control that the
complete-evidence stratum gains least (+10.9 pp, n.s.) argues against the gain being a
pure context-length artefact, but without a sham condition we cannot exclude it.

## 7. Human validation

Two human passes by one annotator. The original pass labelled all 200 units; an audit
against the written procedure flagged 43 as conflicting with an explicit rule; the
annotator re-reviewed those 43 on full context, changing 36 labels and upholding 7.
The final dataset takes the original label for the 157 unflagged units and the review
decision for the 43, with a per-unit provenance chain.

**The review protocol creates dependence.** The annotator was told which units to
re-examine and why, and the changes moved toward what the guidelines prescribe. The
second pass is therefore not independent of the framework being tested. We report the
result as agreement with a guided expert reading, not as independent validation.

| Variant | Accuracy | 95% CI | Macro F1 | κ |
|---|---:|---:|---:|---:|
| Document-gated | 0.6000 | 0.531–0.665 | 0.4764 | 0.3752 |
| **Evidence-gated** | **0.7000** | 0.633–0.759 | **0.4819** | **0.4371** |

Paired: 118 both correct, 22 only evidence-gated, 2 only document-gated, 58 neither;
exact McNemar p < 0.0001.

Per class, evidence-gated:

| Class | Support | P | R | F1 |
|---|---:|---:|---:|---:|
| `wrong_retrieval` | 136 | 0.917 | 0.897 | 0.907 |
| `answered_when_unanswerable` | 9 | 1.000 | 1.000 | 1.000 |
| `partial_answer` | 22 | 0.385 | 0.227 | 0.286 |
| `ok` | 32 | 0.500 | 0.094 | 0.158 |
| `incorrect_answer` | 1 | 0.030 | 1.000 | 0.059 |
| `hallucination` | 0 | 0.000 | — | — |

**Retrieval-side attribution is validated; generation-side classification is not.**
The taxonomy predicts `incorrect_answer` 33 times where the human used it once, and
recovers 3 of 32 answers the human called `ok`.

Agreement with an automated LLM reference pass over the same units is markedly higher
(0.805 accuracy, κ 0.631) than agreement with humans. Two automated readings share
failure directions; the human figure is the one we report.

### 7.1 Threshold ablation

To test whether the generation-side failure is a threshold artefact, we split the 200
units 50/50 under a fixed seed, grid-searched 144 configurations on the tuning half
only, and scored the winner once on the held-out half.

| Gate | Held-out accuracy | Macro F1 | κ |
|---|---:|---:|---:|
| Document-gated, shipped thresholds | 0.6800 | 0.5124 | 0.4533 |
| Document-gated, tuned | 0.6900 | 0.5108 | 0.4630 |
| Evidence-gated, shipped thresholds | 0.7300 | 0.5132 | 0.4810 |
| **Evidence-gated, tuned** | **0.7500** | **0.5490** | **0.5117** |

Tuning gives a modest gain for the evidence gate and essentially none for the document
gate, and the gate advantage survives tuning both. So the generation-side weakness is
only partly a threshold problem; the rules themselves are the larger part.

## 8. How far the span-based gold standard can be trusted

The 11 units where the annotator kept an answer-quality label at high confidence
despite zero span coverage prompted a check of the gold standard itself. Over all 133
answerable units with zero gold-span coverage, we compute two proxies: lexical
presence of the reference answer's content words in the retrieved context, and maximum
cosine similarity between the reference answer and any retrieved sentence.

| Bucket | n | Share |
|---|---:|---:|
| Both agree the answer is absent — span rule correct | 36 | 27.1% |
| Both agree the answer is present outside the gold span — **span rule wrong** | 10 | 7.5% |
| Semantically close, lexically different | 8 | 6.0% |
| Lexical overlap only | 21 | 15.8% |
| Signals disagree or both mid-range | 58 | 43.6% |

A lexical-only reading suggests 23.3% under-coverage; requiring both proxies to agree
puts the confident lower bound at 7.5%. **Neither is the answer.** 65.4% of these units
are unresolved by any automated signal, and settling them needs entailment judgements
from a human, which we have not performed. What we can say is that the span rule
over-charges retrieval by somewhere between 7.5% and roughly a third of the affected
units, and that this bounds any claim built on span-level ground truth.

## 9. Error analysis

Of the 30 units the human calls a retrieval failure but the document gate charges to
generation, 22 carry `evidence_status = none` — the pipeline had already recorded that
nothing usable arrived. The evidence gate recovers those at a cost: `wrong_retrieval`
precision falls from 1.000 to 0.917.

The generation-side failures are systematic, not noisy. `partial_answer` fires on
partial token overlap where the guidelines require part of *what the reference states*;
`hallucination` cannot be validated because the annotated run uses an extractive
control that cannot invent content; `ok` requires all reference key facts where the
human credits any accepted alternative answer.

## 10. Discussion

Span-level evidence evaluation earns its place for retrieval-stage attribution: it
agrees better with human judgement, it recovers misattributed units through a
mechanism we can trace, and it can change which retriever a comparison recommends. It
does not currently support generation-stage attribution, both because the taxonomy's
generation rules are weak and because the gold standard underneath is incomplete in a
direction that inflates retrieval blame.

The practical recommendation is narrow: report span-level coverage beside
document-level recall, use it to attribute retrieval failures, and do not treat it as
ground truth for anything downstream without checking annotation completeness first.

## 11. Limitations

- **One annotator, and a guided review.** No inter-annotator agreement exists. The
  second pass was directed by an audit of the same guidelines being tested.
- **The span gold standard is incomplete** by 7.5–23% of affected units (§8), with
  65% unresolved.
- **Generation-side categories are unvalidated**; three have zero support.
- **One corpus for the human study** (QASPER dev), one configuration (k = 5, 256
  tokens, MiniLM).
- **Small generators** (0.5B, 0.36B); no frontier model was available.
- **No sham control** in the oracle experiment, unlike the work it replicates.
- **Contamination is mitigated, not eliminated** — NQ and both multi-hop corpora
  derive from Wikipedia.
- **Targeted, not systematic, literature review** (§2).
- 22 of the 200 units carry labels from a pilot on a truncated package; 7 were
  re-reviewed on full context, 15 were not flagged.

## 12. Reproducibility

Every number above maps to a script, a command and an output file in
`docs/paper/REPRODUCIBILITY.md`. The retrieval and evidence experiments run offline
with no API key. `reports/` is gitignored, so annotation artifacts are produced
locally rather than shipped.

## 13. Conclusion

Taking as given that document-level retrieval metrics overstate success, we asked what
span-level evaluation buys and costs. It buys better agreement with human judgement on
retrieval attribution and the ability to distinguish retrievers that a document-level
metric ranks the other way. It costs precision on the retrieval class, and it rests on
a gold standard we show to be incomplete. The honest summary is that span-level
evidence is a better instrument for one specific job and not yet a sound ground truth
for the rest.

## References

Full bibliographic details in `literature_review.md`. Principal works:
arXiv:2602.17981; aclanthology.org/2026.trustnlp-main.27; arXiv:2608.08944;
arXiv:2408.02854; arXiv:2603.09891; arXiv:2504.15068; arXiv:2506.05167;
arXiv:2605.03534; aclanthology.org/2024.emnlp-main.845.
