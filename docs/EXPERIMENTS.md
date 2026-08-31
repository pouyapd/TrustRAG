# Experiments

Everything below was produced by the code in this repository against real
third-party corpora. Nothing is estimated, projected or illustrative.

**Read this first — what these runs can and cannot support.**

| Claim type | Status |
|---|---|
| Retrieval and evidence measurement | **Real.** Real corpora, real embeddings, real ChromaDB retrieval, real human-annotated gold spans. |
| Failure attribution between retrieval and generation | **Real for the retrieval side.** A row is charged to retrieval on evidence grounds that do not involve the generator. |
| Robustness to the embedding model | **Measured.** Four models from three training lineages, everything else held constant. |
| Robustness to retrieval depth | **Measured.** k ∈ {1, 3, 5, 10, 20}, retrieved natively at each depth. |
| Multi-hop evidence behaviour | **Real, and replicated.** HotpotQA distractor and 2WikiMultihopQA, 150 questions each, all `all_required`. |
| Generation-side behaviour of a real language model | **Limited.** Measured with small local open-weight models (0.36B and 0.5B). Enough to test whether evidence status predicts generation failure; **not** enough to characterise any deployed or frontier model. No hosted-API run was possible — no key was available. |
| Hallucination rates, faithfulness benchmarking, model comparison | **Not supported.** Nothing here licenses a claim about how often a model hallucinates. |
| Agreement between the taxonomy and human judgement | **Not measured.** The full protocol — package, guidelines, scoring — exists. **Zero human labels have been collected.** |

---

## Setup

The **default configuration**, used by every headline run. Each robustness
experiment varies exactly one of these and holds the rest fixed.

| | Default | Swept in |
|---|---|---|
| Retriever | ChromaDB, cosine similarity | — |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Result 7 (4 models) |
| Chunking | 256 tokens, 32 overlap, tiktoken `cl100k_base` | Result 2 (128/256/512) |
| top-k | 5 | Result 8 (1, 3, 5, 10, 20) |
| Generator | `MockExtractiveLLM` — deterministic extractive control | Result 11 (real LM) |
| Corpus | QASPER, NQ, HotpotQA | Result 9 (2WikiMultihopQA) |
| Taxonomy | v2.0 | — |

The generator copies the sentence from retrieved context with the greatest
overlap with the question. It bounds generation quality from below and makes
runs fully deterministic. Crucially, **it does not affect the retrieval or
evidence measurements**, which are what the headline result is about.

---

## Experimental matrix

Every run in the repository. "Held constant" is the important column: a
robustness claim is only as strong as the list of things that did *not* change.

| # | Experiment | Dataset | n | Embedder | Chunk | k | Generator | Varies | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `qasper_dev_300` | QASPER dev | 290 | MiniLM | 256/32 | 5 | extractive control | — | Headline: granularity effect |
| 2 | `nq_val_300_fixed` | NQ validation | 300 | MiniLM | 256/32 | 5 | extractive control | — | Headline: granularity effect |
| 3 | `hotpot_150` | HotpotQA distractor | 150 | MiniLM | 256/32 | 5 | extractive control | — | Headline: quantifier effect |
| 4 | `qasper_c128` / `qasper_c512` | QASPER dev | 290 | MiniLM | 128/32, 512/32 | 5 | extractive control | chunk size | Mechanism: gap tracks chunks per document |
| 5 | `twowiki_150` | 2WikiMultihopQA dev | 150 | MiniLM | 256/32 | 5 | extractive control | corpus | Replication: quantifier effect on a second multi-hop corpus |
| 6 | `*_emb_{minilm,mpnet,bge,e5}` | QASPER, HotpotQA | 290, 150 | **4 models** | 256/32 | 5 | extractive control | embedder | Robustness: is the effect a property of one model? |
| 7 | `*_topk_k{1,3,5,10,20}` | QASPER, NQ, HotpotQA, 2Wiki | 290–150 | MiniLM | 256/32 | **1–20** | extractive control | retrieval depth | Robustness: does the distinction survive realistic k? |
| 8 | `llm_*` | QASPER, HotpotQA | ≤150 | MiniLM | 256/32 | 5 | **real LM** | generator | Does evidence status predict generation failure? |

### Hypotheses and what would falsify them

| Hypothesis | Falsified by |
|---|---|
| H1 The granularity gap is not an artifact of one embedding model. | The gap vanishing, or reversing, under any embedder in experiment 6. |
| H2 The distinction survives realistic retrieval depth. | The gap reaching ~0 at any k a practitioner would deploy. |
| H3 The quantifier effect is a property of multi-hop questions, not of HotpotQA. | A near-zero A→B step on 2WikiMultihopQA. |
| H4 Evidence status predicts what a language model does. | Correctness and abstention being independent of evidence stratum. |
| H5 The taxonomy agrees with independent human judgement. | Low Cohen's kappa against adjudicated labels. |

H1-H3 are **confirmatory**: the mechanism was stated and the predictions made
before the runs. H4 is **exploratory** — the generation experiment is
underpowered by design and uses small local models. H5 is **not yet tested**;
see the taxonomy-validation section.

### What is held constant, and why it matters

The three robustness experiments each vary exactly one thing:

- **Embedder sweep**: same corpus, same questions, same documents, same chunking
  (256/32), same k (5), same metrics, same taxonomy. Only the embedding model
  changes. Asymmetric models are called with the query and passage prefixes they
  were trained with — omitting those would report a weaker model rather than a
  different one, and the conclusion would be about our plumbing.
- **Depth sweep**: same corpus, chunking and embedder. Only k changes. Every
  depth is retrieved *natively* — the query is re-issued at each k rather than a
  single deep ranking being truncated. Truncation was measured first and looked
  exactly equivalent on all three study corpora (160 query x depth comparisons,
  zero disagreements), but it is not guaranteed: the index is approximate, and
  near-tied neighbours can be ordered differently depending on how many results
  were requested. Re-querying costs seconds and removes the assumption.
- **Second multi-hop corpus**: identical pipeline, identical code path,
  identical evidence-mode handling. Only the corpus changes.

---

## Correction to an earlier framing

An earlier version of this document reported a single number per dataset: the
gap between the conventional document-level retrieval metric and the
span-level one. On HotpotQA that comparison changed **two** things at once —
the granularity (document to span) and the quantifier (any relevant document
to every required document) — and the resulting 48.7 pp was presented as a
granularity result.

It is not. Decomposing it shows the HotpotQA gap is **entirely** the
quantifier and **exactly zero** granularity. The corrected analysis below
separates the two, which changes the interpretation of one headline number and
leaves the other two unchanged. The measurements themselves did not change;
only their attribution to a mechanism did.

| | old framing | corrected |
|---|---|---|
| QASPER 16.6 pp | "granularity" | granularity 16.6, quantifier 0.0 — **unchanged** |
| NQ 26.7 pp | "granularity" | granularity 26.7, quantifier 0.0 — **unchanged** |
| HotpotQA 48.7 pp | "granularity" | **quantifier 48.7, granularity 0.0** — reattributed |

---

## Result 1 — Two distinct blind spots, each isolated

Three definitions of "retrieval succeeded", applied to the same stored records:

- **A — document-level, ANY.** Did any retrieved chunk come from any relevant
  document? This is the conventional metric.
- **B — document-level, quantified.** Under `all_required`, must every gold
  document be retrieved. Identical to A on single-hop data by definition.
- **C — span-level, quantified.** Did a retrieved chunk actually contain the
  gold span?

A→B isolates the **quantifier**; B→C isolates the **granularity**.

| | QASPER dev | NQ validation | HotpotQA |
|---|---|---|---|
| n | 290 | 300 | 150 |
| evidence mode | any_sufficient | any_sufficient | all_required |
| median chunks per gold document | 19 | 31 | **2** |
| A document, ANY | 0.441 | 0.997 | 0.993 |
| B document, quantified | 0.441 | 0.997 | **0.507** |
| C span, quantified | 0.276 | 0.730 | 0.507 |
| **quantifier A→B** | 0.0 pp | 0.0 pp | **48.7 pp** (p=2.1e-22) |
| **granularity B→C** | **16.6 pp** (p=7.1e-15) | **26.7 pp** (p=1.7e-24) | 0.0 pp |
| discordant (granularity) | 48 / 0 | 80 / 0 | 0 / 0 |

Two separate failures of conventional retrieval metrics, each demonstrated on
the data where it actually bites:

1. **Granularity blindness** — on long documents, retrieving the document is
   not retrieving the evidence. 16.6 and 26.7 pp.
2. **Quantifier blindness** — on multi-hop questions, retrieving *a* relevant
   document is counted as success when the question needs *all* of them.
   48.7 pp.

Neither is visible to the other's dataset, which is why one number per dataset
was the wrong summary.

---

## Result 2 — The granularity effect scales with chunks per document, as predicted

The mechanism makes a falsifiable prediction: a document that occupies one
chunk cannot show a granularity gap, and the gap should grow as a document
spans more chunks. Varying chunk size on QASPER (same corpus, same questions,
same retriever, n=290 throughout):

| chunk size | median chunks per gold document | granularity gap | McNemar *p* |
|---|---|---|---|
| 128 | 43 | **18.6 pp** | 1.1e-16 |
| 256 (reported) | 19 | **16.6 pp** | 7.1e-15 |
| 512 | 9 | **11.0 pp** | 4.7e-10 |
| HotpotQA paragraphs | 2 | **0.0 pp** | — |

Monotonic across a fourfold range of chunk sizes and across datasets, exactly
as the mechanism predicts. This answers the obvious reviewer objection
directly: the gap **is** sensitive to chunk size, in a predictable and
explained way, and it does not disappear at any realistic setting — at 512
tokens it is still 11 pp with *p* = 4.7e-10.

It also bounds the claim honestly. The gap is a property of *corpus structure
relative to chunk size*, not a universal constant. Systems that retrieve
paragraph-sized documents have no granularity problem; systems that retrieve
long documents have a large one.

---

## Result 3 — Failure attribution changes materially

Same runs, same rows; only the attribution rule differs.

| Attributed to | QASPER doc / evidence | NQ doc / evidence | HotpotQA doc / evidence |
|---|---|---|---|
| retrieval | 162 / **210** | 1 / **81** | 1 / **74** |
| generation | 131 / 74 | 219 / 155 | 111 / 49 |
| abstention | — / 10 | — / — | — / — |
| none | 7 / 6 | 80 / 64 | 38 / 27 |

On NQ and HotpotQA a document-level reading attributes a single failure out of
300 and 150 respectively to retrieval. An engineer acting on that report would
tune prompts; the evidence-aware report says the retriever is the binding
constraint.

## Result 3b — The failure taxonomy inherits the same blind spot

The v2 taxonomy decides `wrong_retrieval` on whether a relevant *document* was
retrieved. A row whose document arrived but whose gold span did not passes that
gate and is labelled by answer quality instead — as a generation failure.
Re-running the identical rules with an evidence-level gate:

| | `wrong_retrieval` v2 → evidence | `incorrect_answer` v2 → evidence | rows moved |
|---|---|---|---|
| QASPER n=300 | 162 → **210** | 88 → **51** | 46 (15%) |
| NQ n=300 | **1** → **81** | 174 → **120** | 66 (22%) |
| HotpotQA n=150 | **1** → **74** | 102 → **45** | 62 (41%) |

On NQ the document-gated taxonomy reports a single retrieval failure in 300
questions; the evidence gate reports 81. Both labels are emitted on every row.
`failure_mode_v2` is frozen for reproducibility; `failure_mode_evidence` is the
one consistent with the attribution and the one new analysis should use.

---

## Result 4 — Dataset properties measured, not assumed

| | QASPER dev | NQ validation | HotpotQA |
|---|---|---|---|
| Questions loaded | 400 | 300 | 150 |
| Chunks indexed | 2,272 | 12,245 | 1,571 |
| Documents | 128 papers | 297 pages | 1,491 paragraphs |
| Mean document length | 21,831 chars | 37,181 chars | ~1,100 chars |
| Answerable / unanswerable | 386 / 14 | 300 / 0 | 150 / 0 |
| Questions with >1 reference answer | — | **49%** | — |
| Mean gold spans per question | 2.16 | 1.0 | 2.0+ |
| Questions requiring >1 document | 0 | 0 | **150 (all)** |
| Duplicate page content under distinct ids | — | **0** | — |
| Dataset validation failures | **0 / 400** | **0 / 300** | **0 / 150** |
| Corpus offset mismatches at index time | **0** | **0** | **0** |

Every gold span in all three datasets resolves exactly against the document
text the loader built — 0 validation failures across 850 questions — and
`build_corpus` re-verified `document[start:end] == chunk.text` for every chunk
at index time, including 2,272 chunks over 2.4 M characters of QASPER, with 0
mismatches. That is the precondition for every evidence number above.

**NQ loader census (300 questions emitted):** 153 items skipped as page-scoped
nulls — correctly *not* treated as corpus-scoped unanswerables — and 90 skipped
for having no extractable short answer.

**QASPER evidence census (dev split):** 384 of 2,808 evidence strings (13.7%)
have no exact position in the paper body. 253 are `FLOAT SELECTED:` figure and
table captions, which QASPER stores outside `full_text`. A question supported
only by those is not answerable from the text corpus and is excluded rather
than reported as a retrieval failure no retriever could have avoided.

---

## Result 5 — The gap is not an artifact of the overlap threshold

Evidence alignment has one free parameter: `min_overlap_chars`, how many
characters a chunk must share with a gold span to count as carrying it. The
reported runs use 1, the most permissive value. Re-scoring the same stored
records at stricter thresholds (no model calls):

| min_overlap_chars | QASPER | NQ | HotpotQA |
|---|---|---|---|
| **1 (reported)** | **16.6 pp** | **26.7 pp** | **48.7 pp** |
| 50 | 16.6 pp | 28.3 pp | 49.3 pp |
| 200 | 19.3 pp | 36.3 pp | 96.0 pp * |

The most permissive setting gives the **smallest** gap in every dataset, so all
reported figures are conservative lower bounds. (* the HotpotQA figure at 200
characters is degenerate — gold spans there are single sentences often shorter
than 200 characters — and is shown for completeness, not used as a result.)

---

## Result 6 — The NQ result is insensitive to a loader defect found afterwards

The NQ parquet loader initially read `yes_no_answer` as a string, but the
HuggingFace distribution encodes it as a ClassLabel index, so every yes/no
question was silently dropped. The run was repeated with the fix. 17 yes/no
questions entered the sample and the measurement did not move:

| | before fix | after fix |
|---|---|---|
| Document-level | 0.9967 | 0.9967 |
| Span-level | 0.730 | 0.730 |
| Gap | 26.67 pp | 26.67 pp |
| McNemar *p* | 1.65e-24 | 1.65e-24 |
| Question types | factoid 242, list 58 | factoid 229, list 54, **yes_no 17** |

Reported NQ figures come from the corrected run. That they are identical to
three decimals is a robustness observation, not a coincidence: the defect
changed which questions were sampled, not how retrieval was measured.

---

## Result 7 — The granularity effect is not a property of one embedding model

**Hypothesis H1.** If the gap were an artifact of MiniLM's particular geometry,
a differently trained retriever should shrink or reverse it.

Four models, three training lineages. Corpus, questions, documents, chunking
(256/32), retrieval depth (k=5), metrics and taxonomy are all held constant;
only the embedder changes. The asymmetric models are called with the query and
passage prefixes they were trained with — omitting those would report a weaker
model rather than a different one.

<!-- BEGIN generated: embedders -->

**QASPER — granularity B→C.** Corpus, questions, chunking (256/32) and k=5 held constant; only the embedder changes.

| Embedder | Family | dim | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | sentence-transformers | 384 | 0.445 | 0.445 | 0.279 | **16.6 pp** | 48/0 | 7.1e-15 |
| `all-mpnet-base-v2` | sentence-transformers | 768 | 0.379 | 0.379 | 0.234 | **14.5 pp** | 42/0 | 4.5e-13 |
| `bge-small-en-v1.5` | BAAI BGE | 384 | 0.476 | 0.476 | 0.293 | **18.3 pp** | 53/0 | 2.2e-16 |
| `e5-small-v2` | Microsoft E5 | 384 | 0.462 | 0.462 | 0.303 | **15.9 pp** | 46/0 | 2.8e-14 |

**HotpotQA — quantifier A→B.** Corpus, questions, chunking (256/32) and k=5 held constant; only the embedder changes.

| Embedder | Family | dim | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | sentence-transformers | 384 | 0.993 | 0.507 | 0.507 | **48.7 pp** | 73/0 | 2.1e-22 |
| `all-mpnet-base-v2` | sentence-transformers | 768 | 1.000 | 0.527 | 0.527 | **47.3 pp** | 71/0 | 8.5e-22 |
| `bge-small-en-v1.5` | BAAI BGE | 384 | 0.993 | 0.707 | 0.707 | **28.7 pp** | 43/0 | 2.3e-13 |
| `e5-small-v2` | Microsoft E5 | 384 | 0.993 | 0.727 | 0.727 | **26.7 pp** | 40/0 | 1.8e-12 |
<!-- END generated: embedders -->

**The effect survives every model.** On QASPER the granularity gap ranges from
14.5 pp to 18.3 pp, every one significant at p < 5e-13, and the discordance is
strictly one-directional in all four (48/0, 42/0, 53/0, 46/0): no question ever
achieves span-level success without document-level success.

Three things worth noting rather than glossing:

- **The stronger model is not the better retriever here.** MPNet has twice the
  dimension and five times the parameters of MiniLM, yet retrieves the relevant
  document *less* often on QASPER (A = 0.379 vs 0.445). Scientific text is not
  what these general-purpose models are strongest on, and capacity does not
  translate into domain fit. It also means the sweep is not secretly a quality
  ranking.
- **The granularity gap does not track A.** BGE has both the highest A (0.476)
  and the largest gap (18.3 pp); MPNet has the lowest A (0.379) and nearly the
  smallest (14.5 pp). Retrieving more documents does not, by itself, close the
  distance between a document and the evidence inside it.
- **The quantifier gap is far more embedder-sensitive than the granularity gap,
  and this is the most interesting result in the sweep.** On HotpotQA it ranges
  from 48.7 pp (MiniLM) down to 26.7 pp (E5) — close to half. The two asymmetric
  instruction-trained retrievers, BGE and E5, are markedly better at getting
  *all* the required documents into the window: B rises from 0.507 to 0.707 and
  0.727 while A stays pinned at 0.993 for all four.

  So the effects behave differently under a change of retriever. Granularity
  blindness is stable at 14.5-18.3 pp — it is a property of chunking against
  document length, and a better encoder does not fix it. Quantifier blindness is
  substantially *mitigated* by a better multi-hop retriever, though not
  eliminated: even E5 leaves a 26.7 pp gap (p = 1.8e-12, 40/0 discordant), and A
  reports 0.993 throughout, so the conventional metric is equally blind to the
  difference between the best and worst configuration here. A practitioner's
  takeaway is that switching retriever is a real lever for multi-hop coverage
  and is not a lever for span coverage — which is precisely the kind of
  distinction a single aggregate metric cannot express.

---

## Result 8 — Both effects survive realistic retrieval depth, and the conventional metric stops discriminating

**Hypothesis H2.** A sceptical reading of the headline result is "just retrieve
more chunks". This tests it directly.

Every depth is retrieved **natively** — the query is re-issued at each k rather
than one deep ranking being truncated. Truncation was measured first and looked
exactly equivalent on all three study corpora (160 query x depth comparisons,
zero disagreements), but it is not guaranteed: the index is approximate, and
near-tied neighbours can be ordered differently depending on how many results
were requested. Re-querying costs seconds and removes the assumption.

<!-- BEGIN generated: topk -->

**QASPER — granularity B→C.** Every depth retrieved natively; corpus, chunking and embedder held constant.

| k | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|
| 1 | 0.293 | 0.293 | 0.090 | **20.3 pp** | 59/0 | 3.5e-18 |
| 3 | 0.397 | 0.397 | 0.197 | **20.0 pp** | 58/0 | 6.9e-18 |
| 5 | 0.445 | 0.445 | 0.279 | **16.6 pp** | 48/0 | 7.1e-15 |
| 10 | 0.490 | 0.490 | 0.362 | **12.8 pp** | 37/0 | 1.5e-11 |
| 20 | 0.562 | 0.562 | 0.417 | **14.5 pp** | 42/0 | 4.5e-13 |

**Natural Questions — granularity B→C.** Every depth retrieved natively; corpus, chunking and embedder held constant.

| k | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|
| 1 | 0.950 | 0.950 | 0.377 | **57.3 pp** | 172/0 | 3.3e-52 |
| 3 | 0.980 | 0.980 | 0.617 | **36.3 pp** | 109/0 | 3.1e-33 |
| 5 | 0.997 | 0.997 | 0.730 | **26.7 pp** | 80/0 | 1.7e-24 |
| 10 | 1.000 | 1.000 | 0.850 | **15.0 pp** | 45/0 | 5.7e-14 |
| 20 | 1.000 | 1.000 | 0.923 | **7.7 pp** | 23/0 | 2.4e-07 |

**HotpotQA — quantifier A→B.** Every depth retrieved natively; corpus, chunking and embedder held constant.

| k | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|
| 1 | 0.787 | 0.000 | 0.000 | **78.7 pp** | 118/0 | 6.0e-36 |
| 3 | 0.940 | 0.407 | 0.407 | **53.3 pp** | 80/0 | 1.7e-24 |
| 5 | 0.993 | 0.507 | 0.507 | **48.7 pp** | 73/0 | 2.1e-22 |
| 10 | 1.000 | 0.727 | 0.727 | **27.3 pp** | 41/0 | 9.1e-13 |
| 20 | 1.000 | 0.833 | 0.833 | **16.7 pp** | 25/0 | 6.0e-08 |

**2WikiMultihopQA — quantifier A→B.** Every depth retrieved natively; corpus, chunking and embedder held constant.

| k | A | B | C | gap | discordant | McNemar p |
|---|---|---|---|---|---|---|
| 1 | 0.840 | 0.000 | 0.000 | **84.0 pp** | 126/0 | 2.4e-38 |
| 3 | 0.947 | 0.247 | 0.220 | **70.0 pp** | 105/0 | 4.9e-32 |
| 5 | 0.967 | 0.320 | 0.307 | **64.7 pp** | 97/0 | 1.3e-29 |
| 10 | 0.973 | 0.447 | 0.440 | **52.7 pp** | 79/0 | 3.3e-24 |
| 20 | 0.987 | 0.507 | 0.500 | **48.0 pp** | 72/0 | 4.2e-22 |
<!-- END generated: topk -->

Retrieving more does help, and the honest summary is that **the magnitude is
strongly k-dependent while the distinction is not**:

- **The gap shrinks but never closes.** On NQ it falls from 57.3 pp at k=1 to
  7.7 pp at k=20 — a large decay — yet the k=20 gap is still significant
  (p = 2.4e-07, 23/0 discordant). On QASPER it barely decays at all
  (20.3 → 14.5 pp). On 2WikiMultihopQA it is still **48.0 pp at k=20**.
- **The conventional metric saturates and becomes uninformative.** On NQ and
  HotpotQA, A reaches exactly 1.000 by k=10 and stays there. A metric with zero
  variance cannot rank systems, diagnose regressions, or tell an engineer
  anything. At NQ k=20 the document-level reading charges **0** failures to
  retrieval; the evidence-level reading still charges **23**.
- **k=1 on multi-hop data is a definitional artifact, not evidence.** A 2-hop
  question cannot have both required documents in a single retrieved slot, so
  B = 0.000 at k=1 follows from the pigeonhole principle. That row is reported
  for completeness and should carry no weight. The k=10 and k=20 rows, where
  there is ample room for every required document, are the informative ones.
- **Deeper retrieval is not free.** k=20 quadruples the context the generator
  must handle relative to k=5. "Just raise k" trades a retrieval problem for a
  context-length and precision problem, and the point of measuring evidence
  directly is to know which one you actually have.

---

## Result 9 — The quantifier effect replicates on a second multi-hop corpus

**Hypothesis H3.** The quantifier effect was measured on HotpotQA alone, which
made it a property of one dataset rather than of multi-hop questions.
2WikiMultihopQA is structurally comparable — ten context paragraphs, gold
evidence as `(title, sentence)` pairs — so the identical pipeline runs over it
unchanged and a difference in outcome is a difference in the data.

<!-- BEGIN generated: multihop -->

| | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| n | 150 | 150 |
| median chunks per gold document | 2 | 2 |
| evidence mode | all_required | all_required |
| A document, ANY | 0.993 | 0.967 |
| B document, quantified | **0.507** | **0.320** |
| C span, quantified | 0.507 | 0.307 |
| **quantifier A→B** | **48.7 pp** (p=2.1e-22) | **64.7 pp** (p=1.3e-29) |
| discordant pairs | 73/0 | 97/0 |
| granularity B→C | 0.0 pp (p=n/a) | 1.3 pp (p=0.5) |
| failures charged to retrieval, document-level | 1 | 5 |
| failures charged to retrieval, evidence-level | **74** | **104** |
<!-- END generated: multihop -->

**It replicates, and it is larger: 64.7 pp against HotpotQA's 48.7 pp**
(p = 1.3e-29, 97 discordant pairs, none in the opposite direction).

The larger effect has a mechanical explanation that was predicted from the
dataset's structure before the run: 28 of the 150 items are 4-hop, requiring
four distinct documents rather than two. The more documents a question requires,
the more often "any relevant document" and "every required document" disagree.

**The granularity effect is absent here, as predicted.** 1.3 pp, p = 0.5 — not
significant, and reported as a null result rather than dropped. 2Wiki documents
have a median length of 232 characters, roughly one chunk at `chunk_size=256`,
so retrieving the document essentially *is* retrieving the span. This is the
same null HotpotQA produces (0.0 pp) and for the same reason. Taken with
Result 2, where the granularity gap tracks chunks-per-document across a
fourfold change in chunk size, the two corpora act as the negative control the
mechanism predicts: **no chunking, no granularity effect.**

---

## Result 10 — Failure attribution across all four corpora

The consequence of the above, and the reason any of it matters to someone
operating a system:

<!-- BEGIN generated: attribution -->

| Corpus | n | retrieval (document-level) | retrieval (evidence-level) | change |
|---|---|---|---|---|
| QASPER | 290 | 162 | **210** | ×1.3 |
| Natural Questions | 300 | 1 | **81** | ×81.0 |
| HotpotQA | 150 | 1 | **74** | ×74.0 |
| 2WikiMultihopQA | 150 | 5 | **104** | ×20.8 |
<!-- END generated: attribution -->

On the three corpora where document-level retrieval looks nearly perfect, a
conventional reading charges almost nothing to retrieval and hands the entire
failure budget to the generator. QASPER is the informative exception: retrieval
there is visibly poor under *both* readings (162 vs 210), so the conventional
metric is not misleading in the same way — which is itself evidence that the
effect is about corpus structure rather than a universal correction.

---

## Result 11 — Evidence status predicts what a real language model does

**Hypothesis H4, exploratory.** Everything above uses a deterministic extractive
control, which cannot hallucinate, refuse, or be inconsistent. That control is
the right reproducible baseline, but it cannot answer the question that makes
evidence-aware retrieval matter: when the supporting passage never reaches the
generator, what does a *language model* actually do?

**Design.** Retrieval is not re-run. The stored records of `qasper_dev_300` are
replayed, the exact context each question received is rebuilt, and only the
generator is swapped. Corpus, questions, chunking, embedder and retrieved
context are therefore identical by construction. Each question is then assigned
to a stratum by the evidence status already computed for it — a property of the
run, not of the generator — and outcomes are reported per stratum.

**Two generators**, both greedy, both run locally on CPU: Qwen2.5-0.5B-Instruct
and SmolLM2-360M-Instruct — different organisations, different training data,
different tokenizers. No API key was available, so these are small open-weight
models rather than frontier ones. n=150 each, 0 generation errors.

**Qwen2.5-0.5B-Instruct** (mean latency 25.4 s/question):

| stratum | n | correct | abstained | answered |
|---|---|---|---|---|
| `COMPLETE` — every required gold span reached the generator | 44 | **18.2%** | 9.1% | 90.9% |
| `NONE_DOC_HIT` — the right document arrived, the span did not | 26 | **0.0%** | 0.0% | 100.0% |
| `NONE` — nothing from any gold document arrived | 75 | 1.3% | 4.0% | 96.0% |
| `no_gold_evidence` | 5 | 20.0% | 0.0% | 100.0% |

**SmolLM2-360M-Instruct** (mean latency 24.0 s/question):

| stratum | n | correct | abstained | answered |
|---|---|---|---|---|
| `COMPLETE` | 44 | **13.6%** | 2.3% | 97.7% |
| `NONE_DOC_HIT` | 26 | 3.8% | 0.0% | 100.0% |
| `NONE` | 75 | 2.7% | 0.0% | 100.0% |
| `no_gold_evidence` | 5 | 0.0% | 0.0% | 100.0% |

**Evidence status predicts correctness, on both models:**

| Generator | P(correct \| complete) | P(correct \| incomplete) | difference | permutation p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.182 | 0.010 | **17.2 pp** | 0.0004 |
| SmolLM2-360M | 0.136 | 0.030 | **10.7 pp** | 0.023 |

These are independent groups, not paired observations — the same question cannot
be in both strata — so a permutation test over the two label distributions is
used rather than McNemar. Qwen's Wilson intervals do not overlap
([0.095, 0.320] against [0.002, 0.054]); SmolLM's effect is smaller and its
p-value correspondingly weaker.

The direction replicates across two independently trained models; the magnitude
does not, and with n=150 split across strata neither run is powered to say much
about the difference between them. That is why this result is labelled
exploratory.

**The `NONE_DOC_HIT` row is the whole argument in one line.** Those 26 questions
are exactly the ones a document-level retrieval metric scores as a *success*:
the right document was retrieved. The span was not. The model answered **every
one of them**, abstained **never**, and was correct **never**. Under a
conventional evaluation all 26 land on the generator's account. They are a
retrieval problem.

**What this does not show.** Absolute accuracy is low even with complete
evidence (18.2% and 13.6%), because sub-billion-parameter models reading NLP
papers are weak — a property of the generator, not of the retrieval. Nothing
here supports a hallucination rate, a faithfulness benchmark, or a comparison
between deployed models, and the gap between 17.2 pp and 10.7 pp should not be
read as one model being better at using evidence.

The abstention behaviour is notable and unflattering to both. Qwen declines on
4-9% of questions and SmolLM on 0-2%, so when the evidence is missing they
almost always answer anyway — SmolLM abstained on **none** of the 101 questions
whose evidence never arrived. Whether a larger or better-aligned model abstains more
appropriately is exactly the experiment a hosted API key would enable, and the
command for it is in the README.

**Terminology.** An answer that does not match the reference is *incorrect*, not
a hallucination; no claim is made about why. `answered` counts substantive
answers. Faithfulness is deliberately left unset, because the only judge
available here would be the same small model that wrote the answer, and
self-judged faithfulness is not evidence.

---

## Reproducing

One command runs every experiment in this document and regenerates the summary
table. **No API key is required** — the embedder runs locally and the generator
is a deterministic extractive control, so the retrieval and evidence
measurements that the findings rest on reproduce offline.

```bash
pip install -r requirements.txt
# fetch the three corpora as documented in docs/DATASETS.md, then:
python scripts/reproduce_study.py --all
```

It prints exactly this table, which is the study:

```
experiment           chunks/doc   A doc  B quant  C span   quant    gran
qasper_dev_300               19   0.441    0.441   0.276    0.0p   16.6p
nq_val_300_fixed             31   0.997    0.997   0.730    0.0p   26.7p
hotpot_150                    2   0.993    0.507   0.507   48.7p    0.0p
qasper_c128                  43   0.445    0.445   0.259    0.0p   18.6p
qasper_c512                   9   0.428    0.428   0.317    0.0p   11.0p
```

`--headline-only` skips the chunk-size sweep; `--skip-existing` reuses completed
runs.

The robustness experiments are separate flags, and are equally deterministic and
key-free:

```bash
python scripts/reproduce_study.py --embedder-sweep   # Result 7  (4 models)
python scripts/reproduce_study.py --topk-sweep       # Result 8  (k = 1..20)
python scripts/reproduce_study.py --multihop         # Result 9  (2WikiMultihopQA)
python scripts/reproduce_study.py --everything       # all of the above
```

The first `--embedder-sweep` run downloads three additional
sentence-transformers models (~700 MB total); everything after that is offline.

Figures and the result tables in this document are regenerated from the stored
result files rather than typed by hand:

```bash
pip install -r requirements-research.txt
python scripts/make_figures.py --all
python scripts/report_tables.py --inject docs/EXPERIMENTS.md
```

**The generation experiment (Result 11) is optional and is never run in CI.** It
replays a finished run with a different generator; retrieval is reused verbatim:

```bash
python scripts/run_llm_experiment.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --generator qwen0.5b --limit 150     --out reports/experiments/llm_qasper_qwen
```

`--generator` accepts `mock` (the control), `qwen0.5b` / `smollm360m` (local open
weights, no credentials), or `openai:MODEL` / `anthropic:MODEL`, which require
the matching key in the environment and fail loudly if it is absent rather than
substituting a different model.

Individual stages remain available:

```bash
# one experiment
python scripts/run_experiment.py --dataset qasper     --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300     --top-k 5 --chunk-size 256 --embedder minilm     --out reports/experiments/qasper_dev_300 --tag qasper_dev_300

# the decomposition, from stored records, no model calls
python scripts/run_ablation.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/experiments/decomp_qasper_dev_300.json --tag qasper_dev_300

# re-score under different taxonomy thresholds, no model calls
python scripts/reclassify.py --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/sweep --sweep-faithfulness 0.3,0.6,0.9

# build the human-annotation package (emits empty labels for a person to fill)
python scripts/build_annotation_package.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/annotation/qasper_dev_300 --n-units 200

# score completed annotations; refuses to run on empty sheets
python scripts/score_annotations.py     --package reports/annotation/qasper_dev_300     --annotator a=.../completed.jsonl --annotator b=.../completed.jsonl
```

Every run writes `summary.json`, `rows.jsonl`, `report.md` and
`inference.jsonl`, each carrying a provenance block: git commit, raw-file
SHA-256, split, sample size, chunking and retrieval configuration, embedder and
generator identity, taxonomy version and threshold fingerprint, package
versions and timestamp. Curated summaries are tracked in `results/`.

**Inspecting failure cases.** `reports/experiments/<tag>/report.md` lists every
failing row with its gold answer, the system answer, the rule that fired and
the evidence status. `rows.jsonl` carries the same per row, machine-readable.

### How exactly this reproduces

Determinism was verified rather than assumed, and the verification found a
limit worth stating.

Re-running the whole study from scratch — fresh index, fresh embeddings, fresh
retrieval — reproduced **every headline A/B/C figure exactly**: QASPER
0.441/0.441/0.276, NQ 0.997/0.997/0.730, HotpotQA 0.993/0.507/0.507.
Re-running 2WikiMultihopQA from scratch under a different run tag reproduced its
decomposition **bit-identically**, conditions and paired steps alike.

What does not reproduce to the last digit are fine-grained aggregates on the
two long-document corpora. Chunk precision, chunk recall and nDCG move by
≤0.001, answer-side means such as faithfulness by ≤0.001, and A/B/C on a
*differently tagged* QASPER run by ≤0.004 — one borderline question in 290
ranking differently between two independently built indices. HNSW is an
approximate index; two builds over the same vectors need not produce the same
graph, and a question whose fifth and sixth neighbours are near-tied can land
either side of the cutoff.

The corpora where this does not happen are the ones whose documents are a
single chunk (HotpotQA, 2Wiki): with two chunks per document there are no
near-ties to resolve.

No reported gap, significance test or conclusion changes. But "reproduces
exactly" would overstate it, and that claim is not made.

**Verified determinism of the scoring half is stronger.** Because inference and
scoring are separate, re-scoring stored records is pure computation: the
decomposition of a stored run is byte-identical across repeated invocations,
and the refactor that added `--k` was checked by confirming its default output
matched the committed decomposition byte for byte.

---

## Threats to validity

Written as a reviewer would raise them, strongest first.

**"`C ≤ B ≤ A` is a theorem, not a result."** Correct, and it is not the claim.
Span coverage implies document coverage, and requiring every gold document
implies requiring one, so neither gap can be negative — which is why the reverse
discordant cell is 0 in every run, and why that zero is used here as an
implementation check rather than presented as a finding. The claims are about
**magnitude** (7.7-57.3 pp depending on corpus and depth), **where it appears**
(long documents, multi-hop questions) and **where it does not** (2-chunk
documents: 0.0 pp on HotpotQA, 1.3 pp and non-significant on 2Wiki), **its
mechanism** (it tracks chunks-per-document across a fourfold chunk-size change),
its **robustness** (four embedders, five depths), and its **consequence** for
where failures get charged. A reviewer who observes that span-level is stricter
is right and has not engaged with any of that.

**"Just retrieve more."** Tested directly in Result 8, and it partly works. The
gap does shrink with depth — dramatically on NQ (57.3 → 7.7 pp), barely on
QASPER (20.3 → 14.5 pp), and hardly at all on 2Wiki (84.0 → 48.0 pp). Three
things keep the distinction meaningful anyway: it remains statistically
significant at k=20 on every corpus; the conventional metric **saturates** at
A = 1.000 by k=10 on NQ and HotpotQA, so it has zero variance and cannot
diagnose anything, while span-level coverage still separates runs; and deeper
retrieval is not free, since k=20 quadruples the context the generator must
handle. "Raise k" trades a retrieval problem for a context and precision
problem.

**"It is an artifact of one embedding model."** Tested in Result 7. Four models
from three training lineages (Sentence-Transformers, BAAI, Microsoft), with the
asymmetric ones called using their documented query/passage prefixes. The gap
ranges 14.5-18.3 pp on QASPER, significant in every case, one-directional in
every case. It also does not track retrieval quality: BGE has both the highest A
and the largest gap, MPNet the lowest A and nearly the smallest.

**"HotpotQA is a special case."** Was true; now tested. The quantifier effect
replicates on 2WikiMultihopQA at 64.7 pp against HotpotQA's 48.7 pp, with a
mechanical explanation — 28 of its 150 items are 4-hop, so more documents are
required and "any" diverges further from "all". Both corpora remain
Wikipedia-derived and both have construction biases, but different ones:
HotpotQA's crowdworkers wrote questions while reading the paragraphs, while
2Wiki's are generated from Wikidata relation paths and templated.

**At k=1, the multi-hop numbers are definitional.** A 2-hop question cannot have
both required documents in one retrieved slot, so B = 0.000 follows from the
pigeonhole principle rather than from anything about retrieval. Those rows are
reported for completeness and carry no evidential weight.

**The generator is a control, and the real-model runs are small.** The
reproducible baseline is a deterministic extractive control, which cannot
hallucinate or refuse and therefore says nothing about generation. The
real-language-model experiment uses open weights of 0.36B and 0.5B parameters
because no API key was available. It can show whether evidence status *predicts*
generation behaviour; it cannot characterise a frontier model, and no
hallucination rate or model ranking is claimed from it.

**Contamination is mitigated, not eliminated.** NQ and both multi-hop corpora
are built from Wikipedia, which is in every current LLM's pretraining data.
`answer_grounded` separates "correct" from "correct **and** supported by
retrieved evidence", and a correct answer without its evidence is charged to
retrieval rather than counted as success. That is a mitigation, and it matters
more for the real-model runs than for the extractive control.

**Retrieval is approximate, and reproduction is not bit-exact.** Re-running the
full study reproduced every headline A/B/C figure exactly and HotpotQA
bit-identically, but chunk-level aggregates on the long-document corpora move by
≤0.001 between independently built indices — one borderline question in three
hundred ranking differently. No reported gap, test or conclusion changes, but
"reproduces exactly" would overstate it.

**Sample sizes are moderate** (n = 290 / 300 / 150 / 150). The McNemar results
are significant because the discordance is large and one-directional; the *rate*
estimates carry Wilson intervals and are reported with them.

**The taxonomy is unvalidated against humans.** Its thresholds were tuned by
inspection on a 20-question fixture, which is therefore development data. The
A/B/C measurements do not depend on those thresholds — evidence alignment is
threshold-free apart from `min_overlap_chars = 1` — but every failure-mode
distribution does. The validation protocol is implemented and **no labels have
been collected**; this remains the largest open gap.

**Confirmatory versus exploratory.** H1-H3 (embedder, depth, second multi-hop
corpus) were stated with their mechanisms before the runs and are confirmatory.
The generation experiment is exploratory and underpowered by design. No
multiple-comparison correction is applied to the confirmatory tests because each
answers a separate pre-stated question rather than searching a family for a
significant one; the p-values are also many orders of magnitude below any
correction threshold that would apply.
