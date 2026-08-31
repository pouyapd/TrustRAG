# Experiments

Everything below was produced by the code in this repository against real
third-party corpora. Nothing is estimated, projected or illustrative.

**Read this first — what these runs can and cannot support.**

| Claim type | Status |
|---|---|
| Retrieval and evidence measurement | **Real.** Real corpora, real MiniLM embeddings, real ChromaDB retrieval, real human-annotated gold spans. |
| Failure attribution between retrieval and generation | **Real for the retrieval side.** A row is charged to retrieval on evidence grounds that do not involve the generator. |
| Generation quality, hallucination rates, LLM comparison | **Not supported.** No API key was available, so the generator is a deterministic extractive control, not a language model. |
| Agreement between the taxonomy and human judgement | **Not measured.** The annotation package exists; no labels have been collected. |
| Multi-hop evidence behaviour | **Real.** Demonstrated on HotpotQA distractor: 150 genuine 2-hop questions, all `all_required`. |

---

## Setup

| | |
|---|---|
| Retriever | ChromaDB, cosine similarity |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (real, 384-dim) |
| Chunking | 256 tokens, 32 overlap, tiktoken `cl100k_base` |
| top-k | 5 |
| Generator | `MockExtractiveLLM` — deterministic extractive control |
| Taxonomy | v2.0 |

The generator copies the sentence from retrieved context with the greatest
overlap with the question. It bounds generation quality from below and makes
runs fully deterministic. Crucially, **it does not affect the retrieval or
evidence measurements**, which are what the headline result is about.

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

`--headline-only` skips the chunk-size sweep; `--skip-existing` reuses
completed runs. Individual stages remain available:

```bash
# one experiment
python scripts/run_experiment.py --dataset qasper     --raw data/raw/qasper-dev-v0.3.json --split dev --limit 300     --top-k 5 --chunk-size 256 --embedder minilm     --out reports/experiments/qasper_dev_300 --tag qasper_dev_300

# the decomposition, from stored records, no model calls
python scripts/run_ablation.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/experiments/decomp_qasper_dev_300.json --tag qasper_dev_300

# re-score under different taxonomy thresholds, no model calls
python scripts/reclassify.py --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/sweep --sweep-faithfulness 0.3,0.6,0.9

# build the human-annotation package (emits empty labels for a person to fill)
python scripts/build_annotation_package.py     --records reports/experiments/qasper_dev_300/inference.jsonl     --out reports/annotation/qasper
```

Every run writes `summary.json`, `rows.jsonl`, `report.md` and
`inference.jsonl`, each carrying a provenance block: git commit, raw-file
SHA-256, split, sample size, chunking and retrieval configuration, embedder and
generator identity, taxonomy version and threshold fingerprint, package
versions and timestamp. Curated summaries are tracked in `results/`.

**Inspecting failure cases.** `reports/experiments/<tag>/report.md` lists every
failing row with its gold answer, the system answer, the rule that fired and
the evidence status. `rows.jsonl` carries the same per row, machine-readable.

---

## Threats to validity

**The generator is not a language model.** Everything about generation quality
in these runs is a property of the extractive control. No hallucination rate,
no faithfulness number and no generation-side comparison should be read from
them. The retrieval and evidence results do not depend on the generator.

**Contamination is mitigated, not eliminated.** NQ is built from Wikipedia,
which is in every current LLM's pretraining data. `answer_grounded` separates
"correct" from "correct **and** supported by retrieved evidence", and a correct
answer without its evidence is charged to retrieval rather than counted as
success. That is a mitigation. With a real LLM the risk would be materially
larger.

**The direction of both gaps is true by construction; the magnitude is not.**
Span-level coverage implies document-level coverage, and requiring every gold
document implies requiring one, so neither gap can be negative. That is why the
reverse discordant cell is 0 in every dataset, and it is used here as an
implementation check rather than presented as a finding. What is measured is
the *magnitude*, its dependence on corpus structure, and its consequence for
attribution. A reviewer who says "of course it is stricter" is right and has
not engaged with the claim.

**One embedder and one top-k.** Chunk size is swept over a fourfold range and
the effect behaves as the mechanism predicts, but the embedder
(all-MiniLM-L6-v2) and k (5) are fixed. A stronger retriever would raise all
three conditions; whether it would narrow the *gap* is untested.

**Each effect rests on limited data.** The granularity effect is shown on two
corpora across four chunk sizes. The quantifier effect rests on a single
multi-hop corpus, HotpotQA, whose crowdworkers wrote questions while looking at
the paragraphs — lexical anchoring makes its retrieval easier than natural
queries, and a second multi-hop corpus would strengthen it considerably.

**Sample size.** The pilots are n≈60. The McNemar results are significant
because the discordance is large and one-directional, but the *rate* estimates
carry wide Wilson intervals at this n, and they are reported with them.

**The taxonomy is unvalidated against humans.** Its thresholds were tuned by
inspection on a 20-question fixture, which is therefore development data. The
Result 1 and 2 measurements do not depend on those thresholds — evidence
alignment is threshold-free apart from `min_overlap_chars = 1` — but the
failure-mode distributions do.
