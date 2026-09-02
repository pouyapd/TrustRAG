# Limitations

Written to be read before any number in `results.md` is quoted. Every item is a
property of the repository as it stands, not a hedge.

---

## 1. The reference set is model-generated, not human annotation

`reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl` —
the 200 labels every headline number is scored against — was produced by a
language model (Claude Opus 5) reading the full retrieved context against
`docs/ANNOTATION_GUIDELINES.md`. It is not a human annotation pass, and its
`PROVENANCE.md` says so.

**What this costs.** The evaluation is a *consistency check between two automated
readings of the same units* — one rule-based (the taxonomy), one instruction-
following (the reference). It does not establish that either matches human
judgement. Any paper claim of the form "the taxonomy agrees with human
annotators" is **not** supported.

**What partially mitigates it.** 22 units in the earlier package carry genuine
human labels, and the reference set agrees with them on 20/22 (kappa 0.7412).
With n=22 — below the repository's own `MIN_N_FOR_INFERENCE = 30` — this is
directional evidence, not validation.

**What would fix it.** One human pass over the same 200 full-context units, from
an annotator who has not seen the reference labels, scored with
`scripts/score_annotations.py`.

## 2. No inter-annotator agreement exists for the full-context package

The kappa of 0.8365 comes from two passes over the **truncated** package. Only
one pass exists over the corrected package, so the reference set has no agreement
statistic of its own. Reporting 0.8365 as the reliability of the reference set
would be wrong.

## 3. The annotated run uses an extractive control, not a language model

The generator for `qasper_dev_300` is `MockExtractiveLLM`, which copies the
highest-overlap sentence from retrieved context. Consequences:

- `hallucination` has **zero support** in the reference set by construction —
  the control cannot invent content absent from the context. The category is
  therefore untested, and its 8 false positives say more about the rule than
  about generator behaviour.
- `refusal_when_answerable` and `ok_abstained` have zero support as well: the
  control never declines.
- Generation-side categories are exercised on extractive spans, which are
  shorter and more literal than LLM output. Their behaviour on free-form text is
  unmeasured.

## 4. One corpus, one configuration for the annotation study

QASPER dev only, chunk size 256, overlap 32, top-k 5, MiniLM embedder. The
retrieval decomposition is replicated across four corpora, four embedders, five
depths and four chunk sizes; **the taxonomy-validation result is not**. Whether
the evidence-gating improvement holds on Natural Questions or a multi-hop corpus
is untested.

## 5. Thresholds were never re-tuned against the reference set

The v2 thresholds (`answer_f1_ok = 0.6`, `key_fact_recall_ok = 1.0`, etc.) were
set by inspection on a 20-question development fixture and are unchanged. The
reference set was used only to *score*, never to fit. That protects the
comparison from circularity, but it also means the reported accuracy is a floor:
some of the `ok` and `partial_answer` errors in `results.md` §5.2 and §5.4 are
plainly threshold artefacts and would likely shrink under tuning. No tuned
numbers are claimed.

## 6. Rare categories are measured on tiny support

Support in the 200-unit reference set: `partial_answer` 3, `answered_when_unanswerable`
9, `ok` 16, `hallucination` 0, and `no_retrieval` / `ok_abstained` /
`refusal_when_answerable` 0. Per-category F1 on a support of 3 is not a stable
estimate, and the macro F1 that averages over those categories inherits the
instability. The stratified sampler deliberately over-samples rare modes relative
to the population, so these supports reflect the *taxonomy's* proposed
distribution, not the corpus.

## 7. Stage attribution is a declared mapping, not a causal claim

`STAGE_ATTRIBUTION` maps each category to `retrieval`, `generation` or `none`.
The repository's own summary states that this is a declared mapping and that a
controlled oracle-context ablation would be required for a causal claim. No such
ablation exists here.

## 8. The generation replay uses very small models

Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct, chosen because no API key was
available. Absolute accuracy is low even with complete evidence (18.2% / 13.6%),
n=150 split across strata, and the two runs are not powered to compare the
models. No hallucination rate, faithfulness benchmark or model ranking is
claimed.

## 9. Retrieval-side caveats carried over from the earlier study

- Magnitude of the granularity gap depends strongly on retrieval depth (NQ:
  57.3 pp at k=1 → 7.7 pp at k=20).
- Four embedders, all small and all English.
- Contamination is mitigated, not eliminated — NQ and both multi-hop corpora
  derive from Wikipedia.
- Approximate nearest-neighbour search means fine-grained aggregates reproduce to
  ≤0.001, not bit-exactly.
- `C ≤ B ≤ A` holds by construction; the finding is the *magnitude* and its
  consequence for attribution, not the inequality.

## 10. Not a deployed system

Containerized, instrumented and CI-tested, but never run at production scale or
under production load. Latency figures (15.4 ms mean) come from an offline
deterministic configuration.

---

## What is missing before the paper can be written

Ordered by how much each blocks a submission.

| # | Missing | Blocks | Cost to obtain |
|---|---|---|---|
| 1 | **A human annotation pass** over the 200 full-context units | Any "human-validated" claim; the reliability of the reference set | ~4–7 hours of annotator time; tooling already exists |
| 2 | **A second independent pass** on the full-context package | Inter-annotator kappa for the corrected package | Same as above, ×2 |
| 3 | **Related-work section and citations** | Section 2 of the paper | Literature search; nothing in the repo |
| 4 | **A second corpus for the taxonomy-validation experiment** | Generality of the evidence-gating result | One `build_annotation_package.py` run + annotation on e.g. NQ |
| 5 | **A run with a real generative model behind the annotated package** | Any claim about `hallucination`, `refusal_when_answerable`, `ok_abstained` | `run_llm_experiment.py` exists; needs a new annotation package on its output |
| 6 | **A threshold-tuning ablation** (held-out split of the reference set) | Whether the taxonomy's errors are rule errors or threshold errors | Cheap: `reclassify.py` re-scores with zero model calls |
| 7 | **An oracle-context ablation** | Causal language about stage attribution | Moderate; not implemented |
| 8 | **Distribution of the annotation package** | Anyone reproducing the headline evaluation from a clone — `reports/` is gitignored, and annotation labels are data, not a computation | Decision only: track the 7.5 MB package, or publish it with a checksum (`reproducibility.md` §12) |

Items 1–3 and 8 are required. Items 4–7 strengthen the paper but each can be named as
future work without weakening the central claim, provided the claim is stated as
what it is: an evaluation-methodology result measured on one corpus against one
independent reading.
