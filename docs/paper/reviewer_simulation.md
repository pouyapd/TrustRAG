> **Post-hoc note (added after the audit that followed this simulation).** Reviewer A's
> and Reviewer B's concerns about the retriever-ranking inversion were, if anything,
> understated. A later audit found the inversion was an artefact of an evidence-mode
> defect in our own BM25 baseline and **withdrew it entirely** (see `paper.md` §5.1).
> Every mention of the inversion below as a "strength" or "concrete finding" is
> superseded. The simulation is kept unedited as a record of what the reviews said
> before the defect was found.

# Reviewer simulation

Three adversarial reads of `paper.md`, written to find grounds for rejection rather
than to praise. Each concern is marked **[FIXED]** where the paper was changed in
response, **[ACCEPTED]** where the criticism stands and is now stated as a limitation,
or **[BLOCKED]** where fixing it needs resources not available.

---

## Reviewer A — RAG / NLP evaluation

**Summary.** A measurement-validity study of span-level evidence evaluation for RAG
failure attribution, with a small human study and a self-critical analysis of its own
gold standard.

**Strengths.** The honesty is unusual and genuinely useful: the authors replicate a
published result and say so, quantify the limits of their own instrument, and report a
human validation that contradicts their earlier automated numbers. ~~The
retriever-ranking inversion is a concrete, actionable finding.~~ *(withdrawn — see note
above)* The two-gate design
(identical run, one boolean changed) is clean and cheap.

**Major concerns.**

1. *The core premise is prior art.* Document-level metrics overstating retrieval is
   established (arXiv:2602.17981). What is left is incremental.
   **[ACCEPTED]** — the paper no longer claims it, and §1 cites the prior work as
   given. The contribution is repositioned as consequences and limits. This caps the
   paper's ceiling; it does not sink it.
2. *The human study is not independent validation.* The annotator was told which units
   to re-review and why, and moved 36 labels toward the guidelines.
   **[FIXED]** — §7 now states the dependence explicitly and calls the result
   "agreement with a guided expert reading". The abstract no longer says "validated".
3. *Generation-side taxonomy is unusable* (`ok` recall 0.094).
   **[ACCEPTED]** — reported as a negative result rather than hidden. §7.1 shows
   tuning does not rescue it.
4. *One annotator.* No IAA, so annotator idiosyncrasy is inseparable from the
   construct. **[BLOCKED]** — needs a second human. Listed as the top blocker.

**Minor.** `hallucination` cannot be studied under an extractive control — say so
earlier than §9. **[FIXED]**, moved into §5.

**Recommendation: weak reject** for a top-tier main track; **weak accept** for a
workshop or short paper, on the strength of the negative results.

---

## Reviewer B — Information retrieval

**Summary.** Compares retrieval evaluation at document and span granularity, adds a
lexical baseline, and reports an inversion in system ranking.

**Strengths.** The BM25 comparison is the right control and was run properly: same
chunks, same depth, same questions, paired test. *(It was not run properly: the
evidence mode differed between baseline and system. The audit caught it; this reviewer
did not.)* The A/B/C decomposition cleanly
separates the multi-hop quantifier effect from the long-document granularity effect,
and the orthogonality across corpora is convincing.

**Major concerns.**

1. *One dense retriever and one lexical retriever is thin.* No reranker, no hybrid, no
   ColBERT-style late interaction — all standard in 2026 IR.
   **[BLOCKED]** — feasible in principle, not run. Stated in Limitations. This is the
   most damaging gap for an IR venue.
2. *The inversion is one corpus out of three.* Presenting it as headline overreaches.
   **[FIXED]** — §5.1 now says explicitly that inversion is "possible and was
   observed, not general", and the table shows the two corpora where it does not occur.
3. *Gold-span coverage as a retrieval metric is only as good as the annotation, and
   QASPER's is sparse.* **[FIXED]** — this became §8, a whole section; since this review, human
   adjudication of 60 sampled units replaced the range with an estimate of
   0.119 [0.096, 0.142].
4. *k = 5 is shallow for a modern system.* **[ACCEPTED]** — a depth sweep exists
   (k = 1…20) and is cited; the human study is still k = 5 only.

**Minor.** BM25 parameters unswept (k1, b at defaults). **[ACCEPTED]**, noted.

**Recommendation: reject** for a main IR venue (insufficient retrieval baselines);
**borderline** for an evaluation-focused workshop.

---

## Reviewer C — Methodology and statistics

**Summary.** Paired comparisons throughout, with a human study and an ablation.

**Strengths.** Correct use of exact McNemar for paired binary outcomes rather than the
χ² approximation. Wilson intervals for proportions. The threshold ablation uses a
genuine held-out split rather than reporting tuned numbers on the tuning data — I
checked, and this is done correctly. Sample sizes are stated everywhere.

**Major concerns.**

1. *Multiple comparisons are not corrected.* Many paired tests across corpora, gates
   and strata, no family-wise control.
   **[ACCEPTED]** — the headline results (p < 0.0001, p = 0.0028) survive any
   reasonable correction; the marginal ones (oracle NONE_DOC_HIT p = 0.031) would not
   survive Bonferroni at this family size. Now stated in §6 and Limitations.
2. *n = 200 for the human study, with per-class support as low as 1.* Per-class F1 on
   support 1 is meaningless. **[FIXED]** — those rows are marked descriptive and the
   `MIN_N_FOR_INFERENCE = 30` convention is applied in the text.
3. *The 7.5%–23% under-coverage range comes from two uncalibrated proxies with
   arbitrary thresholds (0.8 lexical, 0.60 cosine).* The range is not a confidence
   interval and should not read like one.
   **[FIXED, and since resolved]** — 60 of the 87 unresolved units were adjudicated by
   a human under a stratified design, giving 0.119 [0.096, 0.142] with a published
   sensitivity analysis. This reviewer's objection is what prompted that study.
4. *The oracle experiment lacks the sham control its own reference work used*, so
   context-length confounding is not excluded. **[ACCEPTED]** — stated; the
   complete-stratum comparison is offered as a weaker internal control.

**Minor.** Report effect sizes alongside p-values for the paired tests.
**[FIXED]** — percentage-point differences given throughout.

**Recommendation: borderline** — methodologically careful, but underpowered for its
per-class claims and short of controls the cited prior work already ran.

---

## Consolidated outcome

| Concern | Status |
|---|---|
| Core premise is prior art | Accepted; reframed |
| Human study not independent | Fixed in text; underlying dependence remains |
| Generation-side taxonomy fails | Accepted, reported as negative result |
| Single annotator, no IAA | **Blocked** — needs a second human |
| Thin retrieval baselines | **Blocked** — reranker/hybrid not run |
| Inversion generalised from one corpus | Fixed |
| Gold-span annotation incomplete | Fixed — now a full section |
| No multiple-comparison control | Accepted, stated |
| Uncalibrated proxy thresholds | Fixed — presented as buckets, not an interval |
| No sham control in oracle | Accepted, stated |

**Aggregate:** weak reject / reject / borderline. No reviewer found a fatal flaw in
what is reported; all three found the contribution too incremental for a top venue and
the empirical base too narrow. Nothing in the simulation suggests a result should be
withdrawn — the substantive fixes were to claims, not to numbers.
