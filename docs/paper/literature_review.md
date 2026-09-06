# Literature review and novelty audit

Conducted September 2026, after the experiments were complete, specifically to test
whether TrustRAG's framing survives contact with published work. It does not survive
intact. This file records what had to change.

**Method and its limits.** Searches were run over arXiv and the ACL Anthology across
RAG evaluation, retrieval granularity, evidence attribution, failure taxonomies,
counterfactual evidence interventions and gold-annotation completeness. This is a
targeted review, not a systematic one: no formal protocol, no inclusion/exclusion
count, no second screener. It is sufficient to refute a novelty claim and not
sufficient to establish one, which is the asymmetry that matters here.

---

## 1. The headline finding of this review

**The central observation TrustRAG was built around — that document-level retrieval
metrics overstate success because the answer-bearing passage inside the document is
often missed — is already published.** It is stated explicitly in *Decomposing
Retrieval Failures in RAG for Long-Document Financial Question Answering*
(arXiv:2602.17981), which evaluates retrieval at document, page and chunk level on
SEC filings and names the failure mode directly: the correct filing is retrieved
while the answer-bearing context within it is not. That paper also runs an oracle
document condition and quantifies the headroom (page recall 0.46 baseline vs 0.60
oracle).

Two further pieces of prior work close off contributions this project had treated as
its own:

- **A failure taxonomy for RAG already exists at greater scope.** *A Systematic
  Taxonomy of Failure Modes in Retrieval-Augmented Generation*
  (aclanthology.org/2026.trustnlp-main.27) defines 33 failure modes across 7 pipeline
  stages, against TrustRAG's 9 categories over 3 stages. An earlier taxonomy of RAG
  applications (arXiv:2408.02854) covers the design space.
- **The oracle-evidence experiment has been done at far greater scale.** *What Would
  Fix This RAG Failure? Auditing Counterfactual Response with Paired Evidence
  Interventions* (arXiv:2608.08944) supplies annotated gold missing evidence to the
  reader and measures repair, over 11,105 eligible failures, four reader models, and
  with matched sham controls to separate semantic content from prompt-size effects.
  Their support-addition repair rate is **32.8%** [0.292–0.367].

That last number deserves emphasis, because TrustRAG's own oracle experiment
(§Results) finds **32.1%** repair on the stratum with no retrieved evidence, from
n=78 and one reader. The agreement is reassuring for correctness and fatal for
novelty: this experiment is a small replication of published work, and is now
described as such.

## 2. Comparison table

| Work | Task | Evaluation granularity | Evidence-aware? | Retrieval metric | Generation metric | Failure taxonomy | Human validation | Main limitation | TrustRAG's relation |
|---|---|---|---|---|---|---|---|---|---|
| Decomposing Retrieval Failures (arXiv:2602.17981) | Long-doc financial QA | **document / page / chunk** | Yes — answer-bearing context | recall at 3 levels, oracle condition | ROUGE-L, BLEU, numeric match | implicit | No | 150 questions, one domain, one generator | **Prior art for the core claim.** TrustRAG replicates on 4 open corpora and adds the quantifier/granularity split |
| Systematic Taxonomy of Failure Modes (TrustNLP 2026) | RAG generally | pipeline stages | partly | — | — | **33 modes, 7 stages** | not stated | descriptive; no attribution experiment | TrustRAG's 9-class taxonomy is a coarser subset, not new |
| Pair-ID counterfactual interventions (arXiv:2608.08944) | Multi-hop QA | reader-side context | Yes — gold support added/removed | — | EM, alias-normalised | 5 response classes | No | oracle edits unrealistic; 4–8B readers | **Prior art for the oracle experiment.** TrustRAG's n=150 single-reader run replicates it |
| TREC 2025 RAG Track (arXiv:2603.09891) | Open-domain RAG | nugget / attribution | Yes | relevance assessment | completeness, attribution | — | Yes, assessors | shared-task scale, not a method paper | Sets the community standard TrustRAG is far below in scale |
| Nugget Recall (arXiv:2504.15068) | RAG evaluation | fact nuggets | Yes | nugget recall | nugget-based | — | LLM + human | nugget extraction is itself automated | Adjacent: fact-level rather than span-level |
| ECoRAG (arXiv:2506.05167) | Long-context RAG | evidentiality | Yes | evidentiality-guided | downstream QA | — | No | compression focus | Adjacent: uses evidentiality to compress, not to attribute |
| SURE-RAG (arXiv:2605.03534) | Selective RAG | evidence sufficiency | Yes | sufficiency check | abstention | — | not stated | sufficiency for abstention, not attribution | Adjacent: same signal, different use |
| Dense X Retrieval (EMNLP 2024) | Open-domain QA | **proposition vs passage vs sentence** | partly | recall@k per granularity | QA accuracy | — | No | retrieval-unit design, not failure attribution | Prior art that retrieval granularity changes measured performance |
| RAGChecker | RAG diagnosis | claim-level | Yes | claim entailment | claim precision/recall | modular errors | — | claim extraction automated | Adjacent diagnostic framework |

## 3. What survives as potentially distinctive

Stated conservatively. None of these is claimed as novel without the caveat attached.

1. **The quantifier/granularity decomposition.** Separating "retrieved *a* relevant
   document" from "retrieved *all* required documents" from "retrieved the gold
   span", and showing the two gaps are close to orthogonal — each near-null on the
   corpus where the other dominates. I did not find this decomposition stated as
   such. It is a modest analytical contribution, not a discovery.
2. ~~**A retriever-ranking inversion.**~~ **Withdrawn.** This was reported as the
   strongest surviving contribution and was wrong: an evidence-mode mismatch in the
   BM25 baseline under-reported its span coverage. Corrected, BM25 leads at both
   granularities on QASPER and the dense retriever leads at both on NQ and HotpotQA,
   across five depths and three chunk sizes. See `paper.md` §5.1. The negative result
   is retained and reported; it is not a contribution.
3. **Attribution shift measured through a taxonomy and checked against human
   labels.** Prior work shows retrieval is under-credited; TrustRAG measures how the
   choice of gate re-assigns blame across failure categories, and validates that
   re-assignment against human annotation (evidence-gated 0.700 vs document-gated
   0.600 accuracy, paired 22 vs 2, p < 0.0001).
4. **Measurement-validity work on the instrument itself.** The 600-character
   annotation truncation defect, its directional effect on labels, and the
   gold-span under-coverage analysis. Negative results about one's own instrument
   are rarely published and are useful to others building similar studies.

## 4. What is not novel, and is no longer claimed

- That document-level retrieval metrics overstate retrieval success — **prior art**.
- That supplying missing gold evidence repairs a fraction of failures — **prior art**,
  at 70× the scale.
- A failure taxonomy for RAG — **prior art**, at 3.7× the granularity.
- Evidence-aware or evidentiality-aware RAG evaluation as a general idea — an active
  subfield with multiple 2024–2026 entries.

## 5. Consequence for framing

The original framing — "evidence-aware RAG evaluation", positioned as a new way to
evaluate RAG — is not defensible. It claims the thing prior work already established.

The defensible framing is narrower and more honest: **a measurement-validity study of
evidence-level RAG evaluation.** The question becomes not *should we evaluate at span
level* (answered, yes) but *what does span-level evaluation actually buy, what does it
cost, and how far can its own gold standard be trusted*. On that question this project
has something prior work does not: a retriever-ranking inversion, a human-validated
attribution comparison, and a quantified account of where the span-based gold standard
itself breaks down.

That framing is adopted in `docs/paper/paper.md`. It supports a solid workshop or
short-paper contribution, and it does not support a claim of novelty at a top-tier
venue. See `docs/paper/venue_fit.md`.

## Sources

- [Decomposing Retrieval Failures in RAG for Long-Document Financial QA](https://arxiv.org/html/2602.17981v1)
- [A Systematic Taxonomy of Failure Modes in RAG (TrustNLP 2026)](https://aclanthology.org/2026.trustnlp-main.27.pdf)
- [What Would Fix This RAG Failure? Paired Evidence Interventions](https://arxiv.org/html/2608.08944)
- [Creating a Taxonomy for RAG Applications](https://arxiv.org/pdf/2408.02854)
- [Overview of the TREC 2025 RAG Track](https://arxiv.org/html/2603.09891)
- [The Great Nugget Recall](https://arxiv.org/pdf/2504.15068)
- [ECoRAG: Evidentiality-guided Compression](https://arxiv.org/pdf/2506.05167)
- [SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification](https://arxiv.org/html/2605.03534v1)
- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://aclanthology.org/2024.emnlp-main.845.pdf)
- [When Retrieval Succeeds and Fails](https://arxiv.org/pdf/2510.09106)
