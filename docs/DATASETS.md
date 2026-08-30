# Datasets

TrustRAG evaluates on third-party QA corpora. **None of them is redistributed
in this repository.** `data/raw/` is git-ignored; what is committed is the
loaders, the licence metadata, the checksums and the code needed to rebuild an
identical evaluation set from the original sources.

That is a licensing requirement, not a preference. Natural Questions is CC BY-SA
3.0, so any redistributed derivative would inherit ShareAlike; shipping loaders
instead keeps the obligation with the original publisher.

---

## Selected datasets

| Dataset | Licence | Split used | Role |
|---|---|---|---|
| Natural Questions | CC BY-SA 3.0 | validation | Primary. Real search queries over Wikipedia, span-level evidence. |
| QASPER | CC BY 4.0 | dev | Primary. NLP papers, paragraph-level evidence, native unanswerables. |
| HotpotQA | CC BY-SA 4.0 | — | Supplementary. Multi-hop `all_required` evidence. Loader implemented, not yet run. |

The two primaries were chosen because they **differ structurally**, which is
what makes agreement between them informative rather than repetitive:

|  | Natural Questions | QASPER |
|---|---|---|
| Document | Wikipedia page | whole scientific paper |
| Mean document length | ~37,000 chars | ~22,000 chars |
| Evidence granularity | long-answer span | evidence paragraphs |
| Question written by | a real user, before seeing the page | an NLP practitioner who saw only title + abstract |
| Lexical anchoring | low | low |
| Native unanswerables | no (nulls are page-scoped) | yes |
| Memorisation risk | **high** — popular Wikipedia facts | lower — recent paper contents |

### Deliberately excluded

- **TriviaQA** — distant supervision and strong memorisation characteristics
  make it unsuitable for a failure-attribution study: a correct answer cannot
  be separated from a lucky recall.
- **MS MARCO** — its relevance labels are sparse and shallow, so an unlabelled
  passage that genuinely answers a question is scored as a miss. That
  systematically corrupts the evidence-level measurement.
- **CRAG** — CC BY-NC-4.0 does not compose with the ShareAlike sources in a
  single released artifact (`licensing.check_composition` rejects it).

---

## Obtaining the raw data

```bash
mkdir -p data/raw

# QASPER (CC BY 4.0) - 10.8 MB
curl -L -o data/raw/qasper-train-dev-v0.3.tgz \
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar xzf data/raw/qasper-train-dev-v0.3.tgz -C data/raw

# Natural Questions (CC BY-SA 3.0), validation shard 0 - 189 MB
curl -L -o data/raw/nq-validation-0.parquet \
  https://huggingface.co/api/datasets/google-research-datasets/natural_questions/parquet/default/validation/0.parquet
```

### Checksums of the files used in the reported experiments

| File | SHA-256 |
|---|---|
| `qasper-train-dev-v0.3.tgz` | `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a` |
| `nq-validation-0.parquet` | `d38ab58b0dc7065992f0175320203dd321cb14a7d29ae6c622f87c8935fb23d1` |

Every experiment report records the SHA-256 of the raw file it read, so a
result can always be traced to its exact input.

`pyarrow` is required to read the NQ parquet distribution.

---

## The unified schema

Every loader emits `QuestionRecord`s (`src/data/schema.py`). Three decisions
matter more than the rest.

**Evidence is anchored to character spans, never to chunk indices.** Chunking is
a swept experimental variable, so a chunk-level gold label is valid only for the
one chunking configuration that produced it. Spans are stored and chunk
relevance is derived per configuration.

**Answerability is a property of the corpus, not of a passage.** Seven-valued
(`Answerability`), not boolean, so abstention failures can be broken down by
kind rather than collapsed into one rate.

**Identifiers are content hashes, not `hash()`.** Python's `hash()` for `str` is
randomised per process, so ids changed between builds and any annotation keyed
to them was lost on rebuild. `src/data/identity.py` uses truncated SHA-256, so
ids are stable across processes and machines.

---

## Dataset-specific handling

### Natural Questions

Loaded from the HuggingFace parquet distribution of the **full** release
(`src/data/loaders/nq_parquet.py`).

- **Page-scoped nulls are not unanswerables.** An NQ item with no long answer
  means the annotator found no answer *on that page*, which is not the claim
  "the corpus cannot answer this". Such items are skipped and counted
  (`no_long_answer_page_scoped_null`), never emitted as abstention targets.
  Corpus-scoped unanswerables must come from deliberate evidence ablation.
- **The corpus is built from non-HTML tokens.** Markup would otherwise enter
  the retrieval corpus and the generator's context. Because dropping tokens
  shifts positions, the loader keeps an explicit original-token-index →
  character-range map and translates annotation offsets through it.
- **A page is a document.** Documents are keyed by page URL and deduplicated by
  content fingerprint, so several questions about one page share one document.
- **All five annotators' answers are kept.** In a 300-question sample, 49% of
  questions have more than one distinct reference answer, so scoring takes the
  maximum over references.

### QASPER

- A question is unanswerable only when **every** annotator said so.
- The abstract is included in the document body. Note the honest finding: on the
  real dev split **zero** evidence strings match the abstract exactly, so this
  is a robustness improvement rather than the data-loss fix it was expected to
  be.
- **Unresolvable evidence is counted, never guessed.** On the dev split 384 of
  2,808 evidence strings (13.7%) have no exact position in the paper body:
  - 253 are `FLOAT SELECTED:` figure and table captions, which QASPER stores
    outside `full_text`. A question supported only by these is **not answerable
    from the text corpus**, and is skipped as
    `evidence_only_in_figures_tables` rather than reported as a retrieval
    failure that no retriever could have avoided.
  - 131 are unmatched for other reasons, usually normalisation differences.

  Per-question counts are recorded in
  `metadata.unresolved_float_evidence` / `unresolved_unmatched_evidence`.

### HotpotQA

Loader implemented with sentence-level supporting facts mapped to character
offsets and `evidence_mode = all_required` for genuine multi-hop items. **Not
yet run in any reported experiment.**

---

## Splits and leakage

| Purpose | Source |
|---|---|
| Development / threshold inspection | the bundled 20-question fixture, and QASPER **train** |
| Reported evaluation | QASPER **dev**, NQ **validation** |

The v2 taxonomy thresholds were tuned by inspecting the 20-question fixture.
That fixture is therefore development data and is never presented as unbiased
validation. The reported experiments run on splits that played no part in
threshold selection.

**Known contamination risk.** Natural Questions is built from Wikipedia, which
is in the pretraining data of every current LLM. A correct answer on NQ may
reflect memorisation rather than retrieval. The evidence layer addresses this
directly: `answer_grounded` is true only when the answer is correct **and** the
gold evidence was actually in the retrieved context, and a correct answer
without its evidence is attributed to retrieval, not counted as a success. This
mitigates the risk; it does not eliminate it, and no claim here should be read
as eliminating it.

---

## Licence composition

`src/data/licensing.py` encodes the terms and refuses unsafe combinations:

- ShareAlike propagates to derivatives (NQ CC BY-SA 3.0, HotpotQA CC BY-SA 4.0).
- NonCommercial does not compose with ShareAlike in one artifact.
- An unverified licence defaults to `redistribution_allowed=False`.

Terms were recorded at selection time from the datasets' own documentation and
should be re-checked before any publication; dataset terms change between
releases. Nothing here is legal advice.
