# TrustRAG — Architecture & Design Decisions

## Goals

TrustRAG is built around one premise: **production RAG systems silently fail and teams have no systematic way to detect or categorize those failures.** Every design choice serves that goal.

## High-level architecture

```
        ┌──────────────┐      ┌────────────────────┐
Client ─▶│  FastAPI     │ ───▶ │   RAG Pipeline    │
        │  Gateway     │      │ (retrieve→prompt→ │
        │              │      │  generate→score)  │
        └──────┬───────┘      └─────────┬──────────┘
               │                        │
               ▼                        ▼
        ┌──────────────┐         ┌──────────────┐
        │  Prometheus  │         │  ChromaDB    │
        │  + structlog │         │  (vectors)   │
        └──────────────┘         └──────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │  Evaluation Layer           │
                          │  • retrieval metrics        │
                          │  • faithfulness (LLM judge) │
                          │  • failure mode classifier  │
                          └─────────────────────────────┘
```

## Key design decisions

### 1. Pluggable LLM/embedding providers

Why: vendor lock-in is the most common reason production RAG projects rot. The `LLMProvider` and `EmbeddingProvider` abstractions in `src/rag/providers.py` mean that swapping OpenAI for Anthropic for a local model is one config change.

A `MockExtractiveLLM` lives next to the real providers. It's not a toy — it's used in CI to run end-to-end evaluation regression without burning API credits, and it doubles as a baseline (extractive QA) to compare against the real LLM.

### 2. Faithfulness as a first-class output

Every `/query` response carries a faithfulness score (LLM-as-judge), not just an answer. This is what lets downstream systems flag low-confidence answers automatically. The score has a clear contract: 1.0 = every claim grounded, 0.0 = unrelated/contradictory.

### 3. Failure mode classification (not just a number)

`src/evaluation/failure_modes.py` is the differentiator. A failure rate of 12% tells you nothing actionable. *"12%, broken down as 6% wrong-retrieval, 4% hallucination, 2% refusal-when-answerable"* tells you exactly where to invest — re-rank? better prompts? better embeddings?

The classifier is an explicit decision tree, not a model. This is intentional: the classification logic must be human-readable, deterministic, and debuggable in production. This mirrors interpretable failure analysis from safety-critical ML literature.

### 4. Single-source-of-truth metrics

Metrics live in `src/evaluation/metrics.py` and are computed identically by:

- The CLI evaluator (`python -m src.evaluation.runner`)
- The `/evaluate` API endpoint
- The CI regression script (`scripts/run_offline_eval.py`)

There's exactly one implementation of `precision_at_k`, `mean_reciprocal_rank`, etc. No drift between dev, prod, and CI.

### 5. Observability from day one

- **Structured JSON logs** (`structlog`) — every log line is parseable; request_id propagates from middleware into every nested log
- **Prometheus metrics** — query counts, latency histograms, faithfulness distribution, failure-mode counts
- **Health endpoint** — exposes vector store size and provider config

### 6. CI evaluation regression

`.github/workflows/ci.yml` runs unit tests *and* an end-to-end evaluation regression on every PR. If a refactor pushes failure rate above the threshold or drops recall below it, CI breaks. This is the production-RAG equivalent of "tests must pass."

## Why these tradeoffs

| Decision | Tradeoff |
|---|---|
| ChromaDB instead of Pinecone/Qdrant | Simpler local-first dev; swap is a 50-line `VectorStore` change |
| LLM-as-judge for faithfulness | Adds 1 LLM call per query; can be disabled via `score_faithfulness=false` |
| Decision-tree failure classifier (not ML) | Less coverage on edge cases; gain interpretability + zero training data needed |
| In-process eval (not a separate worker) | Simpler ops; not suitable for multi-thousand-question runs (would need a queue) |

## What's deliberately *not* in here (yet)

- **Query rewriting / HyDE** — would help retrieval but adds latency; deferred until metrics show retrieval is the bottleneck
- **Reranking** — same reason
- **Multi-tenant isolation** — single-tenant by design; would need per-collection auth
- **Streaming responses** — easy to add (FastAPI supports SSE), not needed for an evaluation-focused system

## Lineage

The evaluation-first mindset comes directly from my prior work on safety-critical ML:

- *SafeTraj* (MSc thesis) — same pattern: don't just train a model, characterize when and why it fails, and translate failures into human-readable rules
- The decision-tree failure classifier here is the same shape as the decision-tree failure rule extractor used for neural trajectory predictors

The technologies are different (LLMs vs robotics) but the engineering discipline is the same.


---

# Research evaluation layer

The sections above describe the production RAG service. What follows is the
research layer built on top of it, which is where the project's actual claims
live.

## Inference and scoring are separate

`run_inference` is the only phase that calls a model. It writes
`InferenceRecord`s — question, references, retrieved chunks with ranks, scores
and character offsets, the generated answer — to `inference.jsonl`.
`score_records` is a pure function over those records.

The consequence is that re-scoring costs nothing. `scripts/reclassify.py`
re-labels a finished run under different thresholds with zero model calls, and
`scripts/run_ablation.py` applies several *methodologies* to one fixed run, so
any difference between them is attributable to the measurement rather than to
run-to-run variation. Threshold-sensitivity analysis stops being hypothetical.

## Character offsets travel the whole path

```
document text
  -> DocumentChunker      chunk.metadata{start_char, end_char}
  -> ChromaDB metadata    (offsets stored alongside doc_id and source)
  -> RetrievalResult      start_char / end_char
  -> RetrievedChunk       persisted in inference.jsonl
  -> evidence alignment   overlap arithmetic against gold spans
```

Chunk text is **sliced from the source**, never rebuilt by decoding tokens.
That was the change that made everything downstream possible: the offline
tokenizer's `decode` is `" ".join(tokens)`, which destroys the document's
whitespace, so a decoded chunk is not a substring of its source and no offset
could be recovered for it. The invariant
`document[chunk.start_char:chunk.end_char] == chunk.text` now holds by
construction on both tokenizer paths.

Recovering offsets by searching for the chunk text is not an acceptable
substitute. `str.find()` returns the first occurrence, so in any document that
repeats itself — which long scientific papers do constantly — every copy
resolves to the same wrong position.

`build_corpus` re-verifies the invariant for every chunk at index time and
refuses to build a corpus where it fails, because a corpus with bad offsets
would silently invalidate every evidence-level number computed from it.

## Layered metrics, nothing overwritten

Legacy metrics are frozen and still computed for every row, so previously
published numbers stay reproducible. Corrected document- and chunk-level
metrics sit alongside them, and evidence-level metrics alongside those. A
report contains all three, clearly separated.

## Where the research layer lives

| Module | Responsibility |
|---|---|
| `src/data/schema.py` | Unified `QuestionRecord`: span evidence, seven-valued answerability, evidence mode, provenance |
| `src/data/identity.py` | Content-hash ids, stable across processes |
| `src/data/loaders/` | NQ (parquet), QASPER, HotpotQA → unified schema |
| `src/data/corpus.py` | The bridge: `Document` → chunker → vector store, with offset verification |
| `src/data/licensing.py` | Licence terms as executable composition rules |
| `src/evaluation/evidence.py` | Span/chunk overlap alignment and the attribution hierarchy |
| `src/evaluation/taxonomy.py` | Versioned, rule-based failure taxonomy v2 |
| `src/evaluation/metrics.py` | Legacy (frozen) + corrected retrieval metrics |
| `src/evaluation/correctness.py` | Normalised EM/F1, key-fact recall, abstention rates |
| `src/evaluation/statistics.py` | Wilson, bootstrap, exact McNemar, permutation; sufficiency flags |
| `src/evaluation/provenance.py` | Git, environment, package and configuration capture |

## Design decisions worth defending

**Evidence anchored to character spans, not chunk indices.** Chunking is a
swept experimental variable, so a chunk-level gold label is valid only for the
configuration that produced it. Store the span; derive chunk relevance per
configuration.

**Answerability is corpus-scoped.** An NQ item with no long answer means "not
on this page", which is not "not in the corpus". Treating the two as the same
would corrupt every abstention measurement.

**Attribution refuses to credit ungrounded correctness.** A correct answer
produced without the gold evidence in context is charged to retrieval, not
counted as success. On a Wikipedia-derived corpus that distinction is the
difference between measuring RAG and measuring memorisation.

**The store takes its persist directory explicitly.** `from src.config import
settings` binds at import time, so reassigning `config.settings` never changed
where the store wrote. Two experiments relying on that shared one collection
and raced. Configuration that matters is passed, not discovered.
