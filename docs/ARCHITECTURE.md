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
