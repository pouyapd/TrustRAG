# Sample Evaluation Run

This is a real evaluation report produced by running:

```bash
python scripts/run_offline_eval.py
```

against the bundled 20-question Q/A dataset (`data/eval/qa_test.jsonl`) over the sample document corpus (`data/documents/`).

The configuration uses:
- **LLM:** `MockExtractiveLLM` (deterministic, no API key needed)
- **Embeddings:** `HashEmbeddings` (deterministic, no network needed)
- **Vector store:** ChromaDB with cosine similarity
- **Top-k:** 4

This means the run is **fully reproducible** — anyone cloning the repo and running the script will get identical numbers. It's the same setup that runs in CI on every PR.

## Aggregate Results

| Metric | Value |
|---|---|
| Total queries | 20 |
| Recall@k (mean) | 0.90 |
| MRR (mean) | 0.83 |
| Faithfulness (mean) | 1.00 |
| Failure rate | 0.35 |
| Latency (mean) | 1.7 ms |

## What These Numbers Mean

- **Recall@k = 0.90** — The retriever finds at least one relevant document in the top-4 for 90% of questions. Strong retrieval, even with the simple hash embedder.
- **MRR = 0.83** — When a relevant document *is* found, it usually shows up at rank 1. This is the signal that retrieval ordering is healthy.
- **Faithfulness = 1.00** — Expected: the mock LLM is extractive (it copies sentences from context), so by construction it can't hallucinate. With a real LLM this number would drop and become the most informative single metric.
- **Failure rate = 35%** — All 7 failures are `partial_answer`. The mock LLM returns one sentence even when the gold answer is a multi-sentence summary, so token overlap with the reference falls below threshold. This is *exactly the kind of insight* the framework is designed to surface: the *retrieval is fine*, but the *answer generation is the bottleneck*. With a real LLM, partial_answer rate drops sharply.

## Failure Mode Breakdown

```
ok:              13
partial_answer:   7
no_retrieval:     0
wrong_retrieval:  0
hallucination:    0
refusal_when_answerable: 0
```

This is what makes failure-mode classification useful: a single number ("35% failure") would be alarming. The breakdown tells the engineer *exactly* where to invest — in this run, prompt engineering for completeness, not retrieval improvements.

## Reproducing

```bash
git clone https://github.com/pouyapd/TrustRAG.git
cd TrustRAG
pip install -r requirements.txt
python scripts/run_offline_eval.py
```

Output goes to `reports/`:
- `summary.json` — the aggregate metrics above
- `rows.jsonl` — one row per query with all per-question metrics
- `report.md` — a human-readable report with all failure cases listed
