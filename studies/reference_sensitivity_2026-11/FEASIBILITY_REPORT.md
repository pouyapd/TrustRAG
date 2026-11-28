# Model and experiment feasibility

Platform: Windows-11-10.0.26300-SP0, Python 3.12.7; CPU Intel i7-1255U (10 cores / 12 threads) running at a fixed 1.7 GHz on mains power, 15.6 GB RAM, no usable GPU (Iris Xe). Throughput measured on 48 real QASPER chunks of 256 tokens after a warm-up batch (`scripts/benchmark_models.py`).

Full-sample workload: 65,143 chunks to encode (bi-encoders) and 99,492 question–chunk pairs to score (cross-encoders).

| Model | Kind | Backend | items/s | est. hours, full samples | Outcome |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | dense | torch | 23.33 | 0.8 | study model (full samples) |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | cross | torch | 19.08 | 1.4 | study model (full samples) |
| Snowflake/snowflake-arctic-embed-xs | dense | torch | 19.63 | 0.9 | not run (redundant with a run model) |
| Snowflake/snowflake-arctic-embed-xs | dense | onnx | — | — | backend failed: TypeError: ORTModel._from_pretrained() got an unexpected keyword argument 'add_pooling_lay |
| Snowflake/snowflake-arctic-embed-s | dense | torch | 9.7 | 1.9 | not run (redundant with a run model) |
| Snowflake/snowflake-arctic-embed-s | dense | onnx | — | — | backend failed: TypeError: ORTModel._from_pretrained() got an unexpected keyword argument 'add_pooling_lay |
| ibm-granite/granite-embedding-30m-english | dense | torch | 20.32 | 0.9 | run on full samples |
| ibm-granite/granite-embedding-30m-english | dense | onnx | 19.45 | 0.9 | not used (torch used) |
| Alibaba-NLP/gte-modernbert-base | dense | torch | 1.1 | 16.5 | not run: infeasible at full scale |
| Alibaba-NLP/gte-modernbert-base | dense | onnx | 1.13 | 16.0 | not run: infeasible at full scale |
| BAAI/bge-m3 | dense | torch | 0.62 | 29.2 | run on 290 QASPER questions only |
| BAAI/bge-m3 | dense | onnx | — | — | backend failed: KeyError: 'last_hidden_state' |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | cross | torch | 3.59 | 7.7 | not run: infeasible at full scale |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | cross | onnx | 3.7 | 7.5 | not run: infeasible at full scale |
| jinaai/jina-reranker-v1-turbo-en | cross | torch | 6.9 | 4.0 | not used (ONNX faster) |
| jinaai/jina-reranker-v1-turbo-en | cross | onnx | 10.08 | 2.7 | run on full samples |
| Alibaba-NLP/gte-reranker-modernbert-base | cross | torch | 1.09 | 25.4 | not run: infeasible at full scale |
| Alibaba-NLP/gte-reranker-modernbert-base | cross | onnx | 1.38 | 20.0 | not run: infeasible at full scale |

Late-interaction retrieval was not part of the benchmark above but was run afterwards: answerai-colbert-small-v1 through PyLate 1.6.0 in a separate environment (MaxSim scoring, no index needed for within-document ranking), on all QASPER questions and on the earlier study's 300 NQ questions. LLM-based rerankers (e.g. Qwen3-Reranker-0.6B; at 0.6B parameters and more these are larger than BGE-M3, 568M, which ran at 0.62 chunks per second here) were not benchmarked on this CPU. A GPU scorer for BGE-M3, Qwen3-Embedding-0.6B, bge-reranker-v2-m3 and Qwen3-Reranker-0.6B was prepared (`scripts/gpu_score.py`), but on 25 November 2026 Google Colab's free tier reported "No backend with GPU available", so it has not been run and no GPU result is reported.
ONNX: official exports were used where they loaded; the Arctic models' ONNX export failed in optimum 2.1 (`add_pooling_layer`), BGE-M3's returned no `last_hidden_state`. Quantised ONNX was not used, so as not to change model outputs relative to the documented checkpoints.
