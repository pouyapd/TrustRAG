**NQ** — 300 answerable questions with locatable gold spans, 297 documents, 12358 chunks of 256 tokens (overlap 32); 41.8 chunks per gold document, 2.83 gold-overlapping chunks per document.

| Model | Family | Params | hit@1 [95% CI] | hit@3 | hit@5 | MRR | median rank | ranks 2–5 | lift@1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| chance | — | — | 0.116 | 0.277 | 0.391 | 0.255 | — | — | 1.00 |
| BM25 (Okapi, k1=1.5, b=0.75) | lexical | — | 0.353 [0.301, 0.409] | 0.617 | 0.690 | 0.507 | 3 | 0.337 | 3.05 |
| sentence-transformers/all-MiniLM-L6-v2 | dense bi-encoder | 22M | 0.387 [0.333, 0.443] | 0.650 | 0.773 | 0.557 | 2 | 0.387 | 3.34 |
| BAAI/bge-small-en-v1.5 | dense bi-encoder | 33M | 0.463 [0.408, 0.520] | 0.720 | 0.817 | 0.620 | 2 | 0.353 | 4.00 |
| intfloat/e5-small-v2 | dense bi-encoder | 33M | 0.457 [0.401, 0.513] | 0.707 | 0.803 | 0.604 | 2 | 0.347 | 3.95 |
| sentence-transformers/all-mpnet-base-v2 | dense bi-encoder | 110M | 0.403 [0.349, 0.460] | 0.690 | 0.793 | 0.576 | 2 | 0.390 | 3.49 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | cross-encoder | 22M | 0.460 [0.405, 0.516] | 0.730 | 0.813 | 0.618 | 2 | 0.353 | 3.98 |
| BAAI/bge-reranker-base | cross-encoder | 278M | 0.507 [0.450, 0.563] | 0.737 | 0.837 | 0.648 | 1 | 0.330 | 4.38 |
| *BM25 with document-local IDF* | variant | — | 0.343 | 0.560 | 0.667 | 0.491 | 3 | 0.323 | 2.97 |
| *RRF(BM25, MiniLM), k=60* | variant | — | 0.407 | 0.627 | 0.770 | 0.558 | 2 | 0.363 | 3.52 |
| *best-of(BM25, dense partner) per question - a ceiling, not a system* | variant | — | 0.563 | 0.797 | 0.877 | 0.698 | 1 | 0.313 | 4.87 |
| *RRF(BM25, BGE-small), k=60* | variant | — | 0.470 | 0.683 | 0.793 | 0.612 | 2 | 0.323 | 4.06 |

Union over the 7 models: some model ranks the gold first for 0.780 of questions (best single model 0.507); no model does for 0.220 (independence would give 0.018); every model does for 0.110; the gold is outside every model's top 5 for 0.043.

Rank distribution of the gold chunk (share of questions):

| Model | rank 1 | rank 2 | rank 3 | rank 4-5 | rank 6-10 | rank >10 | top quarter of doc |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.35 | 0.13 | 0.13 | 0.07 | 0.12 | 0.19 | 0.77 |
| minilm | 0.39 | 0.19 | 0.07 | 0.12 | 0.11 | 0.12 | 0.85 |
| bge | 0.46 | 0.17 | 0.09 | 0.10 | 0.11 | 0.08 | 0.91 |
| e5 | 0.46 | 0.13 | 0.12 | 0.10 | 0.10 | 0.09 | 0.87 |
| mpnet | 0.40 | 0.20 | 0.09 | 0.10 | 0.11 | 0.10 | 0.88 |
| ce_msmarco_minilm | 0.46 | 0.18 | 0.09 | 0.08 | 0.09 | 0.10 | 0.89 |
| bge_reranker_base | 0.51 | 0.14 | 0.09 | 0.10 | 0.08 | 0.08 | 0.92 |

Pairwise, hit@1 (exact McNemar; Holm-adjusted over all pairs) and reciprocal rank (exact sign test):

| A vs B | Δhit@1 | A-only / B-only | p | p (Holm) | ΔMRR | RR wins A/B | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 vs minilm | -0.033 | 41/51 | 0.3481 | 1.0 | -0.050 | 85/127 | 0.0047 |
| bm25 vs bge | -0.110 | 30/63 | 0.0008 | 0.0146 | -0.113 | 66/136 | 0.0 |
| bm25 vs e5 | -0.103 | 33/64 | 0.0022 | 0.0366 | -0.098 | 69/137 | 0.0 |
| bm25 vs mpnet | -0.050 | 39/54 | 0.1462 | 1.0 | -0.069 | 75/133 | 0.0001 |
| bm25 vs ce_msmarco_minilm | -0.107 | 24/56 | 0.0005 | 0.009 | -0.111 | 58/131 | 0.0 |
| bm25 vs bge_reranker_base | -0.153 | 34/80 | 0.0 | 0.0004 | -0.141 | 64/148 | 0.0 |
| minilm vs bge | -0.077 | 33/56 | 0.0192 | 0.2878 | -0.063 | 82/115 | 0.0224 |
| minilm vs e5 | -0.070 | 35/56 | 0.0354 | 0.4608 | -0.048 | 80/115 | 0.0147 |
| minilm vs mpnet | -0.017 | 37/42 | 0.653 | 1.0 | -0.019 | 80/101 | 0.1369 |
| minilm vs ce_msmarco_minilm | -0.073 | 36/58 | 0.0298 | 0.4167 | -0.061 | 79/124 | 0.0019 |
| minilm vs bge_reranker_base | -0.120 | 37/73 | 0.0008 | 0.0146 | -0.091 | 75/131 | 0.0001 |
| bge vs e5 | +0.007 | 33/31 | 0.9007 | 1.0 | +0.015 | 93/76 | 0.2183 |
| bge vs mpnet | +0.060 | 45/27 | 0.0444 | 0.5325 | +0.044 | 109/72 | 0.0073 |
| bge vs ce_msmarco_minilm | +0.003 | 43/42 | 1.0 | 1.0 | +0.002 | 98/80 | 0.2025 |
| bge vs bge_reranker_base | -0.043 | 36/49 | 0.1928 | 1.0 | -0.028 | 77/95 | 0.1947 |
| e5 vs mpnet | +0.053 | 47/31 | 0.0888 | 0.9766 | +0.028 | 103/86 | 0.2444 |
| e5 vs ce_msmarco_minilm | -0.003 | 43/44 | 1.0 | 1.0 | -0.013 | 95/90 | 0.7688 |
| e5 vs bge_reranker_base | -0.050 | 40/55 | 0.1505 | 1.0 | -0.043 | 77/110 | 0.019 |
| mpnet vs ce_msmarco_minilm | -0.057 | 40/57 | 0.1038 | 1.0 | -0.042 | 90/110 | 0.179 |
| mpnet vs bge_reranker_base | -0.103 | 34/65 | 0.0024 | 0.0383 | -0.072 | 70/119 | 0.0004 |
| ce_msmarco_minilm vs bge_reranker_base | -0.047 | 36/50 | 0.1606 | 1.0 | -0.030 | 72/98 | 0.0549 |

Spearman correlation of the gold chunk's within-document rank between models:

| | bm25 | minilm | bge | e5 | mpnet | ce_msmarco_minilm | bge_reranker_base |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | — | 0.53 | 0.52 | 0.52 | 0.47 | 0.60 | 0.35 |
| minilm | 0.53 | — | 0.58 | 0.55 | 0.61 | 0.56 | 0.39 |
| bge | 0.52 | 0.58 | — | 0.71 | 0.66 | 0.64 | 0.52 |
| e5 | 0.52 | 0.55 | 0.71 | — | 0.61 | 0.56 | 0.48 |
| mpnet | 0.47 | 0.61 | 0.66 | 0.61 | — | 0.55 | 0.46 |
| ce_msmarco_minilm | 0.60 | 0.56 | 0.64 | 0.56 | 0.55 | — | 0.55 |
| bge_reranker_base | 0.35 | 0.39 | 0.52 | 0.48 | 0.46 | 0.55 | — |

Global top-5 over the whole corpus (same questions): reach A, span coverage C, localisation given reach, chunks admitted from the gold document:

| Model | A | C | C given A | admitted given A |
|---|---:|---:|---:|---:|
| bm25 | 0.977 | 0.643 | 0.659 | 4.18 |
| minilm | 0.997 | 0.730 | 0.732 | 4.44 |
| bge | 0.990 | 0.780 | 0.788 | 4.46 |
| e5 | 0.987 | 0.757 | 0.767 | 4.41 |
| mpnet | 0.997 | 0.783 | 0.786 | 4.59 |
