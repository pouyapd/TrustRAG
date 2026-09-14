**QASPER** — 290 answerable questions with locatable gold spans, 111 documents, 2272 chunks of 256 tokens (overlap 32); 20.0 chunks per gold document, 2.79 gold-overlapping chunks per document.

| Model | Family | Params | hit@1 [95% CI] | hit@3 | hit@5 | MRR | median rank | ranks 2–5 | lift@1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| chance | — | — | 0.157 | 0.403 | 0.580 | 0.347 | — | — | 1.00 |
| BM25 (Okapi, k1=1.5, b=0.75) | lexical | — | 0.290 [0.240, 0.344] | 0.579 | 0.755 | 0.481 | 3 | 0.466 | 1.84 |
| sentence-transformers/all-MiniLM-L6-v2 | dense bi-encoder | 22M | 0.376 [0.322, 0.433] | 0.690 | 0.845 | 0.562 | 2 | 0.469 | 2.39 |
| BAAI/bge-small-en-v1.5 | dense bi-encoder | 33M | 0.400 [0.345, 0.457] | 0.690 | 0.845 | 0.579 | 2 | 0.445 | 2.54 |
| intfloat/e5-small-v2 | dense bi-encoder | 33M | 0.359 [0.306, 0.415] | 0.669 | 0.821 | 0.549 | 2 | 0.462 | 2.28 |
| sentence-transformers/all-mpnet-base-v2 | dense bi-encoder | 110M | 0.335 [0.283, 0.391] | 0.617 | 0.797 | 0.518 | 3 | 0.462 | 2.13 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | cross-encoder | 22M | 0.386 [0.332, 0.443] | 0.690 | 0.855 | 0.573 | 2 | 0.469 | 2.46 |
| BAAI/bge-reranker-base | cross-encoder | 278M | 0.310 [0.260, 0.366] | 0.634 | 0.797 | 0.513 | 2 | 0.486 | 1.97 |
| *BM25 with document-local IDF* | variant | — | 0.286 | 0.562 | 0.735 | 0.474 | 3 | 0.448 | 1.82 |
| *RRF(BM25, BGE-small), k=60* | variant | — | 0.345 | 0.679 | 0.814 | 0.540 | 2 | 0.469 | 2.19 |
| *best-of(BM25, dense partner) per question - a ceiling, not a system* | variant | — | 0.503 | 0.772 | 0.903 | 0.663 | 1 | 0.400 | 3.20 |

Union over the 7 models: some model ranks the gold first for 0.769 of questions (best single model 0.400); no model does for 0.231 (independence would give 0.048); every model does for 0.031; the gold is outside every model's top 5 for 0.010.

Rank distribution of the gold chunk (share of questions):

| Model | rank 1 | rank 2 | rank 3 | rank 4-5 | rank 6-10 | rank >10 | top quarter of doc |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.29 | 0.14 | 0.14 | 0.18 | 0.19 | 0.06 | 0.70 |
| minilm | 0.38 | 0.16 | 0.15 | 0.16 | 0.11 | 0.05 | 0.80 |
| bge | 0.40 | 0.18 | 0.11 | 0.16 | 0.09 | 0.07 | 0.79 |
| e5 | 0.36 | 0.18 | 0.13 | 0.15 | 0.14 | 0.04 | 0.80 |
| mpnet | 0.33 | 0.15 | 0.13 | 0.18 | 0.13 | 0.07 | 0.76 |
| ce_msmarco_minilm | 0.39 | 0.17 | 0.14 | 0.17 | 0.12 | 0.02 | 0.80 |
| bge_reranker_base | 0.31 | 0.20 | 0.12 | 0.16 | 0.16 | 0.05 | 0.77 |

Pairwise, hit@1 (exact McNemar; Holm-adjusted over all pairs) and reciprocal rank (exact sign test):

| A vs B | Δhit@1 | A-only / B-only | p | p (Holm) | ΔMRR | RR wins A/B | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 vs minilm | -0.086 | 36/61 | 0.0144 | 0.2734 | -0.081 | 81/137 | 0.0002 |
| bm25 vs bge | -0.110 | 30/62 | 0.0011 | 0.0233 | -0.097 | 77/129 | 0.0004 |
| bm25 vs e5 | -0.069 | 29/49 | 0.0308 | 0.5251 | -0.067 | 84/126 | 0.0046 |
| bm25 vs mpnet | -0.045 | 44/57 | 0.2323 | 1.0 | -0.037 | 103/127 | 0.1292 |
| bm25 vs ce_msmarco_minilm | -0.097 | 22/50 | 0.0013 | 0.0259 | -0.091 | 71/127 | 0.0001 |
| bm25 vs bge_reranker_base | -0.021 | 50/56 | 0.6274 | 1.0 | -0.032 | 111/123 | 0.4722 |
| minilm vs bge | -0.024 | 37/44 | 0.5052 | 1.0 | -0.017 | 91/94 | 0.8831 |
| minilm vs e5 | +0.017 | 46/41 | 0.6683 | 1.0 | +0.013 | 110/92 | 0.2316 |
| minilm vs mpnet | +0.041 | 48/36 | 0.2299 | 1.0 | +0.044 | 116/87 | 0.0491 |
| minilm vs ce_msmarco_minilm | -0.010 | 44/47 | 0.8341 | 1.0 | -0.011 | 101/103 | 0.9442 |
| minilm vs bge_reranker_base | +0.066 | 67/48 | 0.0928 | 1.0 | +0.049 | 128/98 | 0.0535 |
| bge vs e5 | +0.041 | 44/32 | 0.2067 | 1.0 | +0.030 | 100/86 | 0.3405 |
| bge vs mpnet | +0.066 | 51/32 | 0.0475 | 0.7129 | +0.060 | 122/84 | 0.0098 |
| bge vs ce_msmarco_minilm | +0.014 | 45/41 | 0.7465 | 1.0 | +0.006 | 100/98 | 0.9434 |
| bge vs bge_reranker_base | +0.090 | 79/53 | 0.0292 | 0.5251 | +0.065 | 140/96 | 0.005 |
| e5 vs mpnet | +0.024 | 46/39 | 0.5154 | 1.0 | +0.030 | 111/100 | 0.4913 |
| e5 vs ce_msmarco_minilm | -0.028 | 35/43 | 0.4282 | 1.0 | -0.024 | 88/102 | 0.3456 |
| e5 vs bge_reranker_base | +0.048 | 62/48 | 0.215 | 1.0 | +0.035 | 133/95 | 0.0141 |
| mpnet vs ce_msmarco_minilm | -0.052 | 41/56 | 0.1548 | 1.0 | -0.054 | 89/121 | 0.0322 |
| mpnet vs bge_reranker_base | +0.024 | 58/51 | 0.5657 | 1.0 | +0.005 | 121/104 | 0.2861 |
| ce_msmarco_minilm vs bge_reranker_base | +0.076 | 62/40 | 0.0371 | 0.5931 | +0.059 | 128/82 | 0.0018 |

Spearman correlation of the gold chunk's within-document rank between models:

| | bm25 | minilm | bge | e5 | mpnet | ce_msmarco_minilm | bge_reranker_base |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | — | 0.39 | 0.48 | 0.51 | 0.35 | 0.56 | 0.24 |
| minilm | 0.39 | — | 0.56 | 0.49 | 0.47 | 0.45 | 0.18 |
| bge | 0.48 | 0.56 | — | 0.58 | 0.56 | 0.50 | 0.23 |
| e5 | 0.51 | 0.49 | 0.58 | — | 0.52 | 0.54 | 0.29 |
| mpnet | 0.35 | 0.47 | 0.56 | 0.52 | — | 0.37 | 0.17 |
| ce_msmarco_minilm | 0.56 | 0.45 | 0.50 | 0.54 | 0.37 | — | 0.36 |
| bge_reranker_base | 0.24 | 0.18 | 0.23 | 0.29 | 0.17 | 0.36 | — |

Global top-5 over the whole corpus (same questions): reach A, span coverage C, localisation given reach, chunks admitted from the gold document:

| Model | A | C | C given A | admitted given A |
|---|---:|---:|---:|---:|
| bm25 | 0.528 | 0.321 | 0.608 | 3.18 |
| minilm | 0.445 | 0.279 | 0.628 | 2.71 |
| mpnet | 0.383 | 0.238 | 0.622 | 2.90 |
| bge | 0.479 | 0.297 | 0.619 | 2.66 |
| e5 | 0.466 | 0.307 | 0.659 | 2.89 |
