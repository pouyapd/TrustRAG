"""Retrieval and answer-quality metrics for RAG evaluation.

Implements:
- Precision@k, Recall@k, MRR for retrieval
- Token-overlap and embedding-cosine for answer relevance
- LLM-as-judge faithfulness (delegated to RAG pipeline)
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return _TOKEN_RE.findall(text.lower())


# ---------- Retrieval metrics ----------

def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = list(retrieved_doc_ids)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k."""
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0
    top_k = set(list(retrieved_doc_ids)[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """1 / rank of the first relevant doc, or 0 if none."""
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------- Answer-quality metrics ----------

def token_overlap(predicted: str, reference: str) -> float:
    """F1 score over token sets."""
    pred_tokens = set(tokenize(predicted))
    ref_tokens = set(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)
