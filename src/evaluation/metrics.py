"""Retrieval and answer-quality metrics for RAG evaluation.

This module is in two halves.

**Legacy metrics** (`precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`,
`token_overlap`) are frozen. They are still computed for every row so that
previously published numbers remain reproducible, and their behaviour must not
change. They have known defects, documented on each function.

**Corrected metrics** (below the divider) fix those defects:

- *Granularity.* `retrieved_doc_ids` is a chunk-level list: one entry per
  retrieved chunk, so a document that wins three chunks appears three times.
  The legacy precision counts those repeats as separate hits and divides by k,
  so its ceiling depends on how many chunks a document happened to produce.
  The corrected metrics state their unit explicitly — `document_*` deduplicates,
  `chunk_*` does not.
- *Unanswerable questions.* When a question has no relevant document, legacy
  recall returns 0.0, which is indistinguishable from a retrieval failure and
  silently depresses the mean. The corrected metrics return None, so callers
  must decide to exclude them.
- *Missing measures.* nDCG, hit-rate and the rank of the first relevant chunk
  are added.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return _TOKEN_RE.findall(text.lower())


# ---------- Retrieval metrics (LEGACY — frozen, do not change) ----------

def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant.

    LEGACY. Divides by k even when fewer than k chunks were retrieved, and
    counts repeated doc ids as separate hits. Use `document_precision_at_k` or
    `chunk_precision_at_k` for new analysis.
    """
    if k <= 0:
        return 0.0
    top_k = list(retrieved_doc_ids)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k.

    LEGACY. Returns 0.0 when the relevant set is empty, which conflates an
    unanswerable question with a retrieval failure. Use `document_recall_at_k`.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0
    top_k = set(list(retrieved_doc_ids)[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """1 / rank of the first relevant doc, or 0 if none.

    LEGACY. Despite the name this is a single-query reciprocal rank; the mean
    is taken by the caller. Use `reciprocal_rank`, which is named correctly and
    returns None for unanswerable questions.
    """
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------- Answer-quality metrics (LEGACY — frozen) ----------

def token_overlap(predicted: str, reference: str) -> float:
    """F1 score over token sets.

    LEGACY. Set-based, so token multiplicity is ignored, and it applies no
    normalization (case is folded but punctuation and articles are not
    stripped). Use `correctness.normalized_answer_f1` for new analysis.
    """
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


# =====================================================================
# Corrected retrieval metrics
#
# Unit is explicit in every name. `retrieved_doc_ids` is always the
# chunk-level ranking (one entry per retrieved chunk, repeats allowed).
# Every function returns None when the question has no relevant document,
# because no retrieval score is defined in that case.
# =====================================================================

def _top_k_chunks(retrieved_doc_ids: Sequence[str], k: int) -> list[str]:
    """The first k entries of the chunk-level ranking."""
    if k <= 0:
        return []
    return list(retrieved_doc_ids)[:k]


def distinct_documents(retrieved_doc_ids: Sequence[str], k: int) -> list[str]:
    """Distinct documents among the top-k chunks, in first-appearance order."""
    seen: list[str] = []
    for doc_id in _top_k_chunks(retrieved_doc_ids, k):
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def document_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of relevant *documents* represented in the top-k chunks.

    Returns None when the question is unanswerable, rather than 0.0.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None
    found = relevant_set & set(distinct_documents(retrieved_doc_ids, k))
    return len(found) / len(relevant_set)


def document_precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of the *distinct documents* in the top-k that are relevant.

    Deduplicated, and divided by how many documents were actually retrieved
    rather than by k.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None
    docs = distinct_documents(retrieved_doc_ids, k)
    if not docs:
        return 0.0
    return sum(1 for d in docs if d in relevant_set) / len(docs)


def chunk_precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of retrieved *chunks* that come from a relevant document.

    Divided by the number of chunks actually retrieved, so retrieving fewer
    than k chunks is not penalised as if the missing slots were wrong.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None
    chunks = _top_k_chunks(retrieved_doc_ids, k)
    if not chunks:
        return 0.0
    return sum(1 for d in chunks if d in relevant_set) / len(chunks)


def chunk_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
    n_relevant_chunks: int | None,
) -> float | None:
    """Fraction of all relevant *chunks* in the corpus retrieved in the top-k.

    Requires `n_relevant_chunks` — the number of chunks in the corpus that
    belong to the relevant documents — which the vector store can report.
    Returns None when that count is unknown or zero, since the denominator
    would otherwise be a guess.

    Note this is bounded above by k / n_relevant_chunks: with k=4 and 6
    relevant chunks, perfect retrieval scores 0.67. That is the correct
    behaviour for recall@k and is why it must be read alongside hit-rate.
    """
    if not relevant_doc_ids or not n_relevant_chunks:
        return None
    relevant_set = set(relevant_doc_ids)
    retrieved_relevant = sum(1 for d in _top_k_chunks(retrieved_doc_ids, k) if d in relevant_set)
    return retrieved_relevant / n_relevant_chunks


def hit_rate_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """1.0 if any relevant document appears in the top-k, else 0.0.

    The metric that matters most for a RAG pipeline: whether the generator was
    given anything it could have answered from.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None
    return float(any(d in relevant_set for d in _top_k_chunks(retrieved_doc_ids, k)))


def first_relevant_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
) -> int | None:
    """1-based rank of the first relevant chunk, or None if there is none.

    Doubles as the gold-position metric: where in the context window the
    evidence landed.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return rank
    return None


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
) -> float | None:
    """1 / rank of the first relevant chunk.

    Correctly named (single query), and None rather than 0.0 when the question
    has no relevant document.
    """
    if not relevant_doc_ids:
        return None
    rank = first_relevant_rank(retrieved_doc_ids, relevant_doc_ids)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
    n_relevant_chunks: int | None = None,
) -> float | None:
    """Normalized discounted cumulative gain over the chunk ranking.

    Binary gain: a chunk scores 1 when it comes from a relevant document.

    The ideal ranking uses `n_relevant_chunks` when it is known. When it is
    not, the ideal is built from the number of relevant chunks actually
    retrieved — a weaker "nDCG among retrieved" variant that cannot see
    relevant chunks the retriever missed entirely. Callers should pass the
    corpus count whenever the store can supply it.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return None

    top = _top_k_chunks(retrieved_doc_ids, k)
    gains = [1.0 if d in relevant_set else 0.0 for d in top]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    ideal_count = n_relevant_chunks if n_relevant_chunks else int(sum(gains))
    ideal_count = min(ideal_count, k) if k > 0 else 0
    if ideal_count <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0
