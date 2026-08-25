"""Tests for the corrected retrieval metrics.

The legacy metrics are tested in `test_metrics.py` and must keep their old
behaviour; these tests pin the *differences*.
"""
import pytest

from src.evaluation.metrics import (
    chunk_precision_at_k,
    chunk_recall_at_k,
    distinct_documents,
    document_precision_at_k,
    document_recall_at_k,
    first_relevant_rank,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A ranking where one document won three of the four chunk slots.
REPEATED = ["doc_a", "doc_a", "doc_a", "doc_b"]


class TestGranularity:
    def test_distinct_documents_dedupes_in_rank_order(self):
        assert distinct_documents(REPEATED, k=4) == ["doc_a", "doc_b"]

    def test_legacy_precision_counts_repeats_as_separate_hits(self):
        """Documents the defect: three chunks of one document read as three hits."""
        assert precision_at_k(REPEATED, ["doc_a"], k=4) == 0.75

    def test_document_precision_dedupes(self):
        """One of the two retrieved documents is relevant."""
        assert document_precision_at_k(REPEATED, ["doc_a"], k=4) == 0.5

    def test_chunk_precision_is_explicit_about_its_unit(self):
        assert chunk_precision_at_k(REPEATED, ["doc_a"], k=4) == 0.75

    def test_chunk_precision_divides_by_retrieved_not_k(self):
        """Retrieving fewer than k chunks must not be scored as k-2 wrong ones."""
        assert precision_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b"], k=4) == 0.5
        assert chunk_precision_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b"], k=4) == 1.0


class TestUnanswerableQuestions:
    """Empty relevant sets are undefined, not zero."""

    def test_legacy_recall_returns_zero(self):
        assert recall_at_k(["doc_a"], [], k=4) == 0.0

    def test_document_recall_returns_none(self):
        assert document_recall_at_k(["doc_a"], [], k=4) is None

    def test_all_corrected_metrics_return_none(self):
        assert document_precision_at_k(["doc_a"], [], k=4) is None
        assert chunk_precision_at_k(["doc_a"], [], k=4) is None
        assert chunk_recall_at_k(["doc_a"], [], k=4, n_relevant_chunks=3) is None
        assert hit_rate_at_k(["doc_a"], [], k=4) is None
        assert first_relevant_rank(["doc_a"], []) is None
        assert reciprocal_rank(["doc_a"], []) is None
        assert ndcg_at_k(["doc_a"], [], k=4) is None


class TestDocumentRecall:
    def test_full_recall(self):
        assert document_recall_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b"], k=4) == 1.0

    def test_partial_recall(self):
        assert document_recall_at_k(["doc_a", "doc_x"], ["doc_a", "doc_b"], k=4) == 0.5

    def test_respects_k_cutoff(self):
        assert document_recall_at_k(["doc_x", "doc_a"], ["doc_a"], k=1) == 0.0

    def test_empty_retrieval(self):
        assert document_recall_at_k([], ["doc_a"], k=4) == 0.0


class TestChunkRecall:
    def test_uses_corpus_denominator(self):
        """Two of the corpus's four relevant chunks were retrieved."""
        assert chunk_recall_at_k(REPEATED, ["doc_b"], k=4, n_relevant_chunks=4) == 0.25

    def test_none_when_denominator_unknown(self):
        assert chunk_recall_at_k(REPEATED, ["doc_a"], k=4, n_relevant_chunks=None) is None

    def test_none_when_denominator_zero(self):
        assert chunk_recall_at_k(REPEATED, ["doc_a"], k=4, n_relevant_chunks=0) is None


class TestHitRateAndPosition:
    def test_hit(self):
        assert hit_rate_at_k(["doc_x", "doc_a"], ["doc_a"], k=4) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["doc_x", "doc_y"], ["doc_a"], k=4) == 0.0

    def test_miss_outside_k(self):
        assert hit_rate_at_k(["doc_x", "doc_a"], ["doc_a"], k=1) == 0.0

    def test_first_relevant_rank_is_one_based(self):
        assert first_relevant_rank(["doc_x", "doc_a"], ["doc_a"]) == 2

    def test_first_relevant_rank_none_when_absent(self):
        assert first_relevant_rank(["doc_x"], ["doc_a"]) is None

    def test_reciprocal_rank_matches_position(self):
        assert reciprocal_rank(["doc_x", "doc_a"], ["doc_a"]) == 0.5

    def test_reciprocal_rank_zero_when_missed(self):
        assert reciprocal_rank(["doc_x"], ["doc_a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["doc_a", "doc_x"], ["doc_a"], k=2, n_relevant_chunks=1) == 1.0

    def test_demotion_reduces_score(self):
        top = ndcg_at_k(["doc_a", "doc_x"], ["doc_a"], k=2, n_relevant_chunks=1)
        demoted = ndcg_at_k(["doc_x", "doc_a"], ["doc_a"], k=2, n_relevant_chunks=1)
        assert demoted < top
        assert demoted == pytest.approx(1 / 1.5849625007211562, rel=1e-9)

    def test_zero_when_nothing_relevant_retrieved(self):
        assert ndcg_at_k(["doc_x", "doc_y"], ["doc_a"], k=2, n_relevant_chunks=1) == 0.0

    def test_ideal_capped_at_k(self):
        """With more relevant chunks than slots, a full top-k is still perfect."""
        assert ndcg_at_k(["doc_a", "doc_a"], ["doc_a"], k=2, n_relevant_chunks=10) == 1.0

    def test_falls_back_to_retrieved_when_corpus_count_unknown(self):
        assert ndcg_at_k(["doc_a", "doc_x"], ["doc_a"], k=2) == 1.0
