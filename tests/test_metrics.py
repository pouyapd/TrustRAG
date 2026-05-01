"""Tests for evaluation metrics."""
from src.evaluation.metrics import (
    cosine_similarity,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    token_overlap,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(2 / 3)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=3) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_no_relevant_set(self):
        assert recall_at_k(["a"], [], k=2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_third_position(self):
        assert mean_reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_match(self):
        assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestTokenOverlap:
    def test_identical(self):
        assert token_overlap("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert token_overlap("foo bar", "baz qux") == 0.0

    def test_partial(self):
        # Both have 2 tokens, 1 in common -> P=R=0.5 -> F1=0.5
        assert token_overlap("hello world", "hello there") == 0.5


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0


# pytest is imported only for approx; place at top normally
import pytest  # noqa: E402
