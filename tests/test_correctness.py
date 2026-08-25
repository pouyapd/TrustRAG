"""Tests for normalized answer correctness."""
import pytest

from src.evaluation.correctness import (
    REFUSAL_MARKERS,
    abstention_rates,
    answer_precision_recall_f1,
    exact_match,
    is_refusal,
    key_fact_recall,
    key_facts,
    normalize_answer,
    normalized_answer_f1,
    normalized_tokens,
    s_stem,
)


class TestNormalization:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_answer("The Refund, Please!") == "refund please"

    def test_drops_articles(self):
        assert normalize_answer("a refund and the receipt") == "refund and receipt"

    def test_punctuation_becomes_space_not_nothing(self):
        """'30-day' must not collapse into the unmatchable token '30day'."""
        assert normalized_tokens("a 30-day window") == ["30", "day", "window"]

    def test_collapses_whitespace(self):
        assert normalize_answer("  two   words  ") == "two words"


class TestExactMatch:
    def test_identical_after_normalization(self):
        assert exact_match("The answer.", "answer") == 1.0

    def test_different(self):
        assert exact_match("30 days", "14 days") == 0.0

    def test_empty_reference(self):
        assert exact_match("anything", "") == 0.0


class TestAnswerF1:
    def test_identical(self):
        assert normalized_answer_f1("30 days", "30 days") == 1.0

    def test_disjoint(self):
        assert normalized_answer_f1("foo bar", "baz qux") == 0.0

    def test_multiset_counts_repeats(self):
        """Set-based overlap cannot see the duplicate; multiset F1 can."""
        p, r, _ = answer_precision_recall_f1("yes yes", "yes")
        assert p == 0.5
        assert r == 1.0

    def test_empty_prediction(self):
        assert normalized_answer_f1("", "something") == 0.0


class TestSStem:
    def test_strips_plural(self):
        assert s_stem("days") == "day"

    def test_keeps_double_s(self):
        assert s_stem("class") == "class"

    def test_keeps_short_words(self):
        assert s_stem("gas") == "gas"


class TestKeyFacts:
    def test_extracts_numbers_and_content_words(self):
        assert key_facts("14 days from the initial purchase.") == {
            "14",
            "day",
            "initial",
            "purchase",
        }

    def test_drops_function_words(self):
        facts = key_facts("It is on the plan")
        assert "is" not in facts
        assert "the" not in facts

    def test_recall_all_present(self):
        assert key_fact_recall("The window is 30 days from the date of purchase", "30 days from the date of purchase") == 1.0

    def test_recall_none_present(self):
        assert key_fact_recall("The Pro plan allows 1000 API requests per minute", "29 EUR per month") == 0.0

    def test_singular_plural_match(self):
        """'30-day' in the prediction must satisfy the '30 days' reference."""
        assert key_fact_recall(
            "Annual subscribers have a 30-day refund window from the date of purchase.",
            "30 days from the date of purchase.",
        ) == 1.0

    def test_none_when_reference_has_no_facts(self):
        assert key_fact_recall("anything", "it is") is None


class TestRefusal:
    def test_detects_refusal(self):
        assert is_refusal("I cannot answer this from the provided context.")

    def test_normal_answer_is_not_refusal(self):
        assert not is_refusal("The refund window is 30 days.")

    def test_markers_match_v1_classifier(self):
        """v2 carries its own copy of the marker list; it must not drift from v1."""
        from src.evaluation.failure_modes import _REFUSAL_MARKERS

        assert REFUSAL_MARKERS == _REFUSAL_MARKERS


class TestAbstentionRates:
    def test_perfect_behaviour(self):
        rates = abstention_rates(answerable=[True, False], abstained=[False, True])
        assert rates["false_answer_rate"] == 0.0
        assert rates["false_refusal_rate"] == 0.0
        assert rates["abstention_accuracy"] == 1.0

    def test_total_failure_to_abstain(self):
        rates = abstention_rates(answerable=[False, False], abstained=[False, False])
        assert rates["false_answer_rate"] == 1.0
        assert rates["n_unanswerable"] == 2

    def test_false_refusal(self):
        rates = abstention_rates(answerable=[True, True], abstained=[True, False])
        assert rates["false_refusal_rate"] == 0.5

    def test_absent_condition_is_none_not_zero(self):
        """With no unanswerable questions there is no false-answer rate to report."""
        rates = abstention_rates(answerable=[True, True], abstained=[False, False])
        assert rates["false_answer_rate"] is None

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            abstention_rates(answerable=[True], abstained=[True, False])
