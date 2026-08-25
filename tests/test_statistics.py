"""Tests for the statistics layer.

Two properties matter most: the estimators are deterministic (so adding them
does not break offline reproducibility), and they refuse to overstate what a
small sample supports.
"""
import pytest

from src.evaluation.statistics import (
    MIN_N_FOR_INFERENCE,
    bootstrap_mean_ci,
    mcnemar_exact,
    paired_bootstrap_difference,
    permutation_test_distributions,
    sample_size_warning,
    wilson_proportion_ci,
)


class TestBootstrap:
    def test_point_estimate_is_the_mean(self):
        est = bootstrap_mean_ci([0.0, 1.0, 0.5, 0.5])
        assert est.point == pytest.approx(0.5)

    def test_interval_brackets_the_point(self):
        est = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.9])
        assert est.lower <= est.point <= est.upper

    def test_deterministic_across_calls(self):
        values = [0.1, 0.7, 0.3, 0.9, 0.2]
        assert bootstrap_mean_ci(values).as_dict() == bootstrap_mean_ci(values).as_dict()

    def test_seed_changes_interval(self):
        """Needs a sample large enough that the resample distribution is not coarse.

        At n=5 the bootstrap quantiles land on the same discrete value for any
        seed, which is a property of the estimator rather than a bug.
        """
        values = [i / 50 for i in range(50)]
        assert bootstrap_mean_ci(values, seed=1).lower != bootstrap_mean_ci(values, seed=2).lower

    def test_constant_values_give_degenerate_interval(self):
        est = bootstrap_mean_ci([1.0] * 20)
        assert est.lower == est.upper == 1.0

    def test_empty_input(self):
        est = bootstrap_mean_ci([])
        assert est.point is None
        assert not est.sufficient

    def test_single_value_has_no_interval(self):
        est = bootstrap_mean_ci([0.5])
        assert est.point == 0.5
        assert est.lower is None

    def test_small_sample_is_flagged_insufficient(self):
        assert not bootstrap_mean_ci([0.5] * 20).sufficient

    def test_large_sample_is_sufficient(self):
        assert bootstrap_mean_ci([0.5, 0.6] * 40).sufficient

    def test_ignores_none_values(self):
        assert bootstrap_mean_ci([1.0, None, 1.0]).n == 2


class TestWilson:
    def test_half_proportion(self):
        est = wilson_proportion_ci(10, 20)
        assert est.point == 0.5
        assert 0.0 < est.lower < 0.5 < est.upper < 1.0

    def test_all_successes_stays_within_bounds(self):
        est = wilson_proportion_ci(18, 18)
        assert est.point == 1.0
        assert est.upper == 1.0
        assert est.lower < 1.0

    def test_zero_successes(self):
        est = wilson_proportion_ci(0, 20)
        assert est.point == 0.0
        assert est.lower == 0.0
        assert est.upper > 0.0

    def test_empty_sample(self):
        est = wilson_proportion_ci(0, 0)
        assert est.point is None

    def test_wider_confidence_is_wider(self):
        narrow = wilson_proportion_ci(10, 20, confidence=0.90)
        wide = wilson_proportion_ci(10, 20, confidence=0.99)
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


class TestPairedComparisons:
    def test_paired_bootstrap_detects_constant_difference(self):
        est = paired_bootstrap_difference([0.6] * 40, [0.4] * 40)
        assert est.point == pytest.approx(0.2)

    def test_paired_bootstrap_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            paired_bootstrap_difference([1.0], [1.0, 2.0])

    def test_mcnemar_no_discordant_pairs(self):
        result = mcnemar_exact(0, 0)
        assert result.p_value is None
        assert "no discordant pairs" in result.note

    def test_mcnemar_symmetric_case_is_not_significant(self):
        assert mcnemar_exact(10, 10).p_value == pytest.approx(1.0)

    def test_mcnemar_lopsided_case_is_significant(self):
        assert mcnemar_exact(20, 0).p_value < 0.01

    def test_mcnemar_small_counts_flagged_insufficient(self):
        assert not mcnemar_exact(2, 0).sufficient


class TestDistributionComparison:
    def test_identical_distributions_are_not_significant(self):
        labels = ["ok"] * 30 + ["incorrect_answer"] * 10
        result = permutation_test_distributions(labels, list(labels), n_permutations=200)
        assert result.p_value > 0.05

    def test_disjoint_distributions_are_significant(self):
        a = ["ok"] * 40
        b = ["incorrect_answer"] * 40
        result = permutation_test_distributions(a, b, n_permutations=500)
        assert result.p_value < 0.05
        assert result.effect == pytest.approx(1.0)

    def test_deterministic(self):
        a = ["ok", "incorrect_answer"] * 20
        b = ["ok"] * 40
        first = permutation_test_distributions(a, b, n_permutations=200)
        second = permutation_test_distributions(a, b, n_permutations=200)
        assert first.p_value == second.p_value

    def test_p_value_is_never_zero(self):
        result = permutation_test_distributions(["ok"] * 50, ["bad"] * 50, n_permutations=100)
        assert result.p_value > 0.0

    def test_small_groups_flagged_insufficient(self):
        result = permutation_test_distributions(["ok"] * 5, ["bad"] * 5, n_permutations=100)
        assert not result.sufficient
        assert "underpowered" in result.note

    def test_empty_group(self):
        assert permutation_test_distributions([], ["ok"], n_permutations=10).p_value is None


class TestSampleSizeWarning:
    def test_warns_below_threshold(self):
        assert sample_size_warning(20) is not None

    def test_silent_at_threshold(self):
        assert sample_size_warning(MIN_N_FOR_INFERENCE) is None

    def test_includes_context(self):
        assert "eval set" in sample_size_warning(5, context="eval set")
