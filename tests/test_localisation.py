"""The arithmetic the within-document localisation study rests on.

Two pieces are checked exactly rather than by example: the analytic chance level for a
random ordering of a document's chunks, against brute-force enumeration; and the
Holm-Bonferroni adjustment, against hand-computed values. Everything else in the study
is a deterministic sort of model scores, which these tests do not need a model for.
"""
from __future__ import annotations

import itertools
import math

import pytest

from scripts.localisation_probe import random_expectations, rank_stats
from scripts.localisation_report import holm


def _brute_force(D: int, m: int) -> dict:
    """Average first-gold reciprocal rank and hit@k over every placement of m gold chunks."""
    rr = hits = None
    total = 0
    rr_sum = 0.0
    hit_sum = {1: 0, 3: 0, 5: 0}
    for gold in itertools.combinations(range(D), m):
        first = min(gold) + 1
        rr_sum += 1.0 / first
        for k in hit_sum:
            hit_sum[k] += first <= k
        total += 1
    rr = rr_sum / total
    hits = {f"hit@{k}": v / total for k, v in hit_sum.items()}
    return {"rr": rr, **hits}


@pytest.mark.parametrize("D,m", [(1, 1), (5, 1), (5, 2), (20, 3), (12, 5), (7, 7)])
def test_chance_level_matches_enumeration(D: int, m: int) -> None:
    got = random_expectations(D, m)
    want = _brute_force(D, m)
    for key in ("rr", "hit@1", "hit@3", "hit@5"):
        assert math.isclose(got[key], want[key], abs_tol=1e-12), key


def test_chance_level_is_zero_without_gold() -> None:
    assert random_expectations(10, 0) == {"rr": 0.0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0}


def test_rank_stats_reads_the_first_gold_position() -> None:
    stats = rank_stats(order=[4, 2, 0, 1, 3], gold={0, 3}, D=5, m=2)
    assert stats["rank"] == 3
    assert stats["rr"] == pytest.approx(1 / 3)
    assert stats["hit@1"] is False and stats["hit@3"] is True and stats["hit@5"] is True
    assert stats["norm_rank"] == pytest.approx(2 / 4)


def test_holm_adjustment_against_hand_computation() -> None:
    # sorted p: 0.01, 0.02, 0.03, 0.04 -> 0.04, 0.06, 0.06, 0.04, then the running maximum
    # carries 0.06 forward, so the largest raw p is adjusted to 0.06 rather than 0.04
    assert holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    # the running maximum keeps adjusted values monotone
    assert holm([0.01, 0.011, 0.5]) == pytest.approx([0.03, 0.03, 0.5])
    # never above one
    assert holm([0.9, 0.8]) == [1.0, 1.0]


def test_cluster_bootstrap_keeps_clusters_whole_and_brackets_the_mean() -> None:
    from scripts.localisation_robustness import cluster_bootstrap

    # every question in cluster A favours model 1, every question in B favours model 2;
    # resampling whole clusters can only produce means in {-1, 0, +1/3, ...} built from
    # the two cluster sums, never a value the per-question mix could not produce
    clusters = [[1, 1, 1], [-1, -1, -1], [1, 1, 1], [0, 0, 0]]
    out = cluster_bootstrap(clusters, n_boot=2000, seed=1)
    assert out["delta_hit@1"] == pytest.approx(3 / 12)
    assert out["n_clusters"] == 4 and out["n_questions"] == 12
    lo, hi = out["ci95"]
    assert -1.0 <= lo <= out["delta_hit@1"] <= hi <= 1.0
    # a difference that is zero on every question has a degenerate interval at zero
    flat = cluster_bootstrap([[0, 0], [0], [0, 0, 0]], n_boot=200, seed=0)
    assert flat["ci95"] == [0.0, 0.0] and flat["excludes_zero"] is False
    # deterministic under a fixed seed
    assert cluster_bootstrap(clusters, n_boot=500, seed=7) == cluster_bootstrap(clusters, n_boot=500, seed=7)
