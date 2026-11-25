"""Tests for the reference study's critical calculations. Run: pytest studies/reference_sensitivity_2026-11/tests"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
for p in (REPO, REPO / "scripts", REV / "scripts"):
    sys.path.insert(0, str(p))

import analyse_v2 as A  # noqa: E402
from localisation_probe import random_expectations  # noqa: E402
from localisation_report import holm  # noqa: E402
from localisation_robustness import cluster_bootstrap  # noqa: E402
from src.evaluation.statistics import mcnemar_exact  # noqa: E402


@pytest.mark.parametrize("D,m", [(1, 1), (4, 1), (6, 2), (9, 3), (10, 10), (12, 4)])
def test_chance_matches_enumeration(D, m):
    hits = {k: 0 for k in (1, 3, 5)}
    rr = n = 0
    for gold in itertools.combinations(range(D), m):
        r = min(gold) + 1
        rr += 1 / r; n += 1
        for k in hits:
            hits[k] += r <= k
    exact = random_expectations(D, m, ks=(1, 3, 5))
    assert exact["rr"] == pytest.approx(rr / n, abs=1e-12)
    for k in hits:
        assert exact[f"hit@{k}"] == pytest.approx(hits[k] / n, abs=1e-12)


def test_krippendorff_matches_package():
    krippendorff = pytest.importorskip("krippendorff")
    rng = np.random.default_rng(0)
    units = [[int(x) for x in rng.integers(0, 2, rng.integers(2, 5))] for _ in range(300)]
    mat = np.full((4, len(units)), np.nan)
    for j, u in enumerate(units):
        mat[: len(u), j] = u
    assert A.krippendorff_binary(units) == pytest.approx(
        krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal"), abs=1e-10)


def test_krippendorff_perfect_and_chance():
    assert A.krippendorff_binary([[1, 1], [0, 0], [1, 1], [0, 0, 0]]) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    a = A.krippendorff_binary([[int(x) for x in rng.integers(0, 2, 2)] for _ in range(20000)])
    assert abs(a) < 0.03


def test_holm_against_definition():
    p = [0.01, 0.04, 0.03, 0.2]
    # step-down: sorted 0.01,0.03,0.04,0.2 -> 0.04,0.09,0.09,0.2 (monotone)
    assert holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_mcnemar_exact_matches_binomial():
    from scipy.stats import binomtest
    for a, b in [(0, 5), (3, 9), (10, 10), (40, 61)]:
        assert mcnemar_exact(a, b).p_value == pytest.approx(binomtest(min(a, b), a + b, 0.5).pvalue, rel=1e-9)


def test_rank_and_references():
    q = {"annotations": [{"annotator": 0, "status": "ok", "start_char": 0, "end_char": 10},
                         {"annotator": 1, "status": "ok", "start_char": 50, "end_char": 60},
                         {"annotator": 2, "status": "null"}],
         "gold_by_annotator": {"0": [0], "1": [3]}, "chunk_ranges": [[0, 20], [20, 40], [40, 55], [55, 80]]}
    first = A.gold_for(q, "nq", "first")
    union = A.gold_for(q, "nq", "union")
    assert first == {0} and union == {0, 3} and first <= union
    assert A.gold_for(q, "nq", "official") == {0, 3}   # two non-null annotations
    assert A.rank_of([2, 3, 0, 1], first) == 3 and A.rank_of([2, 3, 0, 1], union) == 2


def test_union_never_lowers_hit1():
    rng = random.Random(0)
    for _ in range(200):
        D = rng.randint(2, 30)
        order = list(range(D)); rng.shuffle(order)
        first = set(rng.sample(range(D), rng.randint(1, min(3, D))))
        union = first | set(rng.sample(range(D), rng.randint(0, min(3, D))))
        assert A.rank_of(order, union) <= A.rank_of(order, first)


def test_cluster_bootstrap_centre_and_coverage():
    rng = np.random.default_rng(3)
    clusters = [list(rng.integers(-1, 2, rng.integers(1, 6))) for _ in range(400)]
    res = cluster_bootstrap(clusters, n_boot=2000, seed=0)
    flat = [x for c in clusters for x in c]
    assert res["delta_hit@1"] == pytest.approx(np.mean(flat), abs=1e-4)
    assert res["ci95"][0] < np.mean(flat) < res["ci95"][1]
