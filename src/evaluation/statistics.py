"""Uncertainty quantification for evaluation results.

Every number this project reports is an estimate from a finite sample, and the
bundled dataset has 20 questions. That is far too small for most inferential
claims, so this module does two things:

1. Provides the estimators — bootstrap CIs for means, Wilson intervals for
   proportions, paired bootstrap and exact McNemar for comparing two systems on
   the same questions, and a permutation test for comparing failure-mode
   distributions.
2. Refuses to pretend. Every result carries `sufficient` and `note` fields, and
   comparisons below `MIN_N_FOR_INFERENCE` are explicitly flagged as
   underpowered rather than being reported as if they meant something.

All estimators are seeded and therefore deterministic, so adding them does not
break the reproducibility of the offline evaluation.

Implemented with numpy only — no new dependencies.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

#: Below this many observations, resampling-based intervals are so wide and so
#: lumpy that reporting a point estimate as a finding is misleading. The value
#: is a convention, not a theorem, and is reported alongside every result.
MIN_N_FOR_INFERENCE = 30

#: Default seed. Fixed so that repeated runs of the offline evaluation produce
#: byte-identical intervals.
DEFAULT_SEED = 12345

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class Estimate:
    """A point estimate with an interval and an honest sufficiency flag."""

    point: float | None
    lower: float | None
    upper: float | None
    n: int
    method: str
    confidence: float
    sufficient: bool
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def _sufficiency_note(n: int) -> tuple[bool, str]:
    if n == 0:
        return False, "no observations"
    if n < MIN_N_FOR_INFERENCE:
        return (
            False,
            f"n={n} < {MIN_N_FOR_INFERENCE}: interval is reported for transparency, "
            "not as evidence for a conclusion",
        )
    return True, f"n={n}"


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Percentile bootstrap confidence interval for a mean."""
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    sufficient, note = _sufficiency_note(n)

    if n == 0:
        return Estimate(None, None, None, 0, "bootstrap_percentile", confidence, False, note)
    if n == 1:
        return Estimate(
            clean[0], None, None, 1, "bootstrap_percentile", confidence, False,
            "n=1: no interval can be estimated",
        )

    rng = np.random.default_rng(seed)
    arr = np.asarray(clean, dtype=np.float64)
    draws = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])

    return Estimate(
        point=float(arr.mean()),
        lower=float(lower),
        upper=float(upper),
        n=n,
        method="bootstrap_percentile",
        confidence=confidence,
        sufficient=sufficient,
        note=note,
    )


def wilson_proportion_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Estimate:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at small n and at proportions near
    0 or 1 — exactly the regime this project operates in.
    """
    sufficient, note = _sufficiency_note(n)
    if n == 0:
        return Estimate(None, None, None, 0, "wilson", confidence, False, note)

    # Two-sided normal quantile for the common confidence levels.
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}.get(
        confidence
    )
    if z is None:
        z = math.sqrt(2.0) * _erfinv(confidence)

    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom

    return Estimate(
        point=p,
        lower=max(0.0, center - half),
        upper=min(1.0, center + half),
        n=n,
        method="wilson",
        confidence=confidence,
        sufficient=sufficient,
        note=note,
    )


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki approximation) for uncommon confidences."""
    a = 0.147
    ln1mx2 = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln1mx2 / 2.0
    return math.copysign(math.sqrt(math.sqrt(term * term - ln1mx2 / a) - term), x)


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two systems on the same questions."""

    statistic: float | None
    p_value: float | None
    effect: float | None
    n: int
    method: str
    sufficient: bool
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Bootstrap CI for the mean paired difference (a - b).

    Pairs are resampled together, which is the correct unit when both systems
    answered the same questions.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length sequences")

    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    est = bootstrap_mean_ci(diffs, confidence=confidence, n_resamples=n_resamples, seed=seed)
    return Estimate(
        point=est.point,
        lower=est.lower,
        upper=est.upper,
        n=est.n,
        method="paired_bootstrap_difference",
        confidence=est.confidence,
        sufficient=est.sufficient,
        note=est.note,
    )


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> Comparison:
    """Exact two-sided McNemar test for paired binary outcomes.

    Uses the exact binomial rather than the chi-square approximation, which is
    invalid when the discordant count is small — the usual case here.
    """
    n = only_a_correct + only_b_correct
    if n == 0:
        return Comparison(
            None, None, 0.0, 0, "mcnemar_exact", False,
            "no discordant pairs: the two systems failed and succeeded on exactly the same items",
        )

    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p = min(1.0, 2.0 * tail)

    sufficient = n >= 10
    note = f"{n} discordant pairs" + (
        "" if sufficient else " — fewer than 10 discordant pairs, treat p as indicative only"
    )
    return Comparison(
        statistic=float(only_a_correct - only_b_correct),
        p_value=p,
        effect=(only_a_correct - only_b_correct) / n,
        n=n,
        method="mcnemar_exact",
        sufficient=sufficient,
        note=note,
    )


def _chi_square_statistic(table: np.ndarray) -> float:
    total = table.sum()
    if total == 0:
        return 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / total
    mask = expected > 0
    return float((((table - expected) ** 2)[mask] / expected[mask]).sum())


def permutation_test_distributions(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    n_permutations: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Compare two categorical failure-mode distributions.

    Uses a chi-square statistic with a permutation null instead of the
    asymptotic chi-square distribution, because failure-mode tables are sparse
    and several expected cell counts are below 5 at this sample size — the
    condition under which the asymptotic test is not valid.

    Also reports Cramer's V as an effect size, since a p-value alone says
    nothing about how different the two distributions are.
    """
    n = len(labels_a) + len(labels_b)
    if not labels_a or not labels_b:
        return Comparison(None, None, None, n, "permutation_chi_square", False, "empty group")

    categories = sorted(set(labels_a) | set(labels_b))
    index = {c: i for i, c in enumerate(categories)}

    def to_counts(labels: Sequence[str]) -> np.ndarray:
        counts = np.zeros(len(categories), dtype=np.float64)
        for lab in labels:
            counts[index[lab]] += 1
        return counts

    observed = np.vstack([to_counts(labels_a), to_counts(labels_b)])
    stat = _chi_square_statistic(observed)

    pooled = np.array([index[lab] for lab in list(labels_a) + list(labels_b)])
    n_a = len(labels_a)
    rng = np.random.default_rng(seed)

    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm = np.zeros_like(observed)
        for i in pooled[:n_a]:
            perm[0, i] += 1
        for i in pooled[n_a:]:
            perm[1, i] += 1
        if _chi_square_statistic(perm) >= stat:
            extreme += 1

    # Add-one correction: a permutation p-value is never exactly zero.
    p = (extreme + 1) / (n_permutations + 1)

    total = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = math.sqrt(stat / (total * min_dim)) if total and min_dim else None

    sufficient = min(len(labels_a), len(labels_b)) >= MIN_N_FOR_INFERENCE
    note = (
        f"groups of {len(labels_a)} and {len(labels_b)}"
        if sufficient
        else f"groups of {len(labels_a)} and {len(labels_b)} — below "
        f"{MIN_N_FOR_INFERENCE}, underpowered; report as descriptive"
    )

    return Comparison(
        statistic=stat,
        p_value=p,
        effect=cramers_v,
        n=n,
        method="permutation_chi_square",
        sufficient=sufficient,
        note=note,
    )


def sample_size_warning(n: int, context: str = "") -> str | None:
    """Human-readable warning when a sample is too small for inference."""
    if n >= MIN_N_FOR_INFERENCE:
        return None
    prefix = f"{context}: " if context else ""
    return (
        f"{prefix}n={n} is below the n={MIN_N_FOR_INFERENCE} convention used here. "
        "Reported intervals are descriptive; differences between configurations "
        "at this sample size should not be interpreted as evidence."
    )
