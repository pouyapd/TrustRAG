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
from collections import Counter
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


@dataclass(frozen=True)
class AgreementResult:
    """Inter-annotator agreement on a nominal label set."""

    kappa: float | None
    observed_agreement: float | None
    expected_agreement: float | None
    n: int
    n_categories: int
    #: Per-category Cohen's kappa, treating each category as one-vs-rest. A
    #: single overall kappa hides a category that both annotators use often but
    #: never on the same item.
    per_category: dict
    sufficient: bool
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def _kappa_from_counts(agree: int, expected: float, n: int) -> float | None:
    """(observed - expected) / (1 - expected), guarding the degenerate case."""
    if n == 0:
        return None
    observed = agree / n
    if expected >= 1.0:
        # Both annotators used one category for everything: agreement is total
        # but chance-corrected agreement is undefined, not perfect.
        return None
    return (observed - expected) / (1.0 - expected)


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> AgreementResult:
    """Cohen's kappa for two annotators labelling the same items.

    Kappa rather than raw agreement because the label distribution is heavily
    skewed: if 80% of rows are one category, two annotators who both guess that
    category agree 64% of the time knowing nothing. Kappa subtracts that.

    It is not a sufficient summary on its own. A high kappa with one dominant
    category can coexist with the rules being wrong on every rare category,
    which is why `per_category` is reported beside it and the confusion matrix
    is reported beside that.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"annotators labelled different numbers of items: {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        return AgreementResult(None, None, None, 0, 0, {}, False, "no annotated items")

    categories = sorted(set(labels_a) | set(labels_b))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    agree = sum(1 for x, y in zip(labels_a, labels_b, strict=True) if x == y)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    kappa = _kappa_from_counts(agree, expected, n)

    per_category = {}
    for category in categories:
        binary_a = [x == category for x in labels_a]
        binary_b = [y == category for y in labels_b]
        b_agree = sum(1 for x, y in zip(binary_a, binary_b, strict=True) if x == y)
        pa, pb = sum(binary_a) / n, sum(binary_b) / n
        b_expected = pa * pb + (1 - pa) * (1 - pb)
        per_category[category] = {
            "kappa": _kappa_from_counts(b_agree, b_expected, n),
            "n_annotator_a": count_a[category],
            "n_annotator_b": count_b[category],
            "n_both": sum(
                1 for x, y in zip(labels_a, labels_b, strict=True)
                if x == category and y == category
            ),
        }

    sufficient, note = _sufficiency_note(n)
    if kappa is None:
        note = (note + "; " if note else "") + (
            "chance agreement is total, so kappa is undefined"
        )
    return AgreementResult(
        kappa=None if kappa is None else round(kappa, 4),
        observed_agreement=round(agree / n, 4),
        expected_agreement=round(expected, 4),
        n=n,
        n_categories=len(categories),
        per_category=per_category,
        sufficient=sufficient,
        note=note,
    )


def confusion_matrix(truth: Sequence[str], predicted: Sequence[str]) -> dict:
    """Counts of predicted-vs-truth pairs, plus per-category precision/recall/F1.

    Used two ways: annotator against annotator (where neither is truth), and
    the taxonomy against adjudicated human labels (where the human side is).
    The function does not care which; the caller names the axes.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must be the same length")
    categories = sorted(set(truth) | set(predicted))
    matrix = {t: {p: 0 for p in categories} for t in categories}
    for t, p in zip(truth, predicted, strict=True):
        matrix[t][p] += 1

    per_category = {}
    for category in categories:
        tp = matrix[category][category]
        fn = sum(matrix[category][p] for p in categories if p != category)
        fp = sum(matrix[t][category] for t in categories if t != category)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if (precision is not None and recall is not None) else None)
        )
        per_category[category] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
        }

    n = len(truth)
    correct = sum(matrix[c][c] for c in categories)
    macro = [v["f1"] for v in per_category.values() if v["f1"] is not None]
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "macro_f1": round(sum(macro) / len(macro), 4) if macro else None,
        "categories": categories,
        "matrix": matrix,
        "per_category": per_category,
    }
