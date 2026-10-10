"""Evidence references rebuilt from several annotators' spans.

A within-document localisation score depends on which chunks count as gold, and when
annotators disagree that depends on how the reference is built. The Natural Questions
validation split is five-way annotated; the study's NQ loader keeps the long answer of
the first annotator who marked one. The schemes below rebuild the reference from all of
the annotators instead:

``first``
    the first annotation that is not null: the reference the study used.
``union``
    any annotator's long answer counts. This is how NQ's official evaluation matches a
    predicted long answer (``score_long_answer`` in ``nq_eval.py``: correct if it equals
    one of the non-null gold spans).
``official``
    NQ's official rule in full: a question has a gold long answer only if at least two
    annotators marked one (``long_non_null_threshold = 2`` in ``eval_utils.py``), and then
    any of their spans counts. Questions below the threshold are excluded.
``plurality``
    the span marked by the most annotators; tied spans all count.
``single``
    one annotator drawn uniformly from those with a usable span (``draw_single``), so the
    spread of scores over draws shows how much the choice of annotator matters.

Annotations are the entries the NQ loader writes with ``keep_annotator_spans`` on: dicts
with ``annotator``, ``status`` (``"ok"``, ``"null"`` or ``"not_representable"``) and, when
``status`` is ``"ok"``, ``start_char`` and ``end_char``.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence

SCHEMES: tuple[str, ...] = ("first", "union", "official", "plurality")

#: ``long_non_null_threshold`` in NQ's official ``eval_utils.py``.
OFFICIAL_MIN_NON_NULL = 2

_NON_NULL = ("ok", "not_representable")


def usable_annotations(annotations: Sequence[dict]) -> list[dict]:
    """The annotations whose long answer has a character range in the cleaned text."""
    return [a for a in annotations if a.get("status") == "ok"]


def non_null_count(annotations: Sequence[dict]) -> int:
    """How many annotators marked a long answer, representable or not (NQ's own count)."""
    return sum(1 for a in annotations if a.get("status") in _NON_NULL)


def reference_annotators(annotations: Sequence[dict], scheme: str) -> list[int] | None:
    """Annotator indices whose spans make up the reference under ``scheme``.

    Returns None when the question has no gold long answer under the scheme: for
    ``official`` when fewer than two annotators marked one, and for every scheme when no
    annotation is usable.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown reference scheme {scheme!r}; expected one of {SCHEMES}")
    usable = usable_annotations(annotations)
    if not usable:
        return None
    if scheme == "first":
        first = next((a for a in annotations if a.get("status") in _NON_NULL), None)
        # The loader skips a question whose first long answer is pure markup, so this
        # only happens for records the study never used.
        return [first["annotator"]] if first is not None and first["status"] == "ok" else None
    if scheme == "official" and non_null_count(annotations) < OFFICIAL_MIN_NON_NULL:
        return None
    if scheme in ("union", "official"):
        return [a["annotator"] for a in usable]
    counts = Counter((a["start_char"], a["end_char"]) for a in usable)
    top = max(counts.values())
    winners = {span for span, count in counts.items() if count == top}
    return [a["annotator"] for a in usable if (a["start_char"], a["end_char"]) in winners]


def draw_single(annotations: Sequence[dict], rng: random.Random) -> list[int]:
    """One annotator with a usable span, drawn uniformly (the ``single`` scheme)."""
    usable = usable_annotations(annotations)
    if not usable:
        raise ValueError("no usable annotation to draw from")
    return [rng.choice(usable)["annotator"]]


def gold_chunks(chunk_ranges: Sequence[tuple[int, int]], spans: Iterable[tuple[int, int]]) -> set[int]:
    """Indices of the chunks that overlap any span by at least one character.

    The same rule as the study's probe: half-open ranges, positive overlap.
    """
    spans = list(spans)
    return {i for i, (c0, c1) in enumerate(chunk_ranges)
            if any(min(c1, s1) - max(c0, s0) > 0 for s0, s1 in spans)}


def first_gold_rank(order: Sequence[int], gold: set[int]) -> int | None:
    """1-based position of the first gold chunk in a best-first ordering, or None."""
    return next((position for position, index in enumerate(order, 1) if index in gold), None)


def reference_rank(ranks_by_annotator: dict[int, int], annotators: Iterable[int]) -> int:
    """Rank of the first gold chunk when several annotators' spans count.

    The first chunk that overlaps any of the spans is the earliest of the per-annotator
    first positions, so the reference rank is their minimum.
    """
    return min(ranks_by_annotator[a] for a in annotators)
