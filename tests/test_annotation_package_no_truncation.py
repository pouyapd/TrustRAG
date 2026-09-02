"""The annotation sheet must show the whole retrieved chunk, and keep showing it.

The first `qasper_dev_300` package stored `chunk.text[:600]`. 941 of its 1000
chunks were therefore excerpts, and step 2 of the guidelines — "did the evidence
reach the system?" — was being answered against a prefix. Evidence past the cut
looks exactly like evidence that was never retrieved, so the error is invisible
to the annotator and biases labels toward `wrong_retrieval`.

These tests exist so the same mistake cannot return quietly:

* a builder that slices chunk text fails `test_builder_does_not_slice_chunk_text`,
  whatever constant it slices with;
* a package whose stored text is shorter than its `char_range` fails the
  round-trip test, even if the source looks innocent;
* a genuinely short chunk must *not* be reported as truncated, or the check
  cries wolf and gets switched off.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.annotate import context_completeness, text_completeness  # noqa: E402
from scripts.build_annotation_package import (  # noqa: E402
    context_integrity,
    describe_chunk,
    describe_span,
    truncation_problems,
)

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "build_annotation_package.py"


class FakeChunk:
    """Stands in for `RetrievedChunk` without importing the whole pipeline."""

    def __init__(self, text: str, start: int = 0, rank: int = 1) -> None:
        self.rank = rank
        self.chunk_id = f"chunk_{rank}"
        self.doc_id = "doc_x"
        self.source = "qasper"
        self.text = text
        self.start_char = start
        self.end_char = None if start is None else start + len(text)


def unit_with(chunks: list[dict], spans: list[dict] | None = None) -> dict:
    return {
        "annotation_id": "unit_0000",
        "retrieved_context": chunks,
        "gold_evidence": spans or [],
    }


# --- the builder itself -----------------------------------------------------

def test_builder_does_not_slice_chunk_text():
    """No fixed-length slice may be applied to any text the annotator reads.

    Parsed, not grepped: a comment mentioning 600 is harmless, and a rename to
    `LIMIT = 600` followed by `text[:LIMIT]` would defeat a substring search
    while reintroducing the bug.
    """
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
            continue
        target = ast.unparse(node.value)
        if "text" not in target.lower():
            continue
        offenders.append(f"{target}[{ast.unparse(node.slice)}] on line {node.lineno}")
    assert not offenders, (
        "the package builder slices text the annotator must read in full: "
        + "; ".join(offenders)
    )


def test_describe_chunk_keeps_every_character():
    text = "x" * 1500
    described = describe_chunk(FakeChunk(text, start=100))
    assert described["text"] == text
    assert described["n_chars"] == 1500
    assert described["text_complete"] is True
    assert described["char_range"] == [100, 1600]


def test_describe_chunk_preserves_metadata():
    described = describe_chunk(FakeChunk("abc", start=7, rank=3))
    assert described["rank"] == 3
    assert described["chunk_id"] == "chunk_3"
    assert described["doc_id"] == "doc_x"
    assert described["source"] == "qasper"
    assert described["char_range"] == [7, 10]


def test_describe_span_keeps_every_character():
    text = "g" * 900
    described = describe_span({"doc_id": "d", "start_char": 0, "end_char": 900, "text": text})
    assert described["text"] == text
    assert described["text_complete"] is True


def test_completeness_is_none_without_offsets():
    """Unmeasurable is not the same as complete, and must not be reported as it."""
    chunk = FakeChunk("some text")
    chunk.start_char = None
    chunk.end_char = None
    assert describe_chunk(chunk)["text_complete"] is None


# --- the guard that refuses to write a truncated package --------------------

def test_truncation_problems_flags_a_prefix():
    chunk = {"rank": 1, "char_range": [0, 1200], "n_chars": 600, "text": "x" * 600}
    problems = truncation_problems([unit_with([chunk])])
    assert len(problems) == 1
    assert "600 chars stored for a 1200-char range" in problems[0]


def test_truncation_problems_accepts_a_genuinely_short_chunk():
    """A 150-character chunk is short, not truncated. The check must know the difference."""
    chunk = {"rank": 1, "char_range": [0, 150], "n_chars": 150, "text": "x" * 150}
    assert truncation_problems([unit_with([chunk])]) == []


def test_truncation_problems_flags_gold_spans_too():
    span = {"doc_id": "d", "char_range": [0, 800], "n_chars": 600, "text": "x" * 600}
    problems = truncation_problems([unit_with([], [span])])
    assert len(problems) == 1 and "gold" in problems[0]


def test_context_integrity_separates_the_three_states():
    chunks = [
        {"rank": 1, "char_range": [0, 100], "n_chars": 100, "text_complete": True},
        {"rank": 2, "char_range": [0, 900], "n_chars": 600, "text_complete": False},
        {"rank": 3, "char_range": [None, None], "n_chars": 50, "text_complete": None},
    ]
    tally = context_integrity([unit_with(chunks)])["retrieved_chunks"]
    assert tally == {
        "n": 3,
        "complete": 1,
        "truncated": 1,
        "unverifiable_no_offsets": 1,
        "max_chars": 600,
    }


# --- what the annotation tool reports ---------------------------------------

@pytest.mark.parametrize(
    "chunk,expected",
    [
        ({"char_range": [0, 10], "text": "0123456789"}, "complete"),
        ({"char_range": [0, 99], "text": "short"}, "truncated"),
        ({"char_range": [None, None], "text": "unknown"}, "unverifiable"),
        ({"text_complete": True, "char_range": [0, 5], "text": "abc"}, "complete"),
        ({"text_complete": False, "char_range": [0, 3], "text": "abc"}, "truncated"),
    ],
)
def test_text_completeness(chunk, expected):
    assert text_completeness(chunk) == expected


def test_context_completeness_counts_a_whole_sheet():
    sheet = [
        unit_with([{"rank": 1, "char_range": [0, 4], "text": "abcd"}]),
        unit_with([{"rank": 1, "char_range": [0, 40], "text": "abcd"}]),
    ]
    assert context_completeness(sheet) == {
        "complete": 1,
        "truncated": 1,
        "unverifiable": 0,
    }


# --- end to end, against the real rebuilt package ---------------------------

FULL_PACKAGE = REPO / "reports" / "annotation" / "qasper_dev_300_full_context"


@pytest.mark.skipif(
    not (FULL_PACKAGE / "annotation_sheet.jsonl").exists(),
    reason="the full-context package has not been built in this checkout",
)
def test_built_package_has_no_truncated_chunk():
    sheet = [
        json.loads(line)
        for line in (FULL_PACKAGE / "annotation_sheet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tally = context_completeness(sheet)
    assert tally["truncated"] == 0 and tally["unverifiable"] == 0
    assert tally["complete"] == sum(len(u["retrieved_context"]) for u in sheet)
    for unit in sheet:
        for chunk in unit["retrieved_context"]:
            start, end = chunk["char_range"]
            assert len(chunk["text"]) == end - start, (
                f"{unit['annotation_id']} rank {chunk['rank']} is not the whole chunk"
            )
