"""The gold-span adjudication extension: which units are still open, and how they score.

The first pass adjudicated 60 of the 87 proxy-unresolved units. The extension package
covers everything that still has no human judgement -- the 27 unresolved units the sample
did not draw, plus the 10 units both proxies resolved on their own. The tests below check
the selection logic on a synthetic audit (so they run in a clean checkout) and, when the
real package is present, that it is a clean census with no overlap and no missing text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_goldspan_remainder import PROXY_PRESENT, remaining_ids
from scripts.score_goldspan_adjudication import read_answers, stratified

REPO = Path(__file__).resolve().parents[1]
FIRST = REPO / "reports" / "annotation" / "goldspan_adjudication"
REMAINDER = REPO / "reports" / "annotation" / "goldspan_adjudication_remaining37"
VALID = {"YES", "NO", "CANNOT_TELL"}


def _audit(units: list[tuple[str, str]]) -> dict:
    return {"n_units": len(units),
            "units": [{"annotation_id": i, "bucket": b} for i, b in units]}


# --- selection logic, no repository data needed -----------------------------

def test_remaining_units_are_the_unresolved_leftovers_plus_the_proxy_resolved_present():
    audit = _audit([("u1", "D_ambiguous"), ("u2", "D_ambiguous"),
                    ("u3", "C_possibly_inferable"), ("u4", PROXY_PRESENT),
                    ("u5", "A_genuinely_unsupported")])
    groups = remaining_ids(audit, already={"u1"})
    assert groups == {PROXY_PRESENT: ["u4"], "C_possibly_inferable": ["u3"],
                      "D_ambiguous": ["u2"]}
    # bucket A is never packaged: it stays an assumption, and the manifest says so
    assert "A_genuinely_unsupported" not in groups


def test_units_already_adjudicated_are_never_offered_again():
    audit = _audit([("u1", "D_ambiguous"), ("u2", PROXY_PRESENT)])
    assert remaining_ids(audit, already={"u1", "u2"}) == {}


# --- the estimator degenerates to a count once a stratum is complete --------

def test_full_adjudication_of_a_stratum_removes_its_sampling_error():
    sampled = stratified({"s": (87, 60, 4)})
    assert sampled[1] > 0                      # a sample still carries variance
    census = stratified({"s": (87, 87, 6)})
    assert census == pytest.approx((6.0, 0.0))  # a census does not


# --- the real package, when this checkout has one ---------------------------

needs_package = pytest.mark.skipif(
    not (REMAINDER / "manifest.json").exists() or not (FIRST / "manifest.json").exists(),
    reason="the gold-span adjudication packages are not built in this checkout",
)


@needs_package
def test_remainder_package_is_a_clean_census_of_thirty_seven_units():
    manifest = json.loads((REMAINDER / "manifest.json").read_text(encoding="utf-8"))
    first = json.loads((FIRST / "manifest.json").read_text(encoding="utf-8"))
    ids = [i for group in manifest["ids_by_group"].values() for i in group]
    already = {i for group in first["selected_ids_by_stratum"].values() for i in group}

    assert len(ids) == 37
    assert len(set(ids)) == 37                      # no duplicates
    assert not set(ids) & already                   # no overlap with the first pass
    assert len(already) == 60
    assert {b: len(v) for b, v in manifest["ids_by_group"].items()} == {
        PROXY_PRESENT: 10, "C_possibly_inferable": 2,
        "D_ambiguous": 18, "D_ambiguous_lexical_only": 7}
    # together the two passes cover every unit except the 36 both proxies called absent
    assert len(ids) + len(already) == manifest["population"]["zero_gold_span_coverage_units"] - 36


@needs_package
def test_remainder_units_carry_complete_text_and_valid_ranges():
    units = [json.loads(line) for line
             in (REMAINDER / "units.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(units) == 37
    for u in units:
        assert u["question"] and u["reference_answers"]
        pieces = (u.get("retrieved_context") or []) + (u.get("gold_evidence") or [])
        assert pieces, f"{u['annotation_id']} has neither retrieved text nor a gold span"
        for piece in pieces:
            lo, hi = piece["char_range"]
            assert 0 <= lo < hi, f"{u['annotation_id']}: bad range [{lo}, {hi})"
            assert piece["text_complete"] is True
            assert piece["doc_id"]


@needs_package
def test_remainder_sheet_hides_the_proxy_bucket():
    manifest = json.loads((REMAINDER / "manifest.json").read_text(encoding="utf-8"))
    withheld = set(manifest["blinding"]["withheld"])
    assert {"proxy scores", "bucket name", "first-pass answers"} <= withheld
    # the group labels themselves must not appear; the words "bucket" and "proxy" can,
    # because they occur in the QASPER paper text the reviewer is meant to read
    sheet = (REMAINDER / "review_sheet.txt").read_text(encoding="utf-8")
    for bucket in manifest["ids_by_group"]:
        assert bucket not in sheet
    assert "lexical_presence" not in sheet and "semantic_max_cosine" not in sheet


@needs_package
def test_remainder_answers_file_lists_every_unit_and_only_valid_labels():
    manifest = json.loads((REMAINDER / "manifest.json").read_text(encoding="utf-8"))
    ids = {i for group in manifest["ids_by_group"].values() for i in group}
    rows = [line.split("#")[0].strip()
            for line in (REMAINDER / "answers.csv").read_text(encoding="utf-8").splitlines()]
    listed = [r.split(",")[0] for r in rows if r and not r.lower().startswith("annotation_id")]
    assert set(listed) == ids and len(listed) == len(ids)
    # unlabelled rows are expected until the pass is done; anything written must be valid
    for label in read_answers(REMAINDER / "answers.csv").values():
        assert label in VALID


@needs_package
def test_first_pass_artifacts_are_untouched():
    first = json.loads((FIRST / "manifest.json").read_text(encoding="utf-8"))
    answers = read_answers(FIRST / "answers.csv")
    assert first["sampling"]["n_selected"] == 60
    assert len(answers) == 60
    assert sum(1 for a in answers.values() if a == "YES") == 4
    assert sum(1 for a in answers.values() if a == "NO") == 56
    assert sum(1 for a in answers.values() if a == "CANNOT_TELL") == 0
