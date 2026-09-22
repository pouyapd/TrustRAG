"""The gold-span adjudication extension: which units are still open, and how they score.

The first pass adjudicated 60 of the 87 proxy-unresolved units. The extension package
covers everything that still has no human judgement -- the 27 unresolved units the sample
did not draw, plus the 10 units both proxies resolved on their own. The tests below check
the selection logic on a synthetic audit (so they run in a clean checkout) and, when the
real package is present, that it is a clean census with no overlap and no missing text.
"""
from __future__ import annotations

import json
from collections import Counter
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


def _combined() -> dict:
    return json.loads((REMAINDER / "estimate_combined.json").read_text(
        encoding="utf-8"))["combined_with_remaining_units"]


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
needs_scored = pytest.mark.skipif(
    not (REMAINDER / "estimate_combined.json").exists(),
    reason="the combined estimate has not been produced in this checkout",
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


# --- the completed pass -----------------------------------------------------

@needs_package
def test_every_remainder_unit_carries_the_authors_label():
    answers = read_answers(REMAINDER / "answers.csv")
    manifest = json.loads((REMAINDER / "manifest.json").read_text(encoding="utf-8"))
    ids = {i for group in manifest["ids_by_group"].values() for i in group}

    assert set(answers) == ids and len(answers) == 37     # every unit judged, none invented
    counts = Counter(answers.values())
    assert counts["YES"] == 3
    assert counts["NO"] == 34
    assert counts["CANNOT_TELL"] == 0
    assert set(counts) <= VALID


@needs_scored
def test_the_unresolved_stratum_is_now_a_census_with_no_sampling_error():
    c = _combined()
    assert c["status"] == "scored"
    for bucket, g in c["per_group"].items():
        assert g["adjudicated"] == g["population"], f"{bucket} is not complete"
        assert g["from_first_pass"] + g["from_remainder"] == g["adjudicated"]
    assert c["adjudicated_units"] == 97
    for scenario in c["estimates"].values():
        assert scenario["sampling_standard_error"] == 0.0


@needs_scored
def test_combined_count_is_reproducible_from_the_two_answer_files():
    """The headline figure must be the two answer files added up, nothing else."""
    c = _combined()
    labels = list(read_answers(FIRST / "answers.csv").values())
    labels += list(read_answers(REMAINDER / "answers.csv").values())
    assert len(labels) == 97
    yes = labels.count("YES")
    assert yes == 7                                        # 4 from the first pass, 3 from this one
    assert c["estimates"]["decidable_only"]["under_coverage_count"] == pytest.approx(yes)
    assert c["estimates"]["decidable_only"]["under_coverage_rate"] == pytest.approx(
        yes / 133, abs=5e-5)


@needs_scored
def test_bucket_a_is_still_an_assumption_and_is_reported_as_one():
    c = _combined()
    assert c["units_still_unadjudicated"] == 36
    scen = c["sensitivity_to_bucket_A"]["scenarios"]
    # the reported rate assumes the 36 hold nothing; the band is what relaxing that gives
    assert scen["bucket_a_all_correct_0pct"]["under_coverage_rate"] == pytest.approx(
        c["estimates"]["decidable_only"]["under_coverage_rate"])
    assert (scen["bucket_a_behaves_like_adjudicated_units"]["under_coverage_rate"]
            > scen["bucket_a_all_correct_0pct"]["under_coverage_rate"])


@needs_scored
def test_the_second_round_is_not_described_as_independent():
    """The provenance wording is load-bearing: one reviewer, no agreement statistic."""
    c = _combined()
    manifest = json.loads((REMAINDER / "manifest.json").read_text(encoding="utf-8"))
    blurb = (c["provenance"] + manifest["relation_to_first_pass"]["note"]
             + (REMAINDER / "README.md").read_text(encoding="utf-8")).lower()
    assert "not an independent second annotation" in blurb
    assert "no inter-annotator agreement" in blurb or "yields no inter-annotator" in blurb
    for claim in ("independently validated", "inter-annotator agreement is",
                  "second annotator agreed"):
        assert claim not in blurb


@needs_package
def test_first_pass_artifacts_are_untouched():
    first = json.loads((FIRST / "manifest.json").read_text(encoding="utf-8"))
    answers = read_answers(FIRST / "answers.csv")
    assert first["sampling"]["n_selected"] == 60
    assert len(answers) == 60
    assert sum(1 for a in answers.values() if a == "YES") == 4
    assert sum(1 for a in answers.values() if a == "NO") == 56
    assert sum(1 for a in answers.values() if a == "CANNOT_TELL") == 0
