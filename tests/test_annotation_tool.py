"""Tests for the local annotation interface.

Two categories, and the first matters more than the usual correctness checks.

**Blinding.** The scientific value of the whole validation rests on the human
label being formed without sight of the system's proposed label. That is easy
to state and easy to break — a stray field in the sheet, a convenience endpoint,
a later "helpful" default. These tests assert the tool cannot read the withheld
key, cannot serialise a forbidden field even when one is deliberately injected,
and never invents a label on its own.

**Durability.** An annotator will close the laptop mid-way. Progress must
survive that, the id set must stay complete, and a crash mid-write must not
truncate the file.

Every test runs against a temporary copy of a package. The real
`annotator_a/completed.jsonl` is never created or touched by the suite, because
a synthetic label landing in the real file would silently contaminate the study.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.annotate import (  # noqa: E402
    ALLOWED_CONFIDENCE,
    ALLOWED_LABELS,
    WITHHELD_FILENAME,
    BlindingError,
    Session,
    guard_path,
    read_jsonl,
    validate,
    visible_unit,
)

REPO = Path(__file__).resolve().parent.parent
REAL_PACKAGE = REPO / "reports" / "annotation" / "qasper_dev_300"


def make_unit(n: int, answerable: bool = True) -> dict:
    return {
        "annotation_id": f"unit_{n:04d}",
        "question": f"Question number {n}?",
        "reference_answers": [f"Reference {n}"],
        "corpus_can_answer": answerable,
        "gold_evidence": [
            {"doc_id": "doc_x", "char_range": [0, 20], "text": "gold passage text"}
        ] if answerable else [],
        "retrieved_context": [
            {"rank": 1, "doc_id": "doc_x", "char_range": [0, 30], "text": "retrieved text"}
        ],
        "system_answer": f"System answer {n}",
        "human_label": "",
        "human_notes": "",
        "human_confidence": "",
    }


@pytest.fixture
def package(tmp_path):
    """A miniature package with the same schema as the real one."""
    root = tmp_path / "pkg"
    (root / "annotator_a").mkdir(parents=True)
    units = [make_unit(i, answerable=(i != 3)) for i in range(5)]
    body = "\n".join(json.dumps(u) for u in units) + "\n"
    (root / "annotation_sheet.jsonl").write_text(body, encoding="utf-8")
    (root / "annotator_a" / "annotation_sheet.jsonl").write_text(body, encoding="utf-8")
    # A decoy key file: present, as in the real package, and never to be read.
    (root / WITHHELD_FILENAME).write_text(
        json.dumps({"annotation_id": "unit_0000", "proposed_label": "ok"}) + "\n",
        encoding="utf-8",
    )
    return root


class TestBlinding:
    """The property the study depends on."""

    def test_guard_refuses_the_withheld_key(self, package):
        with pytest.raises(BlindingError, match="withheld"):
            guard_path(package / WITHHELD_FILENAME)

    def test_read_jsonl_refuses_the_withheld_key(self, package):
        """Even a caller that constructs the path directly is stopped."""
        with pytest.raises(BlindingError):
            read_jsonl(package / WITHHELD_FILENAME)

    def test_visible_unit_drops_an_injected_proposed_label(self):
        """A field that should not exist must not survive into the payload."""
        contaminated = make_unit(1) | {
            "proposed_label": "ok",
            "proposed_rule": "R7",
            "attribution_stage": "generation",
            "evidence_status": "COMPLETE",
            "key_fact_recall": 0.9,
        }
        visible = visible_unit(contaminated)
        for leaked in ("proposed_label", "proposed_rule", "attribution_stage",
                       "evidence_status", "key_fact_recall"):
            assert leaked not in visible

    def test_served_state_contains_no_forbidden_field(self, package):
        """Belt and braces: scan the actual JSON the browser would receive."""
        sheet = package / "annotator_a" / "annotation_sheet.jsonl"
        units = read_jsonl(sheet)
        units[0]["proposed_label"] = "incorrect_answer"
        sheet.write_text(
            "\n".join(json.dumps(u) for u in units) + "\n", encoding="utf-8"
        )
        payload = json.dumps(Session(package, "a").state())
        for forbidden in ("proposed_label", "proposed_rule", "attribution_stage",
                          "evidence_status", "failure_mode", "key_fact_recall"):
            assert forbidden not in payload

    def test_module_never_opens_the_key_file(self):
        """No line both names the withheld file and reads it."""
        source = (REPO / "scripts" / "annotate.py").read_text(encoding="utf-8")
        assert "WITHHELD_FILENAME = " in source, "the guard constant must exist"
        reads = ("open(", "read_text", "read_bytes", "read_jsonl", "json.load")
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "proposed_labels_key" in line and any(r in line for r in reads)
        ]
        assert not offenders, f"the withheld key is read at: {offenders}"

    def test_no_label_is_invented(self, package):
        """A fresh session proposes nothing."""
        session = Session(package, "a")
        assert session.annotations == {}
        assert all(u["human_label"] == "" for u in session.units)


class TestLabelSet:
    def test_matches_the_taxonomy(self):
        from src.evaluation.taxonomy import FailureModeV2

        assert list(ALLOWED_LABELS) == [str(m) for m in FailureModeV2]
        assert len(ALLOWED_LABELS) == 9

    def test_contains_every_label_the_guidelines_name(self):
        expected = {
            "ok", "partial_answer", "incorrect_answer", "refusal_when_answerable",
            "hallucination", "no_retrieval", "wrong_retrieval", "ok_abstained",
            "answered_when_unanswerable",
        }
        assert set(ALLOWED_LABELS) == expected

    def test_confidence_values(self):
        assert ALLOWED_CONFIDENCE == ["high", "medium", "low"]


class TestSaving:
    def test_save_writes_all_units_not_just_labelled_ones(self, package):
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "clear case")
        rows = read_jsonl(session.output_path)
        assert len(rows) == 5, "the id set must stay complete"
        assert sum(1 for r in rows if r["human_label"]) == 1

    def test_save_round_trips(self, package):
        session = Session(package, "a")
        session.save("unit_0002", "partial_answer", "medium", "missing a fact")
        row = next(r for r in read_jsonl(session.output_path)
                   if r["annotation_id"] == "unit_0002")
        assert row["human_label"] == "partial_answer"
        assert row["human_confidence"] == "medium"
        assert row["human_notes"] == "missing a fact"

    def test_unit_content_is_preserved_exactly(self, package):
        session = Session(package, "a")
        session.save("unit_0001", "ok", "high", "")
        original = {u["annotation_id"]: u for u in read_jsonl(session.sheet_path)}
        for row in read_jsonl(session.output_path):
            assert visible_unit(row) == visible_unit(original[row["annotation_id"]])

    def test_annotation_id_is_never_rewritten(self, package):
        session = Session(package, "a")
        session.save("unit_0004", "wrong_retrieval", "low", "")
        ids = [r["annotation_id"] for r in read_jsonl(session.output_path)]
        assert ids == session.order

    def test_rejects_a_label_outside_the_taxonomy(self, package):
        session = Session(package, "a")
        with pytest.raises(ValueError, match="not one of"):
            session.save("unit_0000", "looks_wrong", "high", "")
        assert not session.output_path.exists()

    def test_rejects_an_invalid_confidence(self, package):
        session = Session(package, "a")
        with pytest.raises(ValueError, match="not one of"):
            session.save("unit_0000", "ok", "very sure", "")

    def test_rejects_an_unknown_unit(self, package):
        session = Session(package, "a")
        with pytest.raises(KeyError):
            session.save("unit_9999", "ok", "high", "")

    def test_a_label_can_be_cleared(self, package):
        """A misclick must be undoable rather than permanent."""
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "")
        session.save("unit_0000", "", "", "")
        assert "unit_0000" not in session.annotations
        row = next(r for r in read_jsonl(session.output_path)
                   if r["annotation_id"] == "unit_0000")
        assert row["human_label"] == ""

    def test_no_temp_file_is_left_behind(self, package):
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "")
        assert not list(session.dir.glob("*.tmp"))


class TestResume:
    def test_progress_survives_a_restart(self, package):
        first = Session(package, "a")
        first.save("unit_0000", "ok", "high", "first pass")
        first.save("unit_0003", "ok_abstained", "medium", "")

        second = Session(package, "a")
        assert set(second.annotations) == {"unit_0000", "unit_0003"}
        assert second.annotations["unit_0000"]["human_notes"] == "first pass"

    def test_blank_rows_are_not_counted_as_done(self, package):
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "")
        assert len(Session(package, "a").annotations) == 1

    def test_the_sheet_is_never_written_to(self, package):
        sheet = package / "annotator_a" / "annotation_sheet.jsonl"
        before = sheet.read_bytes()
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "notes")
        assert sheet.read_bytes() == before

    def test_the_withheld_key_is_never_written_to(self, package):
        key = package / WITHHELD_FILENAME
        before = key.read_bytes()
        Session(package, "a").save("unit_0000", "ok", "high", "")
        assert key.read_bytes() == before


class TestValidate:
    def _complete(self, package, **overrides):
        session = Session(package, "a")
        for unit_id in session.order:
            session.save(unit_id, overrides.get(unit_id, "ok"), "high", "")
        return session

    def test_rejects_a_partly_finished_file(self, package, capsys):
        session = Session(package, "a")
        session.save("unit_0000", "ok", "high", "")
        assert validate(package, "a") == 1
        assert "no human_label" in capsys.readouterr().out

    def test_rejects_a_missing_file(self, package, capsys):
        assert validate(package, "a") == 1
        assert "no completed file" in capsys.readouterr().out

    def test_rejects_invalid_jsonl(self, package, capsys):
        self._complete(package)
        out = package / "annotator_a" / "completed.jsonl"
        out.write_text(out.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "not valid JSONL" in capsys.readouterr().out

    def test_rejects_a_duplicate_id(self, package, capsys):
        self._complete(package)
        out = package / "annotator_a" / "completed.jsonl"
        rows = read_jsonl(out)
        rows.append(rows[0])
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "duplicate annotation_id" in capsys.readouterr().out

    def test_rejects_a_label_outside_the_taxonomy(self, package, capsys):
        self._complete(package)
        out = package / "annotator_a" / "completed.jsonl"
        rows = read_jsonl(out)
        rows[0]["human_label"] = "not_a_label"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "outside the taxonomy" in capsys.readouterr().out

    def test_rejects_a_bad_confidence(self, package, capsys):
        self._complete(package)
        out = package / "annotator_a" / "completed.jsonl"
        rows = read_jsonl(out)
        rows[0]["human_confidence"] = "certain"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "high/medium/low" in capsys.readouterr().out

    def test_detects_a_modified_sheet(self, package, capsys):
        """The checksum recorded at first launch is what makes this detectable."""
        self._complete(package)
        sheet = package / "annotator_a" / "annotation_sheet.jsonl"
        rows = read_jsonl(sheet)
        rows[0]["question"] = "edited question"
        sheet.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "has changed since annotation began" in capsys.readouterr().out

    def test_detects_altered_unit_content_in_the_output(self, package, capsys):
        self._complete(package)
        out = package / "annotator_a" / "completed.jsonl"
        rows = read_jsonl(out)
        rows[0]["system_answer"] = "rewritten"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert validate(package, "a") == 1
        assert "altered the unit content" in capsys.readouterr().out

    def test_accepts_a_correctly_finished_file(self, package, capsys):
        """The only place a complete set of labels is produced — synthetic, in tmp."""
        self._complete(package)
        assert validate(package, "a") == 0
        out = capsys.readouterr().out
        assert "All checks passed" in out
        assert "label distribution" in out


class TestAgainstTheRealPackage:
    """Read-only checks that the tool's assumptions match the real data."""

    @pytest.fixture(autouse=True)
    def _skip_without_package(self):
        if not (REAL_PACKAGE / "annotator_a" / "annotation_sheet.jsonl").exists():
            pytest.skip("the real annotation package is not present")

    def test_real_sheet_has_the_expected_schema(self):
        units = read_jsonl(REAL_PACKAGE / "annotator_a" / "annotation_sheet.jsonl")
        assert len(units) == 200
        for unit in units:
            assert set(visible_unit(unit)) == {
                "annotation_id", "question", "reference_answers", "corpus_can_answer",
                "gold_evidence", "retrieved_context", "system_answer",
            }

    def test_real_sheet_carries_no_proposed_label(self):
        """If this ever fails, the package builder has leaked the key."""
        text = (REAL_PACKAGE / "annotator_a" / "annotation_sheet.jsonl").read_text(
            encoding="utf-8"
        )
        for forbidden in ("proposed_label", "proposed_rule", "attribution_stage",
                          "evidence_status", "failure_mode"):
            assert forbidden not in text

    def test_real_sheet_is_entirely_unlabelled(self):
        """The suite must never be the thing that fills these in."""
        units = read_jsonl(REAL_PACKAGE / "annotator_a" / "annotation_sheet.jsonl")
        assert all(u["human_label"] == "" for u in units)


class TestLabelLayout:
    """The on-screen grouping must cover the taxonomy exactly, and only once."""

    def test_groups_cover_every_allowed_label(self):
        from scripts.annotate import LABEL_GROUPS

        shown = [label for _, labels in LABEL_GROUPS for label in labels]
        assert sorted(shown) == sorted(ALLOWED_LABELS)

    def test_no_label_appears_twice(self):
        from scripts.annotate import LABEL_GROUPS

        shown = [label for _, labels in LABEL_GROUPS for label in labels]
        assert len(shown) == len(set(shown))

    def test_unanswerable_pair_is_kept_in_its_own_group(self):
        """So it is not reached by accident on an answerable question."""
        from scripts.annotate import LABEL_GROUPS

        group = next(labels for heading, labels in LABEL_GROUPS if "Step 1" in heading)
        assert set(group) == {"ok_abstained", "answered_when_unanswerable"}

    def test_keyboard_numbering_stays_within_one_to_nine(self):
        from scripts.annotate import LABEL_GROUPS

        assert sum(len(labels) for _, labels in LABEL_GROUPS) <= 9
