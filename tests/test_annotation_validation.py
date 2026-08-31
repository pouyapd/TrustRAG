"""Tests for the taxonomy-validation protocol: package, agreement, scoring.

Two things are being protected here. The first is arithmetic — kappa and the
confusion matrix are easy to get subtly wrong, so they are checked against
hand-computed values. The second is the protocol itself: the annotation sheet
must not leak the system's own label, and the scorer must refuse to invent
labels or to break a human disagreement using the system's answer. Those are
integrity properties, and a test is the only thing that keeps them true.

The labels used below are fabricated *for the tests* and are clearly synthetic.
No test result is ever presented as a human annotation of the real corpus.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_annotation_package import (  # noqa: E402
    BOUNDARY_MARGIN,
    boundary_distance,
    stratified_sample,
)
from scripts.score_annotations import adjudicate, disagreement_pairs, labelled_units  # noqa: E402
from src.evaluation.statistics import cohens_kappa, confusion_matrix  # noqa: E402
from src.evaluation.taxonomy import TaxonomyConfig  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


class TestCohensKappa:
    def test_perfect_agreement(self):
        result = cohens_kappa(["a", "b", "c"], ["a", "b", "c"])
        assert result.kappa == 1.0
        assert result.observed_agreement == 1.0

    def test_hand_computed_value(self):
        """observed 4/6, expected 0.3611 -> kappa 0.4783."""
        a = ["ok", "ok", "incorrect_answer", "partial_answer", "ok", "incorrect_answer"]
        b = ["ok", "ok", "incorrect_answer", "incorrect_answer", "partial_answer",
             "incorrect_answer"]
        result = cohens_kappa(a, b)
        assert result.observed_agreement == pytest.approx(0.6667, abs=1e-4)
        assert result.expected_agreement == pytest.approx(0.3611, abs=1e-4)
        assert result.kappa == pytest.approx(0.4783, abs=1e-4)

    def test_chance_level_agreement_is_near_zero(self):
        a = ["x", "y"] * 50
        b = ["x", "y", "y", "x"] * 25
        assert abs(cohens_kappa(a, b).kappa) < 0.15

    def test_single_category_makes_kappa_undefined_not_perfect(self):
        """Both annotators used one label for everything: agreement is vacuous."""
        result = cohens_kappa(["ok"] * 20, ["ok"] * 20)
        assert result.kappa is None
        assert result.observed_agreement == 1.0
        assert "undefined" in result.note

    def test_per_category_kappa_is_reported(self):
        a = ["ok", "ok", "incorrect_answer", "partial_answer"]
        b = ["ok", "incorrect_answer", "incorrect_answer", "partial_answer"]
        per = cohens_kappa(a, b).per_category
        assert set(per) == {"ok", "incorrect_answer", "partial_answer"}
        assert per["partial_answer"]["n_both"] == 1

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="different numbers"):
            cohens_kappa(["a"], ["a", "b"])

    def test_empty_input(self):
        result = cohens_kappa([], [])
        assert result.kappa is None
        assert not result.sufficient

    def test_small_sample_flagged(self):
        assert not cohens_kappa(["a", "b"], ["a", "b"]).sufficient


class TestConfusionMatrix:
    def test_counts_and_accuracy(self):
        cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
        assert cm["matrix"]["a"]["a"] == 1
        assert cm["matrix"]["a"]["b"] == 1
        assert cm["accuracy"] == pytest.approx(2 / 3, abs=1e-4)

    def test_per_category_precision_and_recall(self):
        """'a': 1 true positive, 0 false positives, 1 false negative."""
        cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
        stats = cm["per_category"]["a"]
        assert stats["precision"] == 1.0
        assert stats["recall"] == 0.5
        assert stats["f1"] == pytest.approx(2 / 3, abs=1e-4)

    def test_category_never_predicted_has_no_precision(self):
        cm = confusion_matrix(["a", "b"], ["a", "a"])
        assert cm["per_category"]["b"]["precision"] is None
        assert cm["per_category"]["b"]["recall"] == 0.0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            confusion_matrix(["a"], ["a", "b"])


class FakeRow:
    """Minimal stand-in for an EvalRow, for sampling tests."""

    def __init__(self, mode, kfr=None, f1=0.0, faithfulness=None):
        self.failure_mode_v2 = mode
        self.key_fact_recall = kfr
        self.answer_f1_normalized = f1
        self.decision_features = {"faithfulness": faithfulness}


class TestBoundarySampling:
    def test_distance_is_zero_at_a_threshold(self):
        config = TaxonomyConfig()
        row = FakeRow("partial_answer", kfr=config.key_fact_recall_incorrect)
        assert boundary_distance(row, config) == 0.0

    def test_row_far_from_every_threshold(self):
        config = TaxonomyConfig()
        # kfr 0.6 is 0.4 from both kfr thresholds; f1 0.0 is 0.6 from answer_f1_ok
        row = FakeRow("ok", kfr=0.60, f1=0.0)
        assert boundary_distance(row, config) > BOUNDARY_MARGIN

    def test_no_numeric_features_gives_none(self):
        assert boundary_distance(FakeRow("no_retrieval", kfr=None, f1=None), TaxonomyConfig()) \
            is None

    def test_every_mode_gets_a_floor(self):
        rows = (
            [FakeRow("ok", kfr=0.9, f1=0.9)] * 100
            + [FakeRow("incorrect_answer", kfr=0.0, f1=0.0)] * 10
            + [FakeRow("no_retrieval", kfr=None, f1=None)] * 3
        )
        chosen, strata, _ = stratified_sample(
            rows, n_units=40, seed=1, config=TaxonomyConfig(), min_per_mode=3
        )
        for mode in ("ok", "incorrect_answer", "no_retrieval"):
            assert strata[mode]["sampled"] >= min(3, strata[mode]["population"])

    def test_sample_size_is_respected(self):
        rows = [FakeRow("ok", kfr=0.9, f1=0.9)] * 200
        chosen, _, _ = stratified_sample(
            rows, n_units=50, seed=1, config=TaxonomyConfig(), min_per_mode=5
        )
        assert len(chosen) == 50
        assert len(set(chosen)) == 50

    def test_per_mode_floor_may_exceed_the_requested_budget(self):
        """Coverage of every category wins over hitting the target exactly.

        With many modes present, honouring the floor can push the sample past
        `n_units`. That is deliberate: a confusion matrix with an empty row is
        worse than an annotation set slightly larger than requested.
        """
        rows = [FakeRow(f"mode_{i}", kfr=0.5, f1=0.5) for i in range(10) for _ in range(5)]
        chosen, _, _ = stratified_sample(
            rows, n_units=10, seed=1, config=TaxonomyConfig(), min_per_mode=4
        )
        assert len(chosen) == 40 > 10

    def test_sampling_is_deterministic_given_the_seed(self):
        rows = [FakeRow("ok", kfr=i / 100, f1=i / 100) for i in range(100)]
        first, _, _ = stratified_sample(rows, 30, 7, TaxonomyConfig())
        second, _, _ = stratified_sample(rows, 30, 7, TaxonomyConfig())
        assert first == second

    def test_weights_recover_population_proportions(self):
        rows = ([FakeRow("ok", kfr=0.9, f1=0.9)] * 90
                + [FakeRow("incorrect_answer", kfr=0.0)] * 10)
        _, strata, _ = stratified_sample(rows, 40, 3, TaxonomyConfig(), min_per_mode=5)
        for _mode, stats in strata.items():
            if stats["sampled"]:
                assert stats["weight"] == pytest.approx(
                    stats["population"] / stats["sampled"], rel=1e-3
                )


class TestAdjudication:
    def test_agreement_stands(self):
        resolved, unresolved = adjudicate({"u1": "ok"}, {"u1": "ok"}, None)
        assert resolved == {"u1": "ok"}
        assert unresolved == []

    def test_disagreement_stays_open_without_a_third_pass(self):
        resolved, unresolved = adjudicate({"u1": "ok"}, {"u1": "incorrect_answer"}, None)
        assert resolved == {}
        assert unresolved == ["u1"]

    def test_third_pass_resolves(self):
        resolved, unresolved = adjudicate(
            {"u1": "ok"}, {"u1": "incorrect_answer"}, {"u1": "partial_answer"}
        )
        assert resolved == {"u1": "partial_answer"}
        assert unresolved == []

    def test_only_jointly_labelled_units_are_considered(self):
        resolved, unresolved = adjudicate({"u1": "ok", "u2": "ok"}, {"u1": "ok"}, None)
        assert set(resolved) == {"u1"}

    def test_disagreement_pairs_are_counted(self):
        a = {"u1": "ok", "u2": "ok", "u3": "ok"}
        b = {"u1": "incorrect_answer", "u2": "incorrect_answer", "u3": "ok"}
        pairs = disagreement_pairs(a, b)
        assert pairs[0]["count"] == 2
        assert sorted(pairs[0]["categories"]) == ["incorrect_answer", "ok"]


class TestPackageIntegrity:
    """The properties that keep the validation honest."""

    @pytest.fixture(scope="class")
    def package(self, tmp_path_factory):
        records = REPO / "reports" / "experiments" / "qasper_dev_300" / "inference.jsonl"
        if not records.exists():
            pytest.skip("needs a finished experiment run")
        out = tmp_path_factory.mktemp("annotation")
        subprocess.run(
            [sys.executable, "scripts/build_annotation_package.py",
             "--records", str(records), "--out", str(out), "--n-units", "60"],
            cwd=str(REPO), check=True, capture_output=True,
        )
        return out

    def test_sheet_never_contains_the_proposed_label(self, package):
        """Anchoring the annotator would inflate agreement."""
        text = (package / "annotation_sheet.jsonl").read_text(encoding="utf-8")
        for unit in (json.loads(line) for line in text.splitlines() if line.strip()):
            assert unit["human_label"] == ""
            assert "proposed_label" not in unit
            assert "failure_mode" not in unit
            assert "attribution_stage" not in unit

    def test_key_file_holds_the_proposed_labels(self, package):
        key = [json.loads(x) for x in
               (package / "proposed_labels_key.jsonl").read_text(encoding="utf-8").splitlines()
               if x.strip()]
        assert key and all(k["proposed_label"] for k in key)

    def test_both_annotator_sheets_cover_the_same_units(self, package):
        def ids(path):
            return {json.loads(x)["annotation_id"]
                    for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}

        a = ids(package / "annotator_a" / "annotation_sheet.jsonl")
        b = ids(package / "annotator_b" / "annotation_sheet.jsonl")
        assert a == b and len(a) > 0

    def test_annotator_sheets_are_in_different_orders(self, package):
        def order(path):
            return [json.loads(x)["annotation_id"]
                    for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

        assert order(package / "annotator_a" / "annotation_sheet.jsonl") != \
            order(package / "annotator_b" / "annotation_sheet.jsonl")

    def test_manifest_declares_no_labels_collected(self, package):
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        assert "AWAITING HUMAN ANNOTATION" in manifest["status"]
        assert manifest["n_units"] > 0

    def test_labelled_units_ignores_blank_labels(self, package):
        assert labelled_units(package / "annotation_sheet.jsonl") == {}


class TestScorerRefusesToFabricate:
    def test_exits_nonzero_when_no_labels_present(self, tmp_path):
        records = REPO / "reports" / "experiments" / "qasper_dev_300" / "inference.jsonl"
        if not records.exists():
            pytest.skip("needs a finished experiment run")
        out = tmp_path / "pkg"
        subprocess.run(
            [sys.executable, "scripts/build_annotation_package.py",
             "--records", str(records), "--out", str(out), "--n-units", "20"],
            cwd=str(REPO), check=True, capture_output=True,
        )
        result = subprocess.run(
            [sys.executable, "scripts/score_annotations.py", "--package", str(out),
             "--annotator", f"a={out / 'annotator_a' / 'annotation_sheet.jsonl'}",
             "--annotator", f"b={out / 'annotator_b' / 'annotation_sheet.jsonl'}"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        assert result.returncode == 3
        assert "will not invent labels" in result.stderr
        assert not (out / "agreement_report.json").exists()

    def test_scores_synthetic_labels_end_to_end(self, tmp_path):
        """Synthetic labels, purely to prove the scorer runs and reports."""
        records = REPO / "reports" / "experiments" / "qasper_dev_300" / "inference.jsonl"
        if not records.exists():
            pytest.skip("needs a finished experiment run")
        out = tmp_path / "pkg"
        subprocess.run(
            [sys.executable, "scripts/build_annotation_package.py",
             "--records", str(records), "--out", str(out), "--n-units", "40"],
            cwd=str(REPO), check=True, capture_output=True,
        )
        key = {json.loads(x)["annotation_id"]: json.loads(x)["proposed_label"]
               for x in (out / "proposed_labels_key.jsonl").read_text(
                   encoding="utf-8").splitlines() if x.strip()}

        def fill(path, flip_every):
            units = [json.loads(x) for x in
                     path.read_text(encoding="utf-8").splitlines() if x.strip()]
            for i, unit in enumerate(units):
                label = key[unit["annotation_id"]]
                if flip_every and i % flip_every == 0:
                    label = "partial_answer" if label != "partial_answer" else "ok"
                unit["human_label"] = label
            target = path.parent / "completed.jsonl"
            target.write_text(
                "\n".join(json.dumps(u) for u in units) + "\n", encoding="utf-8"
            )
            return target

        a = fill(out / "annotator_a" / "annotation_sheet.jsonl", 0)
        b = fill(out / "annotator_b" / "annotation_sheet.jsonl", 5)
        result = subprocess.run(
            [sys.executable, "scripts/score_annotations.py", "--package", str(out),
             "--annotator", f"a={a}", "--annotator", f"b={b}"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads((out / "agreement_report.json").read_text(encoding="utf-8"))
        # The per-mode floor can push the sample above the requested budget when
        # the run emitted many distinct modes; the manifest records what was
        # actually produced, and that is what both sheets contain.
        n_units = json.loads(
            (out / "manifest.json").read_text(encoding="utf-8")
        )["n_units"]
        assert report["n_jointly_labelled"] == n_units >= 40
        assert report["inter_annotator"]["agreement"]["observed_agreement"] < 1.0
        assert report["adjudication"]["n_unresolved"] > 0
        # Annotator A reproduced the key exactly, so the taxonomy scores perfectly
        # against the units they agreed on. That is a property of the synthetic
        # labels, not a finding.
        assert report["taxonomy_vs_human"]["accuracy"] == 1.0
