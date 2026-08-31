"""Tests for figure generation.

Figures are a publication artifact, so the failure that matters is a silent one:
a plot drawn from missing data, or error bars that do not correspond to the
interval they claim. The pure helpers are tested directly; the end-to-end build
is exercised only when matplotlib is installed, since it is an optional
research dependency rather than part of the service.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_figures import FIGURE_BUILDERS, load_decomp, wilson_bounds  # noqa: E402


class TestWilsonBounds:
    def test_offsets_are_distances_from_the_point(self):
        low, high = wilson_bounds({"lower": 0.4, "upper": 0.6}, 0.5)
        assert low == pytest.approx(0.1)
        assert high == pytest.approx(0.1)

    def test_asymmetric_interval_is_preserved(self):
        """Wilson intervals are asymmetric near 0 and 1; symmetrising would lie."""
        low, high = wilson_bounds({"lower": 0.80, "upper": 0.99}, 0.95)
        assert low == pytest.approx(0.15)
        assert high == pytest.approx(0.04)

    def test_missing_interval_gives_no_error_bar(self):
        assert wilson_bounds({"lower": None, "upper": None}, 0.5) == (0.0, 0.0)

    def test_never_negative(self):
        """A point outside its own interval must not produce a negative bar."""
        low, high = wilson_bounds({"lower": 0.6, "upper": 0.7}, 0.5)
        assert low >= 0.0 and high >= 0.0


class TestLoadDecomp:
    def test_missing_run_returns_none_rather_than_raising(self):
        """A figure must skip cleanly when its inputs have not been produced."""
        assert load_decomp("a_run_that_was_never_executed") is None

    def test_reads_the_comparison_block(self, tmp_path, monkeypatch):
        from scripts import make_figures

        payload = {"comparison": {"n_paired": 7, "conditions": {"A_document_any": 0.5}}}
        reports = tmp_path / "experiments"
        reports.mkdir(parents=True)
        (reports / "decomp_demo.json").write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(make_figures, "REPORTS", reports)
        assert make_figures.load_decomp("demo")["n_paired"] == 7


class TestBuilders:
    def test_every_builder_is_addressable_from_the_cli(self):
        assert set(FIGURE_BUILDERS) == {"abc", "topk", "embedders", "attribution"}

    def test_builders_return_false_when_data_is_absent(self, tmp_path, monkeypatch):
        """No inputs must mean 'skipped', not a blank or fabricated figure."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from scripts import make_figures

        monkeypatch.setattr(make_figures, "REPORTS", tmp_path / "nothing")
        monkeypatch.setattr(make_figures, "FIGURES", tmp_path / "figures")
        for name, builder in FIGURE_BUILDERS.items():
            assert builder(plt) is False, f"{name} drew a figure from no data"
        assert not (tmp_path / "figures").exists()


class TestEndToEnd:
    def test_builds_from_real_runs_when_present(self, tmp_path, monkeypatch):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from scripts import make_figures

        repo_reports = Path(__file__).resolve().parent.parent / "reports" / "experiments"
        if not (repo_reports / "decomp_qasper_dev_300.json").exists():
            pytest.skip("needs a finished experiment run")

        monkeypatch.setattr(make_figures, "FIGURES", tmp_path / "figures")
        assert make_figures.figure_abc(plt) is True
        assert (tmp_path / "figures" / "abc_decomposition.png").exists()
        assert (tmp_path / "figures" / "abc_decomposition.pdf").exists()
