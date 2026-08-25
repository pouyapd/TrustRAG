"""Tests for the inference/scoring split.

The claim being tested: once a run has produced inference records, scoring and
classification can be repeated with different thresholds and *no model is ever
called*. The `ExplodingLLM` fixture makes that claim enforceable rather than
aspirational.
"""
import json

import pytest

from src.evaluation.records import (
    InferenceRecord,
    RetrievedChunk,
    read_records,
    write_records,
)
from src.evaluation.runner import aggregate, score_record, score_records, write_outputs
from src.evaluation.taxonomy import FailureModeV2, TaxonomyConfig
from src.rag.providers import LLMProvider

REFUSAL = "I cannot answer this from the provided context."


class ExplodingLLM(LLMProvider):
    """Fails the test if anything tries to generate during scoring."""

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise AssertionError("scoring must not call an LLM")


def make_record(
    index: int = 1,
    question: str = "What is the refund window for annual subscribers?",
    reference: str = "30 days from the date of purchase.",
    predicted: str = "Annual subscribers have a 30-day refund window from the date of purchase.",
    relevant: list[str] | None = None,
    retrieved: list[str] | None = None,
    faithfulness: float | None = 1.0,
    n_relevant_chunks: int | None = 1,
) -> InferenceRecord:
    relevant = ["refund_policy"] if relevant is None else relevant
    retrieved = ["refund_policy", "support_faq"] if retrieved is None else retrieved
    return InferenceRecord(
        index=index,
        question=question,
        reference_answer=reference,
        relevant_doc_ids=relevant,
        predicted_answer=predicted,
        retrieved=[
            RetrievedChunk(rank=i, chunk_id=f"{d}_0", doc_id=d, source=f"{d}.md", score=0.9 - i * 0.1, text=f"text of {d}")
            for i, d in enumerate(retrieved, start=1)
        ],
        faithfulness=faithfulness,
        latency_ms=1.5,
        top_k=4,
        n_relevant_chunks=n_relevant_chunks,
    )


class TestScoringWithoutInference:
    def test_scoring_calls_no_model(self):
        """The whole point of Stage 1: re-scoring is free."""
        rows = score_records([make_record()], TaxonomyConfig())
        assert rows[0].failure_mode_v2 == FailureModeV2.OK.value

    def test_scoring_needs_no_pipeline_at_all(self):
        """No vector store, no embedder, no LLM is constructed anywhere here."""
        llm = ExplodingLLM()
        rows = score_records([make_record(), make_record(index=2)])
        assert len(rows) == 2
        with pytest.raises(AssertionError):
            llm.generate("x", "y")  # the guard itself works

    def test_rescoring_with_new_thresholds_changes_labels(self):
        record = make_record(
            predicted="Click 'Forgot password' on the login screen.",
            reference=(
                "Click 'Forgot password' on the login screen to receive a reset link "
                "valid for 30 minutes."
            ),
        )
        lenient = score_record(record, TaxonomyConfig(key_fact_recall_incorrect=0.0))
        strict = score_record(record, TaxonomyConfig(key_fact_recall_incorrect=0.9))
        assert lenient.failure_mode_v2 == FailureModeV2.PARTIAL_ANSWER.value
        assert strict.failure_mode_v2 == FailureModeV2.INCORRECT_ANSWER.value

    def test_rescoring_is_deterministic(self):
        record = make_record()
        assert score_record(record).failure_mode_v2 == score_record(record).failure_mode_v2

    def test_config_fingerprint_recorded_on_every_row(self):
        cfg = TaxonomyConfig(answer_f1_ok=0.75)
        row = score_record(make_record(), cfg)
        assert row.taxonomy_config_fingerprint == cfg.fingerprint()

    def test_decision_features_are_stored(self):
        row = score_record(make_record())
        assert row.decision_features["retrieval_hit"] is True
        assert row.decision_features["num_key_facts"] > 0


class TestLegacyFieldsPreserved:
    def test_legacy_and_v2_labels_both_present(self):
        row = score_record(make_record())
        assert row.failure_mode  # v1
        assert row.failure_mode_v2  # v2
        assert row.failure_rule_v2

    def test_legacy_recall_still_zero_for_unanswerable(self):
        """v1 behaviour is frozen even though v2 reports None."""
        row = score_record(make_record(relevant=[], reference=REFUSAL, n_relevant_chunks=None))
        assert row.recall_at_k == 0.0
        assert row.doc_recall_at_k is None

    def test_unanswerable_row_is_flagged(self):
        row = score_record(make_record(relevant=[], reference=REFUSAL, n_relevant_chunks=None))
        assert row.is_answerable is False
        assert row.failure_mode_v2 == FailureModeV2.ANSWERED_WHEN_UNANSWERABLE.value


class TestRecordRoundTrip:
    def test_write_then_read(self, tmp_path):
        records = [make_record(), make_record(index=2, relevant=[], reference=REFUSAL)]
        path = write_records(records, tmp_path / "inference.jsonl")
        loaded = read_records(path)
        assert len(loaded) == 2
        assert loaded[0].question == records[0].question
        assert loaded[0].retrieved[0].doc_id == "refund_policy"
        assert loaded[1].is_answerable is False

    def test_round_trip_preserves_scores(self, tmp_path):
        path = write_records([make_record()], tmp_path / "r.jsonl")
        loaded = read_records(path)
        assert score_record(loaded[0]).failure_mode_v2 == score_record(make_record()).failure_mode_v2

    def test_records_are_valid_jsonl(self, tmp_path):
        path = write_records([make_record()], tmp_path / "r.jsonl")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["schema_version"]
        assert lines[0]["retrieved"][0]["rank"] == 1

    def test_null_faithfulness_survives(self, tmp_path):
        path = write_records([make_record(faithfulness=None)], tmp_path / "r.jsonl")
        assert read_records(path)[0].faithfulness is None


class TestReportStructure:
    def _dataset_rows(self):
        return score_records(
            [
                make_record(index=1),
                make_record(
                    index=2,
                    predicted="The Pro plan allows 1000 API requests per minute.",
                    reference="29 EUR per month.",
                ),
                make_record(
                    index=3,
                    relevant=[],
                    reference=REFUSAL,
                    predicted="The Starter plan includes 10 GB of storage.",
                    n_relevant_chunks=None,
                ),
                make_record(index=4, predicted=REFUSAL),
            ]
        )

    def test_legacy_keys_present_and_first(self):
        report = aggregate(self._dataset_rows())
        legacy = [
            "total",
            "precision_at_k_mean",
            "recall_at_k_mean",
            "mrr_mean",
            "token_overlap_mean",
            "faithfulness_mean",
            "latency_ms_mean",
            "failure_rate",
            "failure_modes",
        ]
        assert list(report.keys())[: len(legacy)] == legacy

    def test_v2_blocks_present(self):
        report = aggregate(self._dataset_rows())
        for key in (
            "failure_modes_v2",
            "failure_rate_v2",
            "attribution",
            "retrieval_corrected",
            "answer_corrected",
            "abstention",
            "confidence_intervals",
            "statistical_notes",
        ):
            assert key in report

    def test_abstention_failure_is_counted(self):
        report = aggregate(self._dataset_rows())
        assert report["abstention"]["n_unanswerable"] == 1
        assert report["abstention"]["false_answer_rate"] == 1.0

    def test_unanswerable_excluded_from_corrected_retrieval(self):
        report = aggregate(self._dataset_rows())
        assert report["retrieval_corrected"]["n_answerable"] == 3
        assert report["retrieval_corrected"]["n_unanswerable"] == 1

    def test_statistical_notes_flag_small_sample(self):
        notes = " ".join(aggregate(self._dataset_rows())["statistical_notes"])
        assert "below the n=30" in notes

    def test_empty_rows(self):
        assert aggregate([]) == {"total": 0}

    def test_write_outputs_emits_records(self, tmp_path):
        records = [make_record()]
        rows = score_records(records)
        write_outputs(rows, aggregate(rows), tmp_path, records=records)
        for name in ("rows.jsonl", "summary.json", "report.md", "inference.jsonl"):
            assert (tmp_path / name).exists()

    def test_write_outputs_without_records_skips_file(self, tmp_path):
        rows = score_records([make_record()])
        write_outputs(rows, aggregate(rows), tmp_path)
        assert not (tmp_path / "inference.jsonl").exists()


class TestProvenance:
    def test_collect_provenance_has_expected_shape(self):
        from src.evaluation.provenance import collect_provenance

        prov = collect_provenance(dataset={"size": 20})
        assert prov["timestamp_utc"]
        assert "commit" in prov["git"]
        assert prov["python"]
        assert "chromadb" in prov["packages"]
        assert prov["dataset"]["size"] == 20

    def test_git_lookup_never_raises_outside_a_repo(self, tmp_path):
        from src.evaluation.provenance import git_info

        info = git_info(tmp_path)
        assert info["commit"] in ("unavailable",) or isinstance(info["commit"], str)

    def test_git_lookup_survives_missing_git_binary(self, monkeypatch, tmp_path):
        """The Docker image has no git and no .git directory.

        Provenance capture must degrade to 'unavailable' rather than taking an
        evaluation down with it.
        """
        import subprocess

        from src.evaluation import provenance

        def no_git(*args, **kwargs):
            raise FileNotFoundError("git not installed")

        monkeypatch.setattr(subprocess, "run", no_git)
        info = provenance.git_info(tmp_path)
        assert info["commit"] == "unavailable"
        assert info["dirty"] is False
        assert provenance.collect_provenance()["git"]["branch"] == "unavailable"

    def test_describe_component_reports_model(self):
        from src.evaluation.provenance import describe_component

        class Fake:
            model = "gpt-4o-mini"

        assert describe_component(Fake()) == "Fake(gpt-4o-mini)"
        assert describe_component(None) == "unavailable"
