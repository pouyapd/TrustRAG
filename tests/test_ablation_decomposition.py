"""Tests for the retrieval-definition decomposition.

An earlier version of the ablation compared the conventional document-level
definition directly against the span-level one. On multi-hop data that changes
two things at once — granularity (document to span) and quantifier (any
relevant document to every required one) — so the resulting gap could not be
attributed to either, and it was in fact reported as a granularity effect when
on HotpotQA it is entirely a quantifier effect.

These tests pin the separation with hand-built records whose expected values
are unambiguous by construction.
"""
import pytest

from scripts.run_ablation import (
    compare,
    condition_a_document_any,
    condition_b_document_quantified,
    condition_c_span,
    paired_step,
)
from src.evaluation.records import InferenceRecord, RetrievedChunk
from src.evaluation.runner import score_records
from src.evaluation.taxonomy import TaxonomyConfig


def record(retrieved, gold, mode="any_sufficient", n_relevant_chunks=None, answer="x"):
    """A record with hand-specified retrieval and gold evidence.

    `retrieved` is a list of (doc_id, start, end); `gold` likewise.
    """
    return InferenceRecord(
        index=1,
        question="q",
        reference_answer="ref",
        relevant_doc_ids=sorted({d for d, _, _ in gold}),
        predicted_answer=answer,
        retrieved=[
            RetrievedChunk(rank=i, chunk_id=f"c{i}", doc_id=d, source="s",
                           score=1.0, text="t", start_char=s, end_char=e)
            for i, (d, s, e) in enumerate(retrieved, start=1)
        ],
        faithfulness=1.0,
        latency_ms=1.0,
        top_k=len(retrieved) or 5,
        n_relevant_chunks=n_relevant_chunks,
        metadata={
            "supporting_spans": [
                {"doc_id": d, "start_char": s, "end_char": e} for d, s, e in gold
            ],
            "evidence_mode": mode,
            "answers": ["ref"],
        },
    )


class TestConditionsAreDistinct:
    def test_wrong_place_in_right_document_splits_a_from_c(self):
        """The granularity case: document retrieved, evidence not."""
        r = record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)])
        assert condition_a_document_any(r) is True
        assert condition_b_document_quantified(r) is True   # same, single-hop
        assert condition_c_span(r) is False                 # granularity bites

    def test_evidence_actually_retrieved_satisfies_all_three(self):
        r = record(retrieved=[("d1", 480, 620)], gold=[("d1", 500, 600)])
        assert (condition_a_document_any(r), condition_b_document_quantified(r),
                condition_c_span(r)) == (True, True, True)

    def test_multi_hop_partial_splits_a_from_b(self):
        """The quantifier case: one of two required documents retrieved."""
        r = record(retrieved=[("d1", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 0, 100)], mode="all_required")
        assert condition_a_document_any(r) is True    # a relevant doc was retrieved
        assert condition_b_document_quantified(r) is False  # but not all of them
        assert condition_c_span(r) is False

    def test_multi_hop_complete_satisfies_all_three(self):
        r = record(retrieved=[("d1", 0, 100), ("d2", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 0, 100)], mode="all_required")
        assert (condition_a_document_any(r), condition_b_document_quantified(r),
                condition_c_span(r)) == (True, True, True)

    def test_single_hop_a_and_b_are_identical_by_definition(self):
        """Under any_sufficient the quantifier step must be exactly zero."""
        cases = [
            record(retrieved=[("d1", 0, 10)], gold=[("d1", 0, 10)]),
            record(retrieved=[("d1", 0, 10)], gold=[("d1", 500, 600)]),
            record(retrieved=[("dX", 0, 10)], gold=[("d1", 0, 10)]),
            record(retrieved=[("d1", 0, 10)], gold=[("d1", 0, 10), ("d2", 0, 10)]),
        ]
        for r in cases:
            assert condition_a_document_any(r) == condition_b_document_quantified(r)

    def test_unanswerable_rows_are_excluded_from_every_condition(self):
        r = record(retrieved=[("d1", 0, 10)], gold=[])
        assert condition_a_document_any(r) is None
        assert condition_b_document_quantified(r) is None
        assert condition_c_span(r) is None


class TestDecomposition:
    def test_single_hop_gap_is_entirely_granularity(self):
        records = [
            record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)]),
            record(retrieved=[("d1", 480, 620)], gold=[("d1", 500, 600)]),
        ]
        result = compare(records, score_records(records, TaxonomyConfig()))
        assert result["steps"]["quantifier_A_to_B"]["absolute_gap_pp"] == 0.0
        assert result["steps"]["granularity_B_to_C"]["absolute_gap_pp"] == 50.0
        assert result["steps"]["total_A_to_C"]["absolute_gap_pp"] == 50.0

    def test_multi_hop_gap_can_be_entirely_quantifier(self):
        """Documents that are one chunk long cannot show a granularity effect.

        This is the HotpotQA situation: reporting the total as a granularity
        result would attribute the effect to the wrong mechanism.
        """
        records = [
            record(retrieved=[("d1", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 0, 100)], mode="all_required"),
            record(retrieved=[("d1", 0, 100), ("d2", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 0, 100)], mode="all_required"),
        ]
        result = compare(records, score_records(records, TaxonomyConfig()))
        assert result["steps"]["quantifier_A_to_B"]["absolute_gap_pp"] == 50.0
        assert result["steps"]["granularity_B_to_C"]["absolute_gap_pp"] == 0.0

    def test_steps_sum_to_the_total(self):
        records = [
            record(retrieved=[("d1", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 500, 600)], mode="all_required"),
            record(retrieved=[("d1", 0, 100), ("d2", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 500, 600)], mode="all_required"),
            record(retrieved=[("d1", 0, 100), ("d2", 480, 620)],
                   gold=[("d1", 0, 100), ("d2", 500, 600)], mode="all_required"),
        ]
        result = compare(records, score_records(records, TaxonomyConfig()))
        steps = result["steps"]
        # Additivity is exact in the underlying rates; each step is rounded to
        # two decimals independently, so the sum of two rounded steps can differ
        # from the rounded total by up to three half-ulps.
        assert steps["quantifier_A_to_B"]["absolute_gap_pp"] + steps["granularity_B_to_C"][
            "absolute_gap_pp"
        ] == pytest.approx(steps["total_A_to_C"]["absolute_gap_pp"], abs=0.02)

    def test_evidence_level_is_never_more_lenient(self):
        """C implies B implies A, for every record. The ordering is structural."""
        records = [
            record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)]),
            record(retrieved=[("d1", 480, 620)], gold=[("d1", 500, 600)]),
            record(retrieved=[("dX", 0, 10)], gold=[("d1", 0, 10)]),
            record(retrieved=[("d1", 0, 100)],
                   gold=[("d1", 0, 100), ("d2", 0, 100)], mode="all_required"),
        ]
        for r in records:
            a, b, c = (condition_a_document_any(r), condition_b_document_quantified(r),
                       condition_c_span(r))
            assert not (c and not b), "span success without document success"
            assert not (b and not a), "quantified success without any-document success"

    def test_covariate_is_reported(self):
        records = [record(retrieved=[("d1", 0, 10)], gold=[("d1", 0, 10)],
                          n_relevant_chunks=19)]
        result = compare(records, score_records(records, TaxonomyConfig()))
        assert result["median_chunks_per_relevant_document"] == 19
        assert result["evidence_modes"] == {"any_sufficient": 1}


class TestPairedStep:
    def test_no_discordant_pairs_reports_no_p_value(self):
        """0/0 discordant must not fabricate a p-value."""
        step = paired_step([True, False], [True, False], "identical")
        assert step["absolute_gap_pp"] == 0.0
        assert step["mcnemar"]["p_value"] is None
        assert step["discordant_lost"] == 0

    def test_relative_gap_is_the_fraction_of_successes_lost(self):
        step = paired_step([True, True, True, True], [True, True, False, False], "x")
        assert step["rate_before"] == 1.0
        assert step["rate_after"] == 0.5
        assert step["absolute_gap_pp"] == 50.0
        assert step["relative_gap"] == 0.5

    def test_relative_gap_is_none_when_nothing_succeeded(self):
        assert paired_step([False, False], [False, False], "x")["relative_gap"] is None


class TestEvidenceAwareTaxonomy:
    """The taxonomy's R4 gate and the evidence attribution must be able to agree.

    `failure_mode_v2` gates R4 on whether a relevant *document* was retrieved,
    so a row whose document arrived but whose evidence did not passes R4 and is
    then labelled by answer quality — a generation failure. On the reported
    runs that affects 15% of QASPER rows, 22% of NQ and 41% of HotpotQA.
    `failure_mode_evidence` re-runs the same rules with an evidence-level gate.
    """

    def test_document_retrieved_but_evidence_missed_splits_the_two_labels(self):
        r = record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)], answer="totally wrong")
        row = score_records([r], TaxonomyConfig())[0]
        # Document-level gate: R4 passes, so the row is judged on answer quality.
        assert row.failure_stage_v2 == "generation"
        # Evidence-level gate: R4 fires, so the row is charged to retrieval.
        assert row.failure_mode_evidence == "wrong_retrieval"
        assert row.failure_stage_evidence == "retrieval"
        # And it agrees with the independent evidence attribution.
        assert row.attribution_stage == "retrieval"

    def test_labels_coincide_when_the_evidence_was_actually_retrieved(self):
        r = record(retrieved=[("d1", 480, 620)], gold=[("d1", 500, 600)], answer="wrong")
        row = score_records([r], TaxonomyConfig())[0]
        assert row.failure_mode_v2 == row.failure_mode_evidence
        assert row.failure_stage_evidence == "generation"

    def test_v2_label_is_unchanged_by_the_new_field(self):
        """Backward compatibility: the frozen label must not move."""
        r = record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)], answer="wrong")
        row = score_records([r], TaxonomyConfig())[0]
        from src.evaluation.taxonomy import classify_v2

        expected = classify_v2(
            answer="wrong", reference_answer="ref",
            retrieved_doc_ids=["d1"], relevant_doc_ids=["d1"], faithfulness_score=1.0,
        )
        assert row.failure_mode_v2 == expected.mode.value

    def test_rows_without_gold_evidence_reuse_the_v2_label(self):
        r = record(retrieved=[("d1", 0, 100)], gold=[], answer="anything")
        row = score_records([r], TaxonomyConfig())[0]
        assert row.failure_mode_evidence == row.failure_mode_v2

    def test_aggregate_reports_the_relabelling_count(self):
        from src.evaluation.runner import aggregate

        records = [
            record(retrieved=[("d1", 0, 100)], gold=[("d1", 500, 600)], answer="wrong"),
            record(retrieved=[("d1", 480, 620)], gold=[("d1", 500, 600)], answer="wrong"),
        ]
        report = aggregate(score_records(records, TaxonomyConfig()))
        assert "failure_modes_evidence" in report
        assert report["taxonomy_agreement"]["generation_relabelled_as_retrieval"] == 1
