"""Tests for evidence-aware alignment and failure attribution (W2)."""
import pytest

from src.evaluation.evidence import (
    AttributionStage,
    EvidenceMode,
    EvidenceStatus,
    GoldSpan,
    RetrievedSpan,
    align_evidence,
    answer_supported_by_evidence,
    attribute_stage,
    retrieved_from_chunks,
    spans_from_records,
)

GOLD = [GoldSpan("doc_a", 100, 200)]


def chunk(rank, doc_id, start, end):
    return RetrievedSpan(rank=rank, doc_id=doc_id, start_char=start, end_char=end)


class TestOverlap:
    def test_exact_containment(self):
        assert align_evidence(GOLD, [chunk(1, "doc_a", 100, 200)]).is_complete

    def test_chunk_strictly_inside_span(self):
        assert align_evidence(GOLD, [chunk(1, "doc_a", 120, 150)]).is_complete

    def test_partial_overlap_at_the_start(self):
        assert align_evidence(GOLD, [chunk(1, "doc_a", 50, 150)]).is_complete

    def test_touching_but_not_overlapping_is_not_coverage(self):
        """Half-open ranges: [0,100) and [100,200) share nothing."""
        result = align_evidence(GOLD, [chunk(1, "doc_a", 0, 100)])
        assert result.status is EvidenceStatus.NONE

    def test_one_character_of_overlap_counts_by_default(self):
        assert align_evidence(GOLD, [chunk(1, "doc_a", 199, 300)]).is_complete

    def test_min_overlap_threshold_is_enforced(self):
        result = align_evidence(GOLD, [chunk(1, "doc_a", 199, 300)], min_overlap_chars=10)
        assert result.status is EvidenceStatus.NONE

    def test_right_document_wrong_location_is_not_coverage(self):
        """The defect document-level metrics cannot see."""
        result = align_evidence(GOLD, [chunk(1, "doc_a", 900, 1000)])
        assert result.status is EvidenceStatus.NONE
        assert result.evidence_recall == 0.0

    def test_right_location_wrong_document_is_not_coverage(self):
        assert align_evidence(GOLD, [chunk(1, "doc_b", 100, 200)]).status is EvidenceStatus.NONE


class TestRecallAndPrecision:
    def test_recall_counts_distinct_spans(self):
        gold = [GoldSpan("d", 0, 10), GoldSpan("d", 100, 110), GoldSpan("d", 200, 210)]
        result = align_evidence(gold, [chunk(1, "d", 0, 10), chunk(2, "d", 100, 110)])
        assert result.evidence_recall == pytest.approx(2 / 3)
        assert result.n_covered_spans == 2

    def test_precision_reports_padding_in_the_context(self):
        result = align_evidence(
            GOLD,
            [chunk(1, "doc_a", 100, 200), chunk(2, "doc_a", 900, 950), chunk(3, "doc_b", 0, 50)],
        )
        assert result.evidence_precision == pytest.approx(1 / 3)

    def test_one_chunk_covering_two_spans_counts_once_for_precision(self):
        gold = [GoldSpan("d", 10, 20), GoldSpan("d", 30, 40)]
        result = align_evidence(gold, [chunk(1, "d", 0, 100)])
        assert result.n_covered_spans == 2
        assert result.evidence_precision == 1.0

    def test_first_evidence_rank_is_the_best_rank(self):
        gold = [GoldSpan("d", 0, 10), GoldSpan("d", 100, 110)]
        result = align_evidence(gold, [chunk(1, "d", 100, 110), chunk(2, "d", 0, 10)])
        assert result.first_evidence_rank == 1

    def test_first_evidence_rank_none_when_nothing_covered(self):
        assert align_evidence(GOLD, [chunk(1, "d", 0, 5)]).first_evidence_rank is None


class TestMultiHop:
    GOLD2 = [GoldSpan("d1", 0, 10), GoldSpan("d2", 0, 10)]

    def test_all_required_needs_every_document(self):
        result = align_evidence(self.GOLD2, [chunk(1, "d1", 0, 10)], EvidenceMode.ALL_REQUIRED)
        assert result.status is EvidenceStatus.PARTIAL
        assert result.missing_doc_ids == ["d2"]

    def test_all_required_satisfied(self):
        result = align_evidence(
            self.GOLD2, [chunk(1, "d1", 0, 10), chunk(2, "d2", 0, 10)], EvidenceMode.ALL_REQUIRED
        )
        assert result.status is EvidenceStatus.COMPLETE
        assert result.missing_doc_ids == []

    def test_any_sufficient_accepts_one_hop(self):
        result = align_evidence(self.GOLD2, [chunk(1, "d1", 0, 10)], EvidenceMode.ANY_SUFFICIENT)
        assert result.status is EvidenceStatus.COMPLETE

    def test_partial_multihop_is_charged_to_retrieval(self):
        result = align_evidence(self.GOLD2, [chunk(1, "d1", 0, 10)], EvidenceMode.ALL_REQUIRED)
        stage, reason = attribute_stage(
            alignment=result, answer_is_correct=False, is_answerable=True,
            abstained=False, n_retrieved=1,
        )
        assert stage is AttributionStage.RETRIEVAL
        assert "d2" in reason


class TestUnanswerableAndDegraded:
    def test_no_gold_spans_is_not_applicable(self):
        result = align_evidence([], [chunk(1, "d", 0, 10)])
        assert result.status is EvidenceStatus.NOT_APPLICABLE
        assert result.evidence_recall is None

    def test_chunks_without_offsets_cannot_be_credited(self):
        result = align_evidence(GOLD, [RetrievedSpan(1, "doc_a", None, None)])
        assert result.status is EvidenceStatus.NONE
        assert result.degraded_to_document_level is True

    def test_degraded_flag_set_even_when_another_chunk_covers(self):
        result = align_evidence(
            GOLD, [chunk(1, "doc_a", 100, 200), RetrievedSpan(2, "doc_a", None, None)]
        )
        assert result.is_complete
        assert result.degraded_to_document_level is True

    def test_empty_retrieval(self):
        result = align_evidence(GOLD, [])
        assert result.status is EvidenceStatus.NONE
        assert result.evidence_precision == 0.0


class TestAttributionHierarchy:
    def _complete(self):
        return align_evidence(GOLD, [chunk(1, "doc_a", 100, 200)])

    def _missing(self):
        return align_evidence(GOLD, [chunk(1, "doc_a", 900, 950)])

    def test_evidence_present_and_answer_correct_is_no_failure(self):
        stage, _ = attribute_stage(
            alignment=self._complete(), answer_is_correct=True, is_answerable=True,
            abstained=False, n_retrieved=1,
        )
        assert stage is AttributionStage.NONE

    def test_evidence_present_and_answer_wrong_is_generation(self):
        stage, reason = attribute_stage(
            alignment=self._complete(), answer_is_correct=False, is_answerable=True,
            abstained=False, n_retrieved=1,
        )
        assert stage is AttributionStage.GENERATION
        assert "retrieved but the answer is wrong" in reason

    def test_evidence_absent_is_retrieval_even_when_the_answer_is_wrong(self):
        stage, _ = attribute_stage(
            alignment=self._missing(), answer_is_correct=False, is_answerable=True,
            abstained=False, n_retrieved=1,
        )
        assert stage is AttributionStage.RETRIEVAL

    def test_correct_answer_without_evidence_is_still_retrieval_not_success(self):
        """A correct answer with no evidence in context indicates memorisation."""
        stage, reason = attribute_stage(
            alignment=self._missing(), answer_is_correct=True, is_answerable=True,
            abstained=False, n_retrieved=1,
        )
        assert stage is AttributionStage.RETRIEVAL
        assert "not supported by retrieved context" in reason

    def test_nothing_retrieved_is_retrieval(self):
        stage, _ = attribute_stage(
            alignment=align_evidence(GOLD, []), answer_is_correct=False, is_answerable=True,
            abstained=False, n_retrieved=0,
        )
        assert stage is AttributionStage.RETRIEVAL

    def test_correct_abstention_is_not_a_failure(self):
        stage, _ = attribute_stage(
            alignment=align_evidence([], []), answer_is_correct=False, is_answerable=False,
            abstained=True, n_retrieved=3,
        )
        assert stage is AttributionStage.NONE

    def test_failure_to_abstain_is_its_own_stage(self):
        stage, reason = attribute_stage(
            alignment=align_evidence([], []), answer_is_correct=False, is_answerable=False,
            abstained=False, n_retrieved=3,
        )
        assert stage is AttributionStage.ABSTENTION
        assert "cannot support" in reason

    def test_answerable_without_gold_spans_falls_back_to_the_answer(self):
        empty = align_evidence([], [chunk(1, "d", 0, 5)])
        assert attribute_stage(
            alignment=empty, answer_is_correct=True, is_answerable=True,
            abstained=False, n_retrieved=1,
        )[0] is AttributionStage.NONE
        assert attribute_stage(
            alignment=empty, answer_is_correct=False, is_answerable=True,
            abstained=False, n_retrieved=1,
        )[0] is AttributionStage.GENERATION


class TestGroundedness:
    def test_correct_and_grounded(self):
        assert answer_supported_by_evidence(align_evidence(GOLD, [chunk(1, "doc_a", 100, 200)]), True)

    def test_correct_but_ungrounded(self):
        assert not answer_supported_by_evidence(
            align_evidence(GOLD, [chunk(1, "doc_a", 900, 950)]), True
        )

    def test_grounded_but_incorrect(self):
        assert not answer_supported_by_evidence(
            align_evidence(GOLD, [chunk(1, "doc_a", 100, 200)]), False
        )


class TestAdapters:
    def test_spans_from_records(self):
        spans = spans_from_records([{"doc_id": "d", "start_char": 1, "end_char": 5}])
        assert spans == [GoldSpan("d", 1, 5)]

    def test_retrieved_from_chunks_reads_offsets(self):
        from src.evaluation.records import RetrievedChunk

        chunks = [RetrievedChunk(rank=1, chunk_id="c", doc_id="d", source="s",
                                 score=0.5, text="t", start_char=10, end_char=20)]
        assert retrieved_from_chunks(chunks) == [RetrievedSpan(1, "d", 10, 20)]

    def test_retrieved_from_chunks_tolerates_missing_offsets(self):
        from src.evaluation.records import RetrievedChunk

        chunks = [RetrievedChunk(rank=1, chunk_id="c", doc_id="d", source="s", score=0.5)]
        assert retrieved_from_chunks(chunks)[0].has_offsets is False

    def test_alignment_serialises(self):
        data = align_evidence(GOLD, [chunk(1, "doc_a", 100, 200)]).as_dict()
        assert data["status"] == "complete"
        assert data["evidence_recall"] == 1.0
