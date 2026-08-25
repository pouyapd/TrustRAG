"""Tests for failure taxonomy v2.

Covers every category, the ordering decisions that are easy to get wrong, and
the threshold boundaries.
"""
import pytest

from src.evaluation.taxonomy import (
    NON_FAILURE_MODES,
    STAGE_ATTRIBUTION,
    FailureModeV2,
    TaxonomyConfig,
    classify_features,
    classify_v2,
    extract_features,
    is_failure,
)

REFUSAL = "I cannot answer this from the provided context."


def classify(**kwargs):
    """Call classify_v2 with sensible defaults for anything unspecified."""
    params = {
        "answer": "The refund window is 30 days from the date of purchase.",
        "reference_answer": "30 days from the date of purchase.",
        "retrieved_doc_ids": ["refund_policy"],
        "relevant_doc_ids": ["refund_policy"],
        "faithfulness_score": 1.0,
    }
    params.update(kwargs)
    return classify_v2(**params)


class TestAnswerableQuestions:
    def test_correct_answer_is_ok(self):
        diag = classify()
        assert diag.mode == FailureModeV2.OK
        assert not is_failure(diag.mode)

    def test_verbose_but_complete_answer_is_ok(self):
        """All reference facts present, buried in extra text -> R8, not a failure."""
        diag = classify(
            answer=(
                "Here is what the policy says. Annual subscribers have a 30-day refund "
                "window from the date of purchase, processed within 5-7 business days."
            )
        )
        assert diag.mode == FailureModeV2.OK
        assert diag.rule_id == "R8"

    def test_incorrect_answer(self):
        """Right document retrieved, answer asserts something else entirely."""
        diag = classify(
            answer="The Pro plan allows 1000 API requests per minute.",
            reference_answer="29 EUR per month.",
        )
        assert diag.mode == FailureModeV2.INCORRECT_ANSWER
        assert diag.rule_id == "R9"
        assert diag.stage == "generation"

    def test_partial_answer(self):
        """On topic, consistent, but omits half the reference."""
        diag = classify(
            answer="Click 'Forgot password' on the login screen.",
            reference_answer=(
                "Click 'Forgot password' on the login screen to receive a reset link "
                "valid for 30 minutes."
            ),
        )
        assert diag.mode == FailureModeV2.PARTIAL_ANSWER
        assert diag.rule_id == "R11"

    def test_refusal_when_answerable(self):
        diag = classify(answer=REFUSAL)
        assert diag.mode == FailureModeV2.REFUSAL_WHEN_ANSWERABLE
        assert diag.stage == "generation"

    def test_hallucination_when_unfaithful(self):
        diag = classify(faithfulness_score=0.1)
        assert diag.mode == FailureModeV2.HALLUCINATION

    def test_faithfulness_none_skips_hallucination_rule(self):
        diag = classify(faithfulness_score=None)
        assert diag.mode == FailureModeV2.OK


class TestRetrievalFailures:
    def test_no_retrieval(self):
        diag = classify(retrieved_doc_ids=[])
        assert diag.mode == FailureModeV2.NO_RETRIEVAL
        assert diag.rule_id == "R1"
        assert diag.stage == "retrieval"

    def test_wrong_retrieval(self):
        diag = classify(retrieved_doc_ids=["unrelated_doc"])
        assert diag.mode == FailureModeV2.WRONG_RETRIEVAL
        assert diag.stage == "retrieval"

    def test_refusal_after_wrong_retrieval_blames_retrieval(self):
        """Refusing without relevant context is correct behaviour by the generator.

        R4 is deliberately checked before R5 so this row is attributed to
        retrieval rather than penalising the model for behaving well.
        """
        diag = classify(answer=REFUSAL, retrieved_doc_ids=["unrelated_doc"])
        assert diag.mode == FailureModeV2.WRONG_RETRIEVAL
        assert diag.stage == "retrieval"


class TestUnanswerableQuestions:
    """The category v1 could not express at all."""

    def test_correct_abstention_is_not_a_failure(self):
        diag = classify(
            answer=REFUSAL,
            reference_answer=REFUSAL,
            relevant_doc_ids=[],
        )
        assert diag.mode == FailureModeV2.OK_ABSTAINED
        assert diag.rule_id == "R2"
        assert not is_failure(diag.mode)

    def test_failure_to_abstain(self):
        diag = classify(
            answer="The Starter plan includes 10 GB of storage.",
            reference_answer=REFUSAL,
            relevant_doc_ids=[],
        )
        assert diag.mode == FailureModeV2.ANSWERED_WHEN_UNANSWERABLE
        assert diag.rule_id == "R3"
        assert is_failure(diag.mode)

    def test_failure_to_abstain_beats_answer_similarity(self):
        """No answer-quality rule may fire on an unanswerable question."""
        diag = classify(
            answer=REFUSAL.replace("cannot", "can not really"),
            reference_answer=REFUSAL,
            relevant_doc_ids=[],
        )
        assert diag.mode == FailureModeV2.ANSWERED_WHEN_UNANSWERABLE


class TestThresholdBoundaries:
    def test_faithfulness_exactly_at_threshold_is_not_hallucination(self):
        cfg = TaxonomyConfig(faithfulness_threshold=0.6)
        diag = classify(faithfulness_score=0.6)
        assert diag.mode != FailureModeV2.HALLUCINATION
        diag_below = classify_v2(
            answer="The refund window is 30 days from the date of purchase.",
            reference_answer="30 days from the date of purchase.",
            retrieved_doc_ids=["refund_policy"],
            relevant_doc_ids=["refund_policy"],
            faithfulness_score=0.5999,
            config=cfg,
        )
        assert diag_below.mode == FailureModeV2.HALLUCINATION

    def test_key_fact_recall_boundary_is_inclusive_for_incorrect(self):
        features = extract_features(
            answer="totally unrelated wording",
            reference_answer="alpha beta gamma delta epsilon",
            retrieved_doc_ids=["d"],
            relevant_doc_ids=["d"],
            faithfulness_score=1.0,
        )
        assert features.key_fact_recall == 0.0
        strict = classify_features(features, TaxonomyConfig(key_fact_recall_incorrect=0.0))
        assert strict.mode == FailureModeV2.INCORRECT_ANSWER

    def test_threshold_change_moves_a_row_between_categories(self):
        """The same features must be able to yield different labels."""
        features = extract_features(
            answer="alpha beta",
            reference_answer="alpha beta gamma delta epsilon zeta eta theta",
            retrieved_doc_ids=["d"],
            relevant_doc_ids=["d"],
            faithfulness_score=1.0,
        )
        lenient = classify_features(features, TaxonomyConfig(key_fact_recall_incorrect=0.0))
        strict = classify_features(features, TaxonomyConfig(key_fact_recall_incorrect=0.5))
        assert lenient.mode == FailureModeV2.PARTIAL_ANSWER
        assert strict.mode == FailureModeV2.INCORRECT_ANSWER

    def test_missing_key_fact_beats_a_high_f1(self):
        """A fluent answer that drops a reference fact is not `ok`.

        R8/R9 are checked before the F1 shortcut precisely so this row is
        reported as incomplete rather than passing on shared wording.
        """
        features = extract_features(
            answer="alpha beta gamma",
            reference_answer="alpha beta gamma delta",
            retrieved_doc_ids=["d"],
            relevant_doc_ids=["d"],
            faithfulness_score=1.0,
        )
        assert features.answer_f1 > 0.8
        assert classify_features(features, TaxonomyConfig(answer_f1_ok=0.5)).mode == (
            FailureModeV2.PARTIAL_ANSWER
        )

    def test_answer_f1_admits_ok_when_reference_has_no_key_facts(self):
        """R7 is the fallback for references with nothing extractable."""
        features = extract_features(
            answer="it is so",
            reference_answer="it is so",
            retrieved_doc_ids=["d"],
            relevant_doc_ids=["d"],
            faithfulness_score=1.0,
        )
        assert features.num_key_facts == 0
        diag = classify_features(features, TaxonomyConfig(answer_f1_ok=0.5))
        assert diag.mode == FailureModeV2.OK
        assert diag.rule_id == "R7"

    def test_reference_without_key_facts_falls_back_to_f1(self):
        diag = classify(answer="zzz qqq", reference_answer="it is")
        assert diag.mode == FailureModeV2.INCORRECT_ANSWER
        assert diag.rule_id == "R10"


class TestConfigAndFeatures:
    def test_fingerprint_is_stable(self):
        assert TaxonomyConfig().fingerprint() == TaxonomyConfig().fingerprint()

    def test_fingerprint_changes_with_thresholds(self):
        assert TaxonomyConfig().fingerprint() != TaxonomyConfig(answer_f1_ok=0.9).fingerprint()

    def test_features_are_recorded(self):
        diag = classify()
        features = diag.features.as_dict()
        for key in (
            "is_answerable",
            "num_retrieved",
            "retrieval_hit",
            "abstained",
            "faithfulness",
            "answer_f1",
            "key_fact_recall",
            "num_key_facts",
        ):
            assert key in features

    def test_diagnosis_carries_version_and_fingerprint(self):
        cfg = TaxonomyConfig()
        diag = classify_features(
            extract_features(
                answer="a",
                reference_answer="a",
                retrieved_doc_ids=["d"],
                relevant_doc_ids=["d"],
                faithfulness_score=1.0,
            ),
            cfg,
        )
        assert diag.taxonomy_version == cfg.version
        assert diag.config_fingerprint == cfg.fingerprint()

    def test_every_mode_has_a_stage(self):
        for mode in FailureModeV2:
            assert mode in STAGE_ATTRIBUTION

    def test_non_failure_modes_are_not_failures(self):
        for mode in NON_FAILURE_MODES:
            assert not is_failure(mode)
            assert STAGE_ATTRIBUTION[mode] == "none"

    def test_is_failure_accepts_strings(self):
        assert is_failure("incorrect_answer")
        assert not is_failure("ok")

    def test_unknown_mode_string_rejected(self):
        with pytest.raises(ValueError):
            is_failure("not_a_real_mode")
