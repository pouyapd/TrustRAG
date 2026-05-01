"""Tests for failure mode classification."""
from src.evaluation.failure_modes import FailureMode, classify_failure


def test_no_retrieval():
    diag = classify_failure(
        question="What is X?",
        answer="X is...",
        retrieved_doc_ids=[],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=None,
        token_overlap_score=0.0,
    )
    assert diag.mode == FailureMode.NO_RETRIEVAL


def test_wrong_retrieval():
    diag = classify_failure(
        question="What is X?",
        answer="X is...",
        retrieved_doc_ids=["doc_z", "doc_y"],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=0.9,
        token_overlap_score=0.5,
    )
    assert diag.mode == FailureMode.WRONG_RETRIEVAL


def test_refusal_when_answerable():
    diag = classify_failure(
        question="What is X?",
        answer="I cannot answer this from the provided context.",
        retrieved_doc_ids=["doc_a"],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=0.9,
        token_overlap_score=0.5,
    )
    assert diag.mode == FailureMode.REFUSAL_WHEN_ANSWERABLE


def test_hallucination():
    diag = classify_failure(
        question="What is X?",
        answer="X is something completely made up.",
        retrieved_doc_ids=["doc_a"],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=0.2,
        token_overlap_score=0.5,
    )
    assert diag.mode == FailureMode.HALLUCINATION


def test_partial_answer():
    diag = classify_failure(
        question="What is X?",
        answer="brief reply",
        retrieved_doc_ids=["doc_a"],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=0.9,
        token_overlap_score=0.10,
    )
    assert diag.mode == FailureMode.PARTIAL_ANSWER


def test_ok():
    diag = classify_failure(
        question="What is X?",
        answer="X is the thing in question.",
        retrieved_doc_ids=["doc_a"],
        relevant_doc_ids=["doc_a"],
        faithfulness_score=0.9,
        token_overlap_score=0.7,
    )
    assert diag.mode == FailureMode.OK


def test_no_ground_truth_relevant_docs():
    """When the dataset has no relevant docs (open question), only check faithfulness."""
    diag = classify_failure(
        question="Out of scope question?",
        answer="I cannot answer this from the provided context.",
        retrieved_doc_ids=["doc_a"],
        relevant_doc_ids=[],
        faithfulness_score=0.9,
        token_overlap_score=0.7,
    )
    # Refusal is fine when there are no relevant docs
    assert diag.mode == FailureMode.OK
