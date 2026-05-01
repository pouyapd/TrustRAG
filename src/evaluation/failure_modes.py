"""Failure mode classification for RAG outputs.

Inspired by interpretable failure analysis in safety-critical ML — every failure
gets a human-readable category, not just a metric drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureMode(StrEnum):
    """Discrete failure categories."""
    NO_RETRIEVAL = "no_retrieval"
    WRONG_RETRIEVAL = "wrong_retrieval"
    HALLUCINATION = "hallucination"
    PARTIAL_ANSWER = "partial_answer"
    REFUSAL_WHEN_ANSWERABLE = "refusal_when_answerable"
    OK = "ok"


@dataclass
class FailureDiagnosis:
    """Diagnosis attached to an evaluation row."""
    mode: FailureMode
    reason: str


# Phrases that indicate the model refused to answer.
_REFUSAL_MARKERS = (
    "cannot answer",
    "i don't know",
    "i do not know",
    "no information",
    "not contain",
    "not provided",
    "unable to answer",
)


def _is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in _REFUSAL_MARKERS)


def classify_failure(
    *,
    question: str,
    answer: str,
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    faithfulness_score: float | None,
    token_overlap_score: float,
    overlap_threshold: float = 0.30,
    faithfulness_threshold: float = 0.60,
) -> FailureDiagnosis:
    """Classify a single Q/A row into a failure mode.

    Decision tree (interpretable, in the spirit of decision-rule extraction):
    1. No retrieval at all -> NO_RETRIEVAL
    2. None of retrieved docs are relevant -> WRONG_RETRIEVAL
    3. Model refused but relevant docs were retrieved -> REFUSAL_WHEN_ANSWERABLE
    4. Faithfulness below threshold -> HALLUCINATION
    5. Token overlap with reference very low -> PARTIAL_ANSWER
    6. Otherwise -> OK
    """
    # 1. Retrieval missing
    if not retrieved_doc_ids:
        return FailureDiagnosis(FailureMode.NO_RETRIEVAL, "no documents retrieved")

    # 2. Retrieval wrong (only if we have ground-truth relevant docs)
    if relevant_doc_ids:
        relevant_set = set(relevant_doc_ids)
        retrieved_set = set(retrieved_doc_ids)
        if not (relevant_set & retrieved_set):
            return FailureDiagnosis(
                FailureMode.WRONG_RETRIEVAL,
                f"none of {len(retrieved_doc_ids)} retrieved docs match relevant set",
            )

    # 3. Refusal when answerable
    if _is_refusal(answer) and relevant_doc_ids:
        return FailureDiagnosis(
            FailureMode.REFUSAL_WHEN_ANSWERABLE,
            "model refused despite relevant context being retrieved",
        )

    # 4. Hallucination via faithfulness
    if faithfulness_score is not None and faithfulness_score < faithfulness_threshold:
        return FailureDiagnosis(
            FailureMode.HALLUCINATION,
            f"faithfulness={faithfulness_score:.2f} < {faithfulness_threshold}",
        )

    # 5. Partial answer via token overlap with reference
    if token_overlap_score < overlap_threshold:
        return FailureDiagnosis(
            FailureMode.PARTIAL_ANSWER,
            f"token_overlap={token_overlap_score:.2f} < {overlap_threshold}",
        )

    return FailureDiagnosis(FailureMode.OK, "passed all checks")
