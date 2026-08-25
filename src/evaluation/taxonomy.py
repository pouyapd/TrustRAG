"""Failure taxonomy v2 — interpretable, versioned, and auditable.

The v1 classifier in `failure_modes.py` is frozen and still runs on every row,
so historical results stay reproducible. This module is what new analysis
should use. It fixes three defects found in the audit of the v1 taxonomy:

1. **No abstention category.** A question the corpus cannot answer has an empty
   relevant-document set. v1 could not reach `hallucination` (that needs a low
   faithfulness score) or `refusal_when_answerable` (that needs a non-empty
   relevant set), so a system that confidently answered an unanswerable
   question was labelled `partial_answer`. v2 adds
   `answered_when_unanswerable` and `ok_abstained`.

2. **`partial_answer` absorbed wrong answers.** v1 dropped everything with low
   token overlap into one bucket, mixing "terse but right", "incomplete" and
   "flatly wrong". v2 separates `incorrect_answer` from `partial_answer` using
   key-fact recall (see `correctness.key_fact_recall`).

3. **Thresholds were undefended constants.** v2 puts every threshold in a
   versioned, hashable `TaxonomyConfig`, and every classification returns the
   feature vector and the id of the rule that fired, so a run can be re-scored
   under different thresholds with no LLM inference at all.

The classifier remains an ordered set of hand-written rules. It is deliberately
not a learned model: the point of the instrument is that a human can read why a
row was labelled the way it was.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from src.evaluation.correctness import (
    answer_precision_recall_f1,
    exact_match,
    is_refusal,
    key_fact_recall,
    key_facts,
)

TAXONOMY_VERSION = "v2.0"


class FailureModeV2(StrEnum):
    """Failure categories, v2."""

    # Non-failures
    OK = "ok"
    OK_ABSTAINED = "ok_abstained"
    # Retrieval-attributable
    NO_RETRIEVAL = "no_retrieval"
    WRONG_RETRIEVAL = "wrong_retrieval"
    # Generation-attributable
    HALLUCINATION = "hallucination"
    INCORRECT_ANSWER = "incorrect_answer"
    PARTIAL_ANSWER = "partial_answer"
    REFUSAL_WHEN_ANSWERABLE = "refusal_when_answerable"
    ANSWERED_WHEN_UNANSWERABLE = "answered_when_unanswerable"


#: Modes that are *not* failures. `ok_abstained` is a success: the system
#: correctly declined a question the corpus cannot answer.
NON_FAILURE_MODES: frozenset[FailureModeV2] = frozenset(
    {FailureModeV2.OK, FailureModeV2.OK_ABSTAINED}
)

#: Which pipeline stage each mode is attributed to. This is the seed of the
#: retrieval-vs-generation attribution analysis; it is a declared mapping, not
#: an inference, and it is reported as such.
STAGE_ATTRIBUTION: dict[FailureModeV2, str] = {
    FailureModeV2.OK: "none",
    FailureModeV2.OK_ABSTAINED: "none",
    FailureModeV2.NO_RETRIEVAL: "retrieval",
    FailureModeV2.WRONG_RETRIEVAL: "retrieval",
    FailureModeV2.HALLUCINATION: "generation",
    FailureModeV2.INCORRECT_ANSWER: "generation",
    FailureModeV2.PARTIAL_ANSWER: "generation",
    FailureModeV2.REFUSAL_WHEN_ANSWERABLE: "generation",
    FailureModeV2.ANSWERED_WHEN_UNANSWERABLE: "generation",
}


def is_failure(mode: FailureModeV2 | str) -> bool:
    """True when a mode counts against the failure rate."""
    return FailureModeV2(mode) not in NON_FAILURE_MODES


@dataclass(frozen=True)
class TaxonomyConfig:
    """Versioned thresholds for the v2 classifier.

    Every value here is a modelling choice, not a fact. `fingerprint()` hashes
    the whole configuration so a result can always be traced to the exact rules
    that produced it, and `scripts/reclassify.py` can re-score stored rows
    under a different configuration without touching an LLM.
    """

    version: str = TAXONOMY_VERSION
    #: Below this, an answer is treated as ungrounded in its context.
    faithfulness_threshold: float = 0.60
    #: At or above this normalized answer F1, a row is correct outright.
    answer_f1_ok: float = 0.60
    #: At or above this key-fact recall, a row is correct even if verbose.
    key_fact_recall_ok: float = 1.0
    #: At or below this key-fact recall, an answer reproduces so little of the
    #: reference that it is treated as wrong rather than merely incomplete.
    #: Not 0.0: a single incidental token match (a wrong answer that happens to
    #: repeat a common word such as "plan") would otherwise flip a clearly
    #: incorrect answer into `partial_answer`. This value was set by inspecting
    #: the bundled 20-question fixture and is therefore a tuned constant, not a
    #: validated one -- see the limitations section of docs/TAXONOMY.md.
    key_fact_recall_incorrect: float = 0.20
    #: Fallback used only when the reference answer has no extractable facts.
    fallback_f1_incorrect: float = 0.10

    def fingerprint(self) -> str:
        """Stable short hash of this configuration, for provenance."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DecisionFeatures:
    """Every signal the classifier looked at, recorded per row.

    Storing these is what makes the taxonomy auditable and re-runnable: given
    the features, a classification can be reproduced (or revised under new
    thresholds) without re-running retrieval or generation.
    """

    is_answerable: bool
    num_retrieved: int
    num_relevant_retrieved: int
    retrieval_hit: bool
    abstained: bool
    faithfulness: float | None
    answer_f1: float
    answer_precision: float
    answer_recall: float
    answer_exact_match: float
    key_fact_recall: float | None
    num_key_facts: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagnosisV2:
    """Result of classifying one row."""

    mode: FailureModeV2
    reason: str
    rule_id: str
    features: DecisionFeatures
    stage: str = field(default="")
    taxonomy_version: str = TAXONOMY_VERSION
    config_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.stage:
            self.stage = STAGE_ATTRIBUTION[self.mode]


def extract_features(
    *,
    answer: str,
    reference_answer: str,
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    faithfulness_score: float | None,
) -> DecisionFeatures:
    """Compute the decision features for one row. No LLM calls."""
    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    matched = relevant_set & retrieved_set

    precision, recall, f1 = answer_precision_recall_f1(answer, reference_answer)

    return DecisionFeatures(
        is_answerable=bool(relevant_doc_ids),
        num_retrieved=len(retrieved_doc_ids),
        num_relevant_retrieved=len(matched),
        retrieval_hit=bool(matched),
        abstained=is_refusal(answer),
        faithfulness=faithfulness_score,
        answer_f1=f1,
        answer_precision=precision,
        answer_recall=recall,
        answer_exact_match=exact_match(answer, reference_answer),
        key_fact_recall=key_fact_recall(answer, reference_answer),
        num_key_facts=len(key_facts(reference_answer)),
    )


def classify_features(
    features: DecisionFeatures,
    config: TaxonomyConfig | None = None,
) -> DiagnosisV2:
    """Apply the v2 rules to a precomputed feature vector.

    This is the re-runnable half of the classifier: it never touches a model,
    a vector store, or the network. Rules are evaluated in order and the first
    match wins; the id of that rule is recorded on the diagnosis.

    Rule order (see docs/TAXONOMY.md for the rationale of each):
      R1  nothing retrieved                        -> no_retrieval
      R2  unanswerable + abstained                 -> ok_abstained
      R3  unanswerable + answered                  -> answered_when_unanswerable
      R4  answerable + no relevant doc retrieved   -> wrong_retrieval
      R5  answerable + relevant retrieved + refused-> refusal_when_answerable
      R6  faithfulness below threshold             -> hallucination
      R8  all reference key facts present          -> ok
      R9  almost no reference key fact present     -> incorrect_answer
      R7  no key facts available, F1 high          -> ok
      R10 no key facts available, F1 near zero     -> incorrect_answer
      R11 otherwise                                -> partial_answer

    R8/R9 are evaluated before R7 because key facts are the stronger signal
    when the reference has any; F1 is only the fallback for references with no
    extractable facts.
    """
    cfg = config or TaxonomyConfig()

    def done(mode: FailureModeV2, rule: str, reason: str) -> DiagnosisV2:
        return DiagnosisV2(
            mode=mode,
            reason=reason,
            rule_id=rule,
            features=features,
            taxonomy_version=cfg.version,
            config_fingerprint=cfg.fingerprint(),
        )

    # R1 — retrieval returned nothing at all.
    if features.num_retrieved == 0:
        return done(FailureModeV2.NO_RETRIEVAL, "R1", "no documents retrieved")

    # R2/R3 — the corpus cannot answer this question. Abstaining is the correct
    # behaviour; answering anyway is the safety-critical failure v1 could not see.
    if not features.is_answerable:
        if features.abstained:
            return done(
                FailureModeV2.OK_ABSTAINED,
                "R2",
                "correctly abstained on an unanswerable question",
            )
        return done(
            FailureModeV2.ANSWERED_WHEN_UNANSWERABLE,
            "R3",
            "answered although no relevant document exists for this question",
        )

    # R4 — retrieval missed. Checked before refusal on purpose: when the context
    # holds no relevant document, refusing is the *correct* response, so the
    # causal fault belongs to retrieval rather than generation.
    if not features.retrieval_hit:
        return done(
            FailureModeV2.WRONG_RETRIEVAL,
            "R4",
            f"none of {features.num_retrieved} retrieved chunks came from a relevant document",
        )

    # R5 — refused despite having the evidence.
    if features.abstained:
        return done(
            FailureModeV2.REFUSAL_WHEN_ANSWERABLE,
            "R5",
            "refused although a relevant document was retrieved",
        )

    # R6 — answer not grounded in the context it was given.
    if (
        features.faithfulness is not None
        and features.faithfulness < cfg.faithfulness_threshold
    ):
        return done(
            FailureModeV2.HALLUCINATION,
            "R6",
            f"faithfulness={features.faithfulness:.2f} < {cfg.faithfulness_threshold}",
        )

    # R8/R9 — key-fact reasoning separates incomplete from wrong. Checked
    # before the F1 shortcut on purpose: a fluent answer that drops one of the
    # reference's facts can still score a high F1 on the words it shares, and
    # calling that "ok" is precisely the kind of miss this taxonomy exists to
    # catch. When the reference has facts, they decide the verdict.
    if features.num_key_facts > 0 and features.key_fact_recall is not None:
        if features.key_fact_recall >= cfg.key_fact_recall_ok:
            return done(
                FailureModeV2.OK,
                "R8",
                f"all {features.num_key_facts} reference key facts present "
                f"(key_fact_recall={features.key_fact_recall:.2f})",
            )
        if features.key_fact_recall <= cfg.key_fact_recall_incorrect:
            return done(
                FailureModeV2.INCORRECT_ANSWER,
                "R9",
                f"none of the {features.num_key_facts} reference key facts appear in the answer",
            )
        return done(
            FailureModeV2.PARTIAL_ANSWER,
            "R11",
            f"key_fact_recall={features.key_fact_recall:.2f} - some reference facts missing",
        )

    # R7 — no extractable facts in the reference; fall back to overall overlap.
    if features.answer_f1 >= cfg.answer_f1_ok:
        return done(
            FailureModeV2.OK,
            "R7",
            f"no reference key facts available and answer_f1={features.answer_f1:.2f} "
            f">= {cfg.answer_f1_ok}",
        )

    # R10 — no facts to check and almost nothing in common.
    if features.answer_f1 <= cfg.fallback_f1_incorrect:
        return done(
            FailureModeV2.INCORRECT_ANSWER,
            "R10",
            f"no reference key facts available and answer_f1={features.answer_f1:.2f} "
            f"<= {cfg.fallback_f1_incorrect}",
        )

    # R11 — on topic, incomplete.
    return done(
        FailureModeV2.PARTIAL_ANSWER,
        "R11",
        f"answer_f1={features.answer_f1:.2f} below {cfg.answer_f1_ok}",
    )


def classify_v2(
    *,
    answer: str,
    reference_answer: str,
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    faithfulness_score: float | None,
    config: TaxonomyConfig | None = None,
) -> DiagnosisV2:
    """Extract features and classify in one call."""
    features = extract_features(
        answer=answer,
        reference_answer=reference_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        faithfulness_score=faithfulness_score,
    )
    return classify_features(features, config)
