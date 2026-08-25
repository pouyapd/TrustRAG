"""End-to-end evaluation runner.

Reads a JSONL dataset of {question, answer, relevant_doc_ids}, runs each through
the RAG pipeline, computes metrics, classifies failures, and writes a report.

Dataset format (one JSON object per line):
    {"question": "...", "answer": "...", "relevant_doc_ids": ["doc_a", "doc_b"]}

The evaluation is in two independent phases:

1. **Inference** (`run_inference`) calls the pipeline and produces
   `InferenceRecord`s. This is the only phase that touches a model.
2. **Scoring** (`score_records`) turns records into `EvalRow`s: metrics,
   taxonomy v1, taxonomy v2, decision features. It is a pure function and can
   be re-run over stored records with different thresholds at zero inference
   cost — see `scripts/reclassify.py`.

`_run_rows` still exists with its original signature and still runs both
phases, so every existing caller (the CLI, the API and the CI regression
script) behaves exactly as before.

Backward compatibility: legacy metric fields and the v1 failure mode are
computed for every row and every legacy key in the summary keeps its original
name, position and value. New measures are added alongside, never in place of.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from src.evaluation.correctness import (
    abstention_rates,
    answer_precision_recall_f1,
    exact_match,
    key_fact_recall,
    key_facts,
)
from src.evaluation.failure_modes import FailureDiagnosis, classify_failure
from src.evaluation.metrics import (
    chunk_precision_at_k,
    chunk_recall_at_k,
    document_precision_at_k,
    document_recall_at_k,
    first_relevant_rank,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    token_overlap,
)
from src.evaluation.provenance import collect_provenance, describe_component
from src.evaluation.records import (
    RECORDS_FILENAME,
    InferenceRecord,
    RetrievedChunk,
    write_records,
)
from src.evaluation.statistics import (
    MIN_N_FOR_INFERENCE,
    bootstrap_mean_ci,
    mcnemar_exact,
    sample_size_warning,
    wilson_proportion_ci,
)
from src.evaluation.taxonomy import (
    STAGE_ATTRIBUTION,
    DiagnosisV2,
    TaxonomyConfig,
    classify_features,
    extract_features,
    is_failure,
)
from src.logging_setup import get_logger, setup_logging
from src.rag.pipeline import RAGPipeline


@dataclass
class EvalRow:
    """Per-question evaluation result.

    The first thirteen fields are the original schema and keep their original
    meaning. Everything below them is additive.
    """

    question: str
    reference_answer: str
    predicted_answer: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    token_overlap: float
    faithfulness: float | None
    latency_ms: float
    failure_mode: str
    failure_reason: str

    # ---- added: retrieval context ----
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    top_k: int = 0
    is_answerable: bool = True
    n_relevant_chunks: int | None = None

    # ---- added: corrected retrieval metrics (None when unanswerable) ----
    doc_recall_at_k: float | None = None
    doc_precision_at_k: float | None = None
    chunk_precision_at_k: float | None = None
    chunk_recall_at_k: float | None = None
    hit_rate_at_k: float | None = None
    first_relevant_rank: int | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None

    # ---- added: normalized answer correctness ----
    answer_exact_match: float = 0.0
    answer_f1_normalized: float = 0.0
    answer_precision_normalized: float = 0.0
    answer_recall_normalized: float = 0.0
    key_fact_recall: float | None = None
    num_key_facts: int = 0
    abstained: bool = False

    # ---- added: taxonomy v2 ----
    failure_mode_v2: str = ""
    failure_reason_v2: str = ""
    failure_rule_v2: str = ""
    failure_stage_v2: str = ""
    taxonomy_version: str = ""
    taxonomy_config_fingerprint: str = ""
    decision_features: dict = field(default_factory=dict)


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL evaluation dataset."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------
# Phase 1 — inference (the only phase that calls a model)
# ---------------------------------------------------------------

def _relevant_chunk_count(
    relevant_doc_ids: list[str],
    doc_chunk_counts: dict[str, int] | None,
) -> int | None:
    """Total chunks belonging to the relevant documents, if the store knows."""
    if not doc_chunk_counts or not relevant_doc_ids:
        return None
    total = sum(doc_chunk_counts.get(doc_id, 0) for doc_id in relevant_doc_ids)
    return total or None


def run_inference(
    dataset: list[dict],
    pipeline: RAGPipeline,
    top_k: int = 4,
    doc_chunk_counts: dict[str, int] | None = None,
) -> list[InferenceRecord]:
    """Run every question through the pipeline and capture the raw results.

    No metric and no failure label is computed here — that is deliberate, so
    the expensive half of evaluation never has to be repeated when the scoring
    rules change.
    """
    log = get_logger("evaluation.inference")
    records: list[InferenceRecord] = []
    corpus_chunks = sum(doc_chunk_counts.values()) if doc_chunk_counts else None

    for i, item in enumerate(dataset, start=1):
        question: str = item["question"]
        reference: str = item.get("answer", "")
        relevant_ids: list[str] = item.get("relevant_doc_ids", [])

        response = pipeline.query(question, top_k=top_k)

        records.append(
            InferenceRecord(
                index=i,
                question=question,
                reference_answer=reference,
                relevant_doc_ids=relevant_ids,
                predicted_answer=response.answer,
                retrieved=[
                    RetrievedChunk(
                        rank=rank,
                        chunk_id=s.chunk_id,
                        doc_id=s.doc_id,
                        source=s.source,
                        score=s.score,
                        text=s.text,
                    )
                    for rank, s in enumerate(response.sources, start=1)
                ],
                faithfulness=response.faithfulness_score,
                latency_ms=response.latency_ms,
                top_k=top_k,
                n_relevant_chunks=_relevant_chunk_count(relevant_ids, doc_chunk_counts),
                corpus_chunk_count=corpus_chunks,
            )
        )
        log.info("inference_row_done", index=i, num_retrieved=len(response.sources))
    return records


# ---------------------------------------------------------------
# Phase 2 — scoring and classification (pure, no model calls)
# ---------------------------------------------------------------

def score_record(
    record: InferenceRecord,
    taxonomy_config: TaxonomyConfig | None = None,
) -> EvalRow:
    """Turn one inference record into a scored row. Never calls a model."""
    cfg = taxonomy_config or TaxonomyConfig()
    top_k = record.top_k or len(record.retrieved)
    retrieved_doc_ids = record.retrieved_doc_ids
    relevant_ids = record.relevant_doc_ids
    reference = record.reference_answer
    answer = record.predicted_answer

    # --- legacy metrics, unchanged semantics ---
    legacy_precision = precision_at_k(retrieved_doc_ids, relevant_ids, top_k)
    legacy_recall = recall_at_k(retrieved_doc_ids, relevant_ids, top_k)
    legacy_mrr = mean_reciprocal_rank(retrieved_doc_ids, relevant_ids)
    legacy_overlap = token_overlap(answer, reference) if reference else 0.0

    diagnosis: FailureDiagnosis = classify_failure(
        question=record.question,
        answer=answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_ids,
        faithfulness_score=record.faithfulness,
        token_overlap_score=legacy_overlap,
    )

    # --- corrected answer correctness ---
    ans_precision, ans_recall, ans_f1 = answer_precision_recall_f1(answer, reference)

    # --- taxonomy v2 ---
    features = extract_features(
        answer=answer,
        reference_answer=reference,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_ids,
        faithfulness_score=record.faithfulness,
    )
    diagnosis_v2: DiagnosisV2 = classify_features(features, cfg)

    return EvalRow(
        question=record.question,
        reference_answer=reference,
        predicted_answer=answer,
        relevant_doc_ids=relevant_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        precision_at_k=legacy_precision,
        recall_at_k=legacy_recall,
        mrr=legacy_mrr,
        token_overlap=legacy_overlap,
        faithfulness=record.faithfulness,
        latency_ms=record.latency_ms,
        failure_mode=diagnosis.mode.value,
        failure_reason=diagnosis.reason,
        retrieved_chunk_ids=record.retrieved_chunk_ids,
        top_k=top_k,
        is_answerable=record.is_answerable,
        n_relevant_chunks=record.n_relevant_chunks,
        doc_recall_at_k=document_recall_at_k(retrieved_doc_ids, relevant_ids, top_k),
        doc_precision_at_k=document_precision_at_k(retrieved_doc_ids, relevant_ids, top_k),
        chunk_precision_at_k=chunk_precision_at_k(retrieved_doc_ids, relevant_ids, top_k),
        chunk_recall_at_k=chunk_recall_at_k(
            retrieved_doc_ids, relevant_ids, top_k, record.n_relevant_chunks
        ),
        hit_rate_at_k=hit_rate_at_k(retrieved_doc_ids, relevant_ids, top_k),
        first_relevant_rank=first_relevant_rank(retrieved_doc_ids, relevant_ids),
        reciprocal_rank=reciprocal_rank(retrieved_doc_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_doc_ids, relevant_ids, top_k, record.n_relevant_chunks),
        answer_exact_match=exact_match(answer, reference),
        answer_f1_normalized=ans_f1,
        answer_precision_normalized=ans_precision,
        answer_recall_normalized=ans_recall,
        key_fact_recall=key_fact_recall(answer, reference),
        num_key_facts=len(key_facts(reference)),
        abstained=features.abstained,
        failure_mode_v2=diagnosis_v2.mode.value,
        failure_reason_v2=diagnosis_v2.reason,
        failure_rule_v2=diagnosis_v2.rule_id,
        failure_stage_v2=diagnosis_v2.stage,
        taxonomy_version=diagnosis_v2.taxonomy_version,
        taxonomy_config_fingerprint=diagnosis_v2.config_fingerprint,
        decision_features=features.as_dict(),
    )


def score_records(
    records: list[InferenceRecord],
    taxonomy_config: TaxonomyConfig | None = None,
) -> list[EvalRow]:
    """Score a list of inference records. Pure — safe to re-run offline."""
    cfg = taxonomy_config or TaxonomyConfig()
    return [score_record(r, cfg) for r in records]


def run_evaluation_inline(
    dataset: list[dict],
    pipeline: RAGPipeline | None = None,
    top_k: int = 4,
    include_provenance: bool = False,
) -> tuple[dict, list[dict]]:
    """Run evaluation against an in-memory dataset (used by the API).

    Returns (summary_dict, list_of_row_dicts).

    Provenance is off by default here: the API path should not shell out to git
    on every request.
    """
    log = get_logger("evaluation.inline")
    pipeline = pipeline or RAGPipeline()
    rows = _run_rows(dataset, pipeline, top_k)
    report = aggregate(rows)
    if include_provenance:
        report["provenance"] = collect_provenance(**_pipeline_provenance(pipeline, top_k))
    log.info("inline_eval_done", count=len(rows))
    return report, [asdict(r) for r in rows]


def _run_rows(
    dataset: list[dict],
    pipeline: RAGPipeline,
    top_k: int,
    taxonomy_config: TaxonomyConfig | None = None,
) -> list[EvalRow]:
    """Shared row-evaluation loop used by both CLI and API runners.

    Signature preserved for backward compatibility; internally this is now
    inference followed by scoring.
    """
    log = get_logger("evaluation.runner")
    records = run_inference(
        dataset, pipeline, top_k, doc_chunk_counts=_safe_doc_chunk_counts(pipeline)
    )
    rows = score_records(records, taxonomy_config)
    for row in rows:
        log.info("eval_row_done", mode=row.failure_mode, mode_v2=row.failure_mode_v2)
    return rows


def _safe_doc_chunk_counts(pipeline: RAGPipeline) -> dict[str, int] | None:
    """Ask the vector store for per-document chunk counts, tolerating absence."""
    store = getattr(pipeline, "vector_store", None)
    getter = getattr(store, "doc_chunk_counts", None)
    if getter is None:
        return None
    try:
        return getter() or None
    except Exception:  # pragma: no cover - defensive
        return None


def _pipeline_provenance(pipeline: RAGPipeline, top_k: int) -> dict:
    """Identity of the components that produced a run."""
    store = getattr(pipeline, "vector_store", None)
    return {
        "pipeline": {
            "llm": describe_component(getattr(pipeline, "llm", None)),
            "embedder": describe_component(getattr(store, "embedder", None)),
            "vector_store": describe_component(store),
            "judge": describe_component(getattr(pipeline, "llm", None)),
            "judge_is_generator": True,
            "top_k": top_k,
        }
    }


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    top_k: int = 4,
    taxonomy_config: TaxonomyConfig | None = None,
) -> dict:
    """Run evaluation from a JSONL file and write report to disk."""
    setup_logging()
    log = get_logger("evaluation")

    cfg = taxonomy_config or TaxonomyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)
    log.info("dataset_loaded", count=len(dataset), path=str(dataset_path))

    pipeline = RAGPipeline()
    records = run_inference(
        dataset, pipeline, top_k, doc_chunk_counts=_safe_doc_chunk_counts(pipeline)
    )
    rows = score_records(records, cfg)

    report = aggregate(rows, taxonomy_config=cfg)
    report["provenance"] = collect_provenance(
        dataset={"path": str(dataset_path), "size": len(dataset)},
        taxonomy={"version": cfg.version, "fingerprint": cfg.fingerprint()},
        **_pipeline_provenance(pipeline, top_k),
    )
    write_outputs(rows, report, output_dir, records=records)
    return report


# ---------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------

def _mean_or_none(values: list[float | None]) -> float | None:
    """Mean over the non-None values, or None when there are none."""
    present = [v for v in values if v is not None]
    return round(mean(present), 3) if present else None


def aggregate(rows: list[EvalRow], taxonomy_config: TaxonomyConfig | None = None) -> dict:
    """Compute aggregate metrics over all rows.

    Legacy keys (`total` through `failure_modes`) keep their original names,
    order and values. Everything after them is additive.
    """
    if not rows:
        return {"total": 0}

    cfg = taxonomy_config or TaxonomyConfig()
    faithful = [r.faithfulness for r in rows if r.faithfulness is not None]
    failures = [r for r in rows if r.failure_mode != "ok"]
    mode_counts = Counter(r.failure_mode for r in rows)

    report: dict = {
        # ---- legacy block: frozen ----
        "total": len(rows),
        "precision_at_k_mean": round(mean(r.precision_at_k for r in rows), 3),
        "recall_at_k_mean": round(mean(r.recall_at_k for r in rows), 3),
        "mrr_mean": round(mean(r.mrr for r in rows), 3),
        "token_overlap_mean": round(mean(r.token_overlap for r in rows), 3),
        "faithfulness_mean": round(mean(faithful), 3) if faithful else None,
        "latency_ms_mean": round(mean(r.latency_ms for r in rows), 1),
        "failure_rate": round(len(failures) / len(rows), 3),
        "failure_modes": dict(mode_counts),
    }

    answerable = [r for r in rows if r.is_answerable]
    v2_counts = Counter(r.failure_mode_v2 for r in rows)
    v2_failures = [r for r in rows if is_failure(r.failure_mode_v2)]

    report["taxonomy_v2"] = {
        "version": cfg.version,
        "config_fingerprint": cfg.fingerprint(),
        "thresholds": cfg.as_dict(),
    }
    report["failure_modes_v2"] = dict(v2_counts)
    report["failure_rate_v2"] = round(len(v2_failures) / len(rows), 3)
    report["failure_rules_v2"] = dict(Counter(r.failure_rule_v2 for r in rows))

    stage_counts = Counter(r.failure_stage_v2 for r in v2_failures)
    report["attribution"] = {
        "retrieval": stage_counts.get("retrieval", 0),
        "generation": stage_counts.get("generation", 0),
        "total_failures": len(v2_failures),
        "retrieval_share": (
            round(stage_counts.get("retrieval", 0) / len(v2_failures), 3)
            if v2_failures
            else None
        ),
        "note": (
            "Stage attribution is a declared mapping from failure mode to pipeline "
            "stage (see taxonomy.STAGE_ATTRIBUTION), not an inferred causal claim. "
            "A controlled oracle-context ablation is required for causal attribution."
        ),
    }

    report["retrieval_corrected"] = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(rows) - len(answerable),
        "doc_recall_at_k_mean": _mean_or_none([r.doc_recall_at_k for r in rows]),
        "doc_precision_at_k_mean": _mean_or_none([r.doc_precision_at_k for r in rows]),
        "chunk_precision_at_k_mean": _mean_or_none([r.chunk_precision_at_k for r in rows]),
        "chunk_recall_at_k_mean": _mean_or_none([r.chunk_recall_at_k for r in rows]),
        "hit_rate_at_k_mean": _mean_or_none([r.hit_rate_at_k for r in rows]),
        "ndcg_at_k_mean": _mean_or_none([r.ndcg_at_k for r in rows]),
        "reciprocal_rank_mean": _mean_or_none([r.reciprocal_rank for r in rows]),
        "first_relevant_rank_mean": _mean_or_none(
            [float(r.first_relevant_rank) if r.first_relevant_rank else None for r in rows]
        ),
        "note": (
            "Computed over answerable questions only; unanswerable questions have no "
            "defined retrieval target and are excluded rather than scored as 0."
        ),
    }

    report["answer_corrected"] = {
        "exact_match_mean": round(mean(r.answer_exact_match for r in rows), 3),
        "f1_normalized_mean": round(mean(r.answer_f1_normalized for r in rows), 3),
        "key_fact_recall_mean": _mean_or_none([r.key_fact_recall for r in rows]),
    }

    report["abstention"] = abstention_rates(
        answerable=[r.is_answerable for r in rows],
        abstained=[r.abstained for r in rows],
    )

    report["confidence_intervals"] = _confidence_intervals(rows, failures, v2_failures)
    report["taxonomy_comparison"] = _taxonomy_comparison(rows)
    report["statistical_notes"] = _statistical_notes(rows)
    return report


def _confidence_intervals(
    rows: list[EvalRow],
    failures: list[EvalRow],
    v2_failures: list[EvalRow],
) -> dict:
    """Interval estimates for the headline numbers."""
    n = len(rows)
    intervals = {
        "failure_rate": wilson_proportion_ci(len(failures), n).as_dict(),
        "failure_rate_v2": wilson_proportion_ci(len(v2_failures), n).as_dict(),
        "token_overlap_mean": bootstrap_mean_ci([r.token_overlap for r in rows]).as_dict(),
        "answer_f1_normalized_mean": bootstrap_mean_ci(
            [r.answer_f1_normalized for r in rows]
        ).as_dict(),
    }

    # Per-mode shares of the v2 distribution. A failure-mode breakdown is a set
    # of proportions estimated from the same small sample, so each count needs
    # its own interval; reporting "5 incorrect answers" as if it were exact is
    # how a 20-question run turns into an overconfident claim.
    v2_counts = Counter(r.failure_mode_v2 for r in rows)
    intervals["failure_mode_shares_v2"] = {
        mode: wilson_proportion_ci(count, n).as_dict() for mode, count in sorted(v2_counts.items())
    }

    hit_rates = [r.hit_rate_at_k for r in rows if r.hit_rate_at_k is not None]
    if hit_rates:
        intervals["hit_rate_at_k_mean"] = wilson_proportion_ci(
            int(sum(hit_rates)), len(hit_rates)
        ).as_dict()

    faithful = [r.faithfulness for r in rows if r.faithfulness is not None]
    if faithful:
        intervals["faithfulness_mean"] = bootstrap_mean_ci(faithful).as_dict()
    return intervals


def _taxonomy_comparison(rows: list[EvalRow]) -> dict:
    """Paired comparison of the v1 and v2 taxonomies over the same rows.

    Both taxonomies label the same questions, so the correct comparison is
    paired. McNemar tests whether they disagree about *whether* a row failed;
    the crosstab shows how v1's labels were redistributed by v2, which is where
    the substantive difference usually lives.
    """
    v1_fail = [r.failure_mode != "ok" for r in rows]
    v2_fail = [is_failure(r.failure_mode_v2) for r in rows]

    only_v1 = sum(1 for a, b in zip(v1_fail, v2_fail, strict=True) if a and not b)
    only_v2 = sum(1 for a, b in zip(v1_fail, v2_fail, strict=True) if b and not a)

    crosstab: dict[str, dict[str, int]] = {}
    for row in rows:
        crosstab.setdefault(row.failure_mode, {})
        crosstab[row.failure_mode][row.failure_mode_v2] = (
            crosstab[row.failure_mode].get(row.failure_mode_v2, 0) + 1
        )

    return {
        "mcnemar_failure_agreement": mcnemar_exact(only_v1, only_v2).as_dict(),
        "v1_to_v2_crosstab": crosstab,
        "note": (
            "The crosstab is descriptive. Agreement on whether a row failed says "
            "nothing about agreement on why it failed, which is the distinction the "
            "v2 taxonomy exists to make."
        ),
    }


def _statistical_notes(rows: list[EvalRow]) -> list[str]:
    """Explicit warnings about what this sample can and cannot support."""
    notes: list[str] = []
    warning = sample_size_warning(len(rows), context="evaluation dataset")
    if warning:
        notes.append(warning)

    n_unanswerable = sum(1 for r in rows if not r.is_answerable)
    if 0 < n_unanswerable < 10:
        notes.append(
            f"Only {n_unanswerable} unanswerable question(s): abstention rates are "
            "point estimates on a handful of items and are not generalisable."
        )
    if n_unanswerable == 0:
        notes.append(
            "No unanswerable questions in this dataset: failure-to-abstain cannot be "
            "measured at all."
        )

    faithful = {r.faithfulness for r in rows if r.faithfulness is not None}
    if len(faithful) == 1:
        notes.append(
            f"Faithfulness is constant at {next(iter(faithful))} across all rows — it has "
            "zero variance and is not discriminating between systems. With an extractive "
            "generator scored by the same model, this is expected by construction."
        )

    n_docs = len({d for r in rows for d in r.retrieved_doc_ids})
    max_k = max((r.top_k for r in rows), default=0)
    if n_docs and max_k >= n_docs:
        notes.append(
            f"top_k={max_k} is greater than or equal to the {n_docs} distinct documents "
            "ever retrieved: retrieval can return the entire corpus, so retrieval metrics "
            "are saturated and cannot discriminate between retrievers."
        )
    notes.append(
        f"Inference is deterministic in the offline configuration, so repeated runs have "
        f"zero variance. Intervals here describe sampling uncertainty over questions "
        f"(n={len(rows)}), not run-to-run variability."
    )
    return notes


# ---------------------------------------------------------------
# Output
# ---------------------------------------------------------------

def write_outputs(
    rows: list[EvalRow],
    report: dict,
    output_dir: Path,
    records: list[InferenceRecord] | None = None,
) -> None:
    """Write JSONL rows + JSON summary + Markdown report.

    When inference records are supplied they are written to `inference.jsonl`,
    which is what makes re-scoring without a model possible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(rows, report), encoding="utf-8")

    if records is not None:
        write_records(records, output_dir / RECORDS_FILENAME)


def _fmt_interval(interval: dict | None) -> str:
    """Render a confidence interval, flagging when it is underpowered."""
    if not interval or interval.get("lower") is None:
        return ""
    flag = "" if interval.get("sufficient") else " *"
    return f" [{interval['lower']:.3f}, {interval['upper']:.3f}]{flag}"


def _render_markdown(rows: list[EvalRow], report: dict) -> str:
    """Render a human-readable Markdown report."""
    ci = report.get("confidence_intervals", {})
    lines = [
        "# TrustRAG Evaluation Report",
        "",
        "## Aggregate Metrics",
        "",
        f"- **Total queries:** {report['total']}",
        f"- **Precision@k (mean):** {report['precision_at_k_mean']}",
        f"- **Recall@k (mean):** {report['recall_at_k_mean']}",
        f"- **MRR (mean):** {report['mrr_mean']}",
        f"- **Token overlap (mean):** {report['token_overlap_mean']}"
        f"{_fmt_interval(ci.get('token_overlap_mean'))}",
        f"- **Faithfulness (mean):** {report['faithfulness_mean']}"
        f"{_fmt_interval(ci.get('faithfulness_mean'))}",
        f"- **Latency ms (mean):** {report['latency_ms_mean']}",
        f"- **Failure rate:** {report['failure_rate']}"
        f"{_fmt_interval(ci.get('failure_rate'))}",
        "",
        "Legacy metrics above are retained unchanged for reproducibility. "
        "Intervals marked `*` are below the sample size at which they should be "
        "treated as evidence.",
        "",
        "## Failure Mode Breakdown (v1, legacy)",
        "",
    ]
    for mode, count in report["failure_modes"].items():
        lines.append(f"- `{mode}`: {count}")

    if "failure_modes_v2" in report:
        lines += [
            "",
            f"## Failure Mode Breakdown (taxonomy {report['taxonomy_v2']['version']})",
            "",
            f"- **Failure rate (v2):** {report['failure_rate_v2']}"
            f"{_fmt_interval(ci.get('failure_rate_v2'))}",
            "",
        ]
        for mode, count in sorted(report["failure_modes_v2"].items()):
            lines.append(f"- `{mode}`: {count}")

        attribution = report.get("attribution", {})
        lines += [
            "",
            "### Stage attribution",
            "",
            f"- retrieval-attributable failures: {attribution.get('retrieval', 0)}",
            f"- generation-attributable failures: {attribution.get('generation', 0)}",
            "",
            f"> {attribution.get('note', '')}",
        ]

    corrected = report.get("retrieval_corrected", {})
    if corrected:
        lines += [
            "",
            "## Corrected Retrieval Metrics",
            "",
            f"- answerable questions: {corrected.get('n_answerable')} "
            f"(unanswerable excluded: {corrected.get('n_unanswerable')})",
            f"- Document Recall@k: {corrected.get('doc_recall_at_k_mean')}",
            f"- Document Precision@k: {corrected.get('doc_precision_at_k_mean')}",
            f"- Chunk Precision@k: {corrected.get('chunk_precision_at_k_mean')}",
            f"- Chunk Recall@k: {corrected.get('chunk_recall_at_k_mean')}",
            f"- Hit rate@k: {corrected.get('hit_rate_at_k_mean')}"
            f"{_fmt_interval(ci.get('hit_rate_at_k_mean'))}",
            f"- nDCG@k: {corrected.get('ndcg_at_k_mean')}",
            f"- Mean reciprocal rank: {corrected.get('reciprocal_rank_mean')}",
        ]

    answer = report.get("answer_corrected", {})
    if answer:
        lines += [
            "",
            "## Answer Correctness (normalized)",
            "",
            f"- Exact match: {answer.get('exact_match_mean')}",
            f"- Token F1: {answer.get('f1_normalized_mean')}"
            f"{_fmt_interval(ci.get('answer_f1_normalized_mean'))}",
            f"- Key-fact recall: {answer.get('key_fact_recall_mean')}",
        ]

    abstention = report.get("abstention", {})
    if abstention:
        lines += [
            "",
            "## Abstention",
            "",
            f"- answerable / unanswerable: {abstention.get('n_answerable')} / "
            f"{abstention.get('n_unanswerable')}",
            f"- false-answer rate (failed to abstain): {abstention.get('false_answer_rate')}",
            f"- false-refusal rate: {abstention.get('false_refusal_rate')}",
            f"- abstention accuracy: {abstention.get('abstention_accuracy')}",
        ]

    notes = report.get("statistical_notes", [])
    if notes:
        lines += ["", "## Statistical Notes", ""]
        lines += [f"- {note}" for note in notes]

    lines += ["", "## Failure Cases", ""]
    for r in rows:
        if r.failure_mode == "ok" and not is_failure(r.failure_mode_v2 or "ok"):
            continue
        lines += [
            f"### {r.failure_mode_v2 or r.failure_mode}",
            f"- **Question:** {r.question}",
            f"- **Reference:** {r.reference_answer[:200]}",
            f"- **Predicted:** {r.predicted_answer[:200]}",
            f"- **v1:** `{r.failure_mode}` — {r.failure_reason}",
            f"- **v2:** `{r.failure_mode_v2}` (rule {r.failure_rule_v2}) — {r.failure_reason_v2}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrustRAG evaluation")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=TaxonomyConfig.faithfulness_threshold,
        help="v2 taxonomy: below this, an answer counts as ungrounded",
    )
    parser.add_argument(
        "--answer-f1-ok",
        type=float,
        default=TaxonomyConfig.answer_f1_ok,
        help="v2 taxonomy: at or above this normalized F1, an answer counts as correct",
    )
    args = parser.parse_args()

    cfg = TaxonomyConfig(
        faithfulness_threshold=args.faithfulness_threshold,
        answer_f1_ok=args.answer_f1_ok,
    )
    report = run_evaluation(Path(args.dataset), Path(args.out), top_k=args.top_k, taxonomy_config=cfg)
    print(json.dumps(report, indent=2))


__all__ = [
    "MIN_N_FOR_INFERENCE",
    "STAGE_ATTRIBUTION",
    "EvalRow",
    "aggregate",
    "load_dataset",
    "main",
    "run_evaluation",
    "run_evaluation_inline",
    "run_inference",
    "score_record",
    "score_records",
    "write_outputs",
]


if __name__ == "__main__":
    main()
