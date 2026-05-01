"""End-to-end evaluation runner.

Reads a JSONL dataset of {question, answer, relevant_doc_ids}, runs each through
the RAG pipeline, computes metrics, classifies failures, and writes a report.

Dataset format (one JSON object per line):
    {"question": "...", "answer": "...", "relevant_doc_ids": ["doc_a", "doc_b"]}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from src.evaluation.failure_modes import FailureDiagnosis, classify_failure
from src.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    token_overlap,
)
from src.logging_setup import get_logger, setup_logging
from src.rag.pipeline import RAGPipeline


@dataclass
class EvalRow:
    """Per-question evaluation result."""
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


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL evaluation dataset."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_evaluation_inline(
    dataset: list[dict],
    pipeline: RAGPipeline | None = None,
    top_k: int = 4,
) -> tuple[dict, list[dict]]:
    """Run evaluation against an in-memory dataset (used by the API).

    Returns (summary_dict, list_of_row_dicts).
    """
    log = get_logger("evaluation.inline")
    pipeline = pipeline or RAGPipeline()
    rows = _run_rows(dataset, pipeline, top_k)
    log.info("inline_eval_done", count=len(rows))
    return aggregate(rows), [asdict(r) for r in rows]


def _run_rows(dataset: list[dict], pipeline: RAGPipeline, top_k: int) -> list[EvalRow]:
    """Shared row-evaluation loop used by both CLI and API runners."""
    log = get_logger("evaluation.runner")
    rows: list[EvalRow] = []

    for i, item in enumerate(dataset, start=1):
        question: str = item["question"]
        reference: str = item.get("answer", "")
        relevant_ids: list[str] = item.get("relevant_doc_ids", [])

        response = pipeline.query(question, top_k=top_k)
        retrieved_doc_ids = [s.doc_id for s in response.sources]

        p_at_k = precision_at_k(retrieved_doc_ids, relevant_ids, top_k)
        r_at_k = recall_at_k(retrieved_doc_ids, relevant_ids, top_k)
        mrr = mean_reciprocal_rank(retrieved_doc_ids, relevant_ids)
        overlap = token_overlap(response.answer, reference) if reference else 0.0

        diagnosis: FailureDiagnosis = classify_failure(
            question=question,
            answer=response.answer,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_ids,
            faithfulness_score=response.faithfulness_score,
            token_overlap_score=overlap,
        )

        rows.append(
            EvalRow(
                question=question,
                reference_answer=reference,
                predicted_answer=response.answer,
                relevant_doc_ids=relevant_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                precision_at_k=p_at_k,
                recall_at_k=r_at_k,
                mrr=mrr,
                token_overlap=overlap,
                faithfulness=response.faithfulness_score,
                latency_ms=response.latency_ms,
                failure_mode=diagnosis.mode.value,
                failure_reason=diagnosis.reason,
            )
        )
        log.info("eval_row_done", index=i, mode=diagnosis.mode.value)
    return rows


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    top_k: int = 4,
) -> dict:
    """Run evaluation from a JSONL file and write report to disk."""
    setup_logging()
    log = get_logger("evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)
    log.info("dataset_loaded", count=len(dataset), path=str(dataset_path))

    pipeline = RAGPipeline()
    rows = _run_rows(dataset, pipeline, top_k)

    report = aggregate(rows)
    write_outputs(rows, report, output_dir)
    return report


def aggregate(rows: list[EvalRow]) -> dict:
    """Compute aggregate metrics over all rows."""
    if not rows:
        return {"total": 0}

    faithful = [r.faithfulness for r in rows if r.faithfulness is not None]
    failures = [r for r in rows if r.failure_mode != "ok"]
    mode_counts = Counter(r.failure_mode for r in rows)

    return {
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


def write_outputs(rows: list[EvalRow], report: dict, output_dir: Path) -> None:
    """Write JSONL rows + JSON summary + Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(rows, report), encoding="utf-8")


def _render_markdown(rows: list[EvalRow], report: dict) -> str:
    """Render a human-readable Markdown report."""
    lines = [
        "# TrustRAG Evaluation Report",
        "",
        "## Aggregate Metrics",
        "",
        f"- **Total queries:** {report['total']}",
        f"- **Precision@k (mean):** {report['precision_at_k_mean']}",
        f"- **Recall@k (mean):** {report['recall_at_k_mean']}",
        f"- **MRR (mean):** {report['mrr_mean']}",
        f"- **Token overlap (mean):** {report['token_overlap_mean']}",
        f"- **Faithfulness (mean):** {report['faithfulness_mean']}",
        f"- **Latency ms (mean):** {report['latency_ms_mean']}",
        f"- **Failure rate:** {report['failure_rate']}",
        "",
        "## Failure Mode Breakdown",
        "",
    ]
    for mode, count in report["failure_modes"].items():
        lines.append(f"- `{mode}`: {count}")

    lines += ["", "## Failure Cases", ""]
    for r in rows:
        if r.failure_mode == "ok":
            continue
        lines += [
            f"### {r.failure_mode}",
            f"- **Question:** {r.question}",
            f"- **Predicted:** {r.predicted_answer[:200]}",
            f"- **Reason:** {r.failure_reason}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrustRAG evaluation")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    report = run_evaluation(Path(args.dataset), Path(args.out), top_k=args.top_k)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
