"""Re-score a completed evaluation run without calling any model.

This is what decoupling inference from classification buys. A finished run
writes `inference.jsonl` alongside its report; this script reads those records
back, applies a (possibly different) `TaxonomyConfig`, and writes a fresh set
of outputs. No LLM, no embedder, no vector store, no network.

That makes threshold sensitivity analysis essentially free: the expensive half
of evaluation is already paid for and stored.

Examples
--------
Re-score the last run with the default thresholds into a new directory:

    python scripts/reclassify.py --records reports/inference.jsonl --out reports/recheck

Ask what happens if a stricter answer threshold is used:

    python scripts/reclassify.py --records reports/inference.jsonl \
        --out reports/strict --answer-f1-ok 0.8

Compare several faithfulness thresholds in one pass:

    python scripts/reclassify.py --records reports/inference.jsonl \
        --out reports/sweep --sweep-faithfulness 0.3,0.5,0.6,0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance
from src.evaluation.records import read_records
from src.evaluation.runner import aggregate, score_records, write_outputs
from src.evaluation.taxonomy import TaxonomyConfig
from src.logging_setup import get_logger, setup_logging


def _build_config(args: argparse.Namespace, faithfulness: float | None = None) -> TaxonomyConfig:
    return TaxonomyConfig(
        faithfulness_threshold=(
            faithfulness if faithfulness is not None else args.faithfulness_threshold
        ),
        answer_f1_ok=args.answer_f1_ok,
        key_fact_recall_ok=args.key_fact_recall_ok,
        key_fact_recall_incorrect=args.key_fact_recall_incorrect,
        fallback_f1_incorrect=args.fallback_f1_incorrect,
    )


def _reclassify_one(records, config: TaxonomyConfig, out_dir: Path, source: Path) -> dict:
    """Score records under one configuration and write the outputs."""
    rows = score_records(records, config)
    report = aggregate(rows, taxonomy_config=config)
    report["provenance"] = collect_provenance(
        reclassified_from=str(source),
        taxonomy={"version": config.version, "fingerprint": config.fingerprint()},
        inference_reused=True,
        note="Scored from stored inference records. No model was called.",
    )
    write_outputs(rows, report, out_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-classify a stored evaluation run without model inference"
    )
    parser.add_argument(
        "--records",
        default="reports/inference.jsonl",
        help="Path to inference.jsonl produced by a previous run",
    )
    parser.add_argument("--out", default="reports/reclassified", help="Output directory")
    parser.add_argument("--faithfulness-threshold", type=float,
                        default=TaxonomyConfig.faithfulness_threshold)
    parser.add_argument("--answer-f1-ok", type=float, default=TaxonomyConfig.answer_f1_ok)
    parser.add_argument("--key-fact-recall-ok", type=float,
                        default=TaxonomyConfig.key_fact_recall_ok)
    parser.add_argument("--key-fact-recall-incorrect", type=float,
                        default=TaxonomyConfig.key_fact_recall_incorrect)
    parser.add_argument("--fallback-f1-incorrect", type=float,
                        default=TaxonomyConfig.fallback_f1_incorrect)
    parser.add_argument(
        "--sweep-faithfulness",
        default="",
        help="Comma-separated faithfulness thresholds; writes one sub-report per value",
    )
    args = parser.parse_args()

    setup_logging()
    log = get_logger("reclassify")

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"No inference records at {records_path}.", file=sys.stderr)
        print(
            "Run an evaluation first (python scripts/run_offline_eval.py) "
            "to produce inference.jsonl.",
            file=sys.stderr,
        )
        return 1

    records = read_records(records_path)
    log.info("records_loaded", count=len(records), path=str(records_path))

    out_dir = Path(args.out)

    if args.sweep_faithfulness:
        values = [float(v) for v in args.sweep_faithfulness.split(",") if v.strip()]
        summary: dict[str, dict] = {}
        for value in values:
            config = _build_config(args, faithfulness=value)
            report = _reclassify_one(records, config, out_dir / f"faithfulness_{value}", records_path)
            summary[str(value)] = {
                "failure_rate_v2": report["failure_rate_v2"],
                "failure_modes_v2": report["failure_modes_v2"],
                "config_fingerprint": config.fingerprint(),
            }
        (out_dir / "sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        print(f"\nWrote {len(values)} re-classifications to {out_dir} - no model calls.")
        return 0

    config = _build_config(args)
    report = _reclassify_one(records, config, out_dir, records_path)

    print(json.dumps(
        {
            "total": report["total"],
            "failure_rate": report["failure_rate"],
            "failure_rate_v2": report["failure_rate_v2"],
            "failure_modes_v2": report["failure_modes_v2"],
            "taxonomy_config_fingerprint": config.fingerprint(),
        },
        indent=2,
    ))
    print(f"\nRe-classified {report['total']} rows from stored records - no model calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
