"""Offline evaluation runner — uses MockExtractiveLLM so it runs in CI without API keys.

This is what 'evaluation regression' looks like: every PR runs this and fails
if the failure rate goes above the threshold. It's the same idea as a unit test,
but for end-to-end RAG quality.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance
from src.evaluation.runner import (
    aggregate,
    run_inference,
    score_records,
    write_outputs,
)
from src.evaluation.taxonomy import TaxonomyConfig
from src.logging_setup import get_logger, setup_logging
from src.rag.chunking import DocumentChunker, load_directory
from src.rag.mock_llm import MockExtractiveLLM
from src.rag.pipeline import RAGPipeline
from src.rag.providers import HashEmbeddings
from src.rag.vector_store import VectorStore

# Quality thresholds — failing these breaks CI
MAX_FAILURE_RATE = 0.50      # at most 50% failures with the mock LLM
MIN_RECALL_AT_K = 0.40       # retrieval should find relevant docs at least 40% of the time
MAX_FAILURE_RATE_V2 = 0.60   # v2 separates wrong from incomplete answers; own ceiling


def main() -> int:
    setup_logging()
    log = get_logger("offline_eval")

    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "data" / "documents"
    dataset_path = repo_root / "data" / "eval" / "qa_test.jsonl"
    output_dir = repo_root / "reports"

    # Use a tempdir for the vector store so CI is reproducible
    with tempfile.TemporaryDirectory(prefix="offline_eval_") as tmpdir:
        import os
        os.environ["CHROMA_PERSIST_DIR"] = tmpdir
        from src import config
        config.settings = config.Settings()

        store = VectorStore(HashEmbeddings())
        store.reset()

        chunker = DocumentChunker(chunk_size=256, chunk_overlap=20)
        for text, doc_id, source in load_directory(docs_dir):
            store.add(chunker.chunk_text(text, doc_id=doc_id, source=source))

        log.info("docs_indexed", count=store.count())

        pipeline = RAGPipeline(vector_store=store, llm=MockExtractiveLLM())

        with dataset_path.open(encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        taxonomy_config = TaxonomyConfig()

        # Inference and scoring are separate phases: the records written below
        # let scripts/reclassify.py re-score this run with different thresholds
        # without calling the model again.
        records = run_inference(
            dataset,
            pipeline,
            top_k=4,
            doc_chunk_counts=store.doc_chunk_counts(),
        )
        rows = score_records(records, taxonomy_config)
        report = aggregate(rows, taxonomy_config=taxonomy_config)
        report["provenance"] = collect_provenance(
            dataset={"path": str(dataset_path), "size": len(dataset)},
            taxonomy={
                "version": taxonomy_config.version,
                "fingerprint": taxonomy_config.fingerprint(),
            },
            pipeline={
                "llm": "MockExtractiveLLM",
                "embedder": "HashEmbeddings",
                "judge": "MockExtractiveLLM",
                "judge_is_generator": True,
                "top_k": 4,
                "chunk_size": 256,
                "chunk_overlap": 20,
            },
            mode="offline_deterministic",
        )
        write_outputs(rows, report, output_dir, records=records)

    print("\n=== Offline Evaluation Report ===")
    print(json.dumps(report, indent=2))

    # Regression checks
    failures: list[str] = []
    if report["failure_rate"] > MAX_FAILURE_RATE:
        failures.append(
            f"failure_rate {report['failure_rate']} > {MAX_FAILURE_RATE}"
        )
    if report["recall_at_k_mean"] < MIN_RECALL_AT_K:
        failures.append(
            f"recall_at_k_mean {report['recall_at_k_mean']} < {MIN_RECALL_AT_K}"
        )
    if report["failure_rate_v2"] > MAX_FAILURE_RATE_V2:
        failures.append(
            f"failure_rate_v2 {report['failure_rate_v2']} > {MAX_FAILURE_RATE_V2}"
        )
    if not (output_dir / "inference.jsonl").exists():
        failures.append("inference.jsonl was not written — reclassification would be impossible")

    if failures:
        print("\nREGRESSION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll regression checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
