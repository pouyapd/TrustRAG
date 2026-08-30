"""Run one evidence-aware evaluation experiment end to end.

    raw dataset -> normalised schema -> corpus -> chunking -> vector store
    -> retrieval -> generation -> inference records -> evidence-aware scoring
    -> failure taxonomy -> statistics -> report

Every run writes a report carrying enough provenance to reproduce it: dataset
file and checksum, split, sample size, chunking and retrieval configuration,
embedder and generator identity, taxonomy version and threshold fingerprint,
git commit and package versions.

Example
-------
    python scripts/run_experiment.py --dataset qasper \
        --raw data/raw/qasper-dev-v0.3.json --split dev \
        --limit 200 --embedder minilm --out reports/experiments/qasper_dev
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import build_corpus
from src.data.loaders import get_loader
from src.data.schema import index_documents, validate_dataset
from src.evaluation.provenance import collect_provenance
from src.evaluation.runner import aggregate, run_inference, score_records, write_outputs
from src.evaluation.taxonomy import TaxonomyConfig
from src.logging_setup import get_logger, setup_logging
from src.rag.chunking import DocumentChunker
from src.rag.mock_llm import MockExtractiveLLM
from src.rag.pipeline import RAGPipeline
from src.rag.providers import HashEmbeddings, LocalEmbeddings
from src.rag.vector_store import VectorStore


def file_checksum(path: Path) -> str:
    """SHA-256 of a raw dataset file, so a run names the exact input it used."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_embedder(name: str):
    """Embedder by explicit name rather than by fallback chain.

    The production `get_embeddings()` resolves through a fallback chain, which
    is right for a service and wrong for an experiment: a run must record which
    embedder it actually used, not discover it.
    """
    if name == "minilm":
        return LocalEmbeddings("all-MiniLM-L6-v2"), "sentence-transformers/all-MiniLM-L6-v2"
    if name == "hash":
        return HashEmbeddings(), "HashEmbeddings(deterministic-bag-of-words)"
    raise ValueError(f"unknown embedder {name!r}; expected 'minilm' or 'hash'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an evidence-aware RAG experiment")
    parser.add_argument("--dataset", required=True, help="registered loader name, e.g. qasper, nq")
    parser.add_argument("--raw", required=True, help="path to the raw dataset file")
    parser.add_argument("--split", default="test", help="split label recorded on every question")
    parser.add_argument("--limit", type=int, default=200, help="number of questions to load")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--embedder", default="minilm", choices=["minilm", "hash"])
    parser.add_argument("--out", required=True, help="output directory for the report")
    parser.add_argument(
        "--tag", default="", help="short label distinguishing this run in comparisons"
    )
    args = parser.parse_args()

    setup_logging()
    log = get_logger("experiment")

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"raw dataset not found: {raw_path}", file=sys.stderr)
        return 1

    # ---- 1. dataset ----
    loader = get_loader(args.dataset)
    result = loader.load(raw_path, limit=args.limit, split=args.split)
    documents = result.documents
    questions = result.questions
    if not questions:
        print("loader produced no questions", file=sys.stderr)
        return 1

    doc_index = index_documents(documents)
    problems = validate_dataset(questions, doc_index)
    if problems:
        # A span that does not resolve means evidence alignment would be wrong.
        print(f"dataset validation failed for {len(problems)} records", file=sys.stderr)
        for qid, issues in list(problems.items())[:5]:
            print(f"  {qid}: {issues}", file=sys.stderr)
        return 1
    log.info("dataset_loaded", questions=len(questions), documents=len(documents))

    # ---- 2. corpus + index ----
    embedder, embedder_name = build_embedder(args.embedder)
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # A private directory and collection per run: two experiments must never
    # share a collection, and the store is told explicitly rather than through
    # a global that was already bound at import time.
    #
    # The index lives under data/build/ (git-ignored) rather than in a
    # TemporaryDirectory. On Windows, ChromaDB still holds its SQLite and HNSW
    # files open when the context manager tries to remove them, so automatic
    # cleanup raised PermissionError and killed the run before it could score
    # anything. Keeping the index also makes a run auditable after the fact.
    index_dir = Path("data/build") / f"index_{args.tag or args.dataset}"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(
        embedder,
        persist_dir=str(index_dir),
        collection_name=f"exp_{args.dataset}_{args.split}",
    )
    corpus_stats = build_corpus(documents, store, chunker, reset=True)
    log.info("corpus_built", **{k: v for k, v in corpus_stats.as_dict().items()
                               if k != "chunks_per_document"})

    # ---- 3. inference ----
    pipeline = RAGPipeline(vector_store=store, llm=MockExtractiveLLM())
    dataset_items = [q.to_experiment_item() for q in questions]
    records = run_inference(
        dataset_items, pipeline, top_k=args.top_k,
        doc_chunk_counts=store.doc_chunk_counts(),
    )

    # ---- 4. scoring ----
    taxonomy_config = TaxonomyConfig()
    rows = score_records(records, taxonomy_config)
    report = aggregate(rows, taxonomy_config=taxonomy_config)

    # ---- 5. provenance ----
    report["experiment"] = {
        "tag": args.tag or f"{args.dataset}_{args.split}",
        "dataset": args.dataset,
        "split": args.split,
        "n_questions": len(questions),
        "n_documents": len(documents),
        "loader_skipped": result.skipped,
        "corpus": {k: v for k, v in corpus_stats.as_dict().items() if k != "chunks_per_document"},
        "retrieval": {
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedder": embedder_name,
        },
        "generator": {
            "name": "MockExtractiveLLM",
            "kind": "extractive control condition",
            "note": (
                "No language model was called. The extractive control copies the "
                "sentence from retrieved context with the greatest overlap with the "
                "question, which bounds generation quality from below and makes the "
                "run fully deterministic. Generation-side conclusions must not be "
                "drawn from it; retrieval and evidence measurements are unaffected "
                "because retrieval is real."
            ),
        },
        "license": loader.license_spdx,
        "source_url": loader.source_url,
    }
    report["provenance"] = collect_provenance(
        dataset={
            "name": args.dataset,
            "raw_file": raw_path.name,
            "sha256": file_checksum(raw_path),
            "split": args.split,
            "limit": args.limit,
        },
        taxonomy={
            "version": taxonomy_config.version,
            "fingerprint": taxonomy_config.fingerprint(),
        },
        pipeline={"embedder": embedder_name, "llm": "MockExtractiveLLM", "top_k": args.top_k},
    )

    out_dir = Path(args.out)
    write_outputs(rows, report, out_dir, records=records)

    ev = report.get("evidence", {})
    print(f"\n=== {report['experiment']['tag']} ===")
    print(f"questions            : {report['total']}")
    print(f"documents / chunks   : {len(documents)} / {corpus_stats.n_chunks}")
    print(f"doc recall@k (legacy): {report['recall_at_k_mean']}")
    print(f"doc hit-rate@k       : {report['retrieval_corrected'].get('hit_rate_at_k_mean')}")
    print(f"evidence complete    : {ev.get('evidence_complete_rate')}")
    print(f"evidence recall      : {ev.get('evidence_recall_mean')}")
    print(f"attribution          : {ev.get('attribution_stages')}")
    print(f"failure_modes_v2     : {report['failure_modes_v2']}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
