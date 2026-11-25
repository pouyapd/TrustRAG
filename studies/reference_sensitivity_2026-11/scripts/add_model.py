#!/usr/bin/env python
"""Add a bi-encoder's or cross-encoder's within-document orderings to a scored file, resumably.

Generalises add_cross_encoder.py. Rebuilds documents and chunks exactly as score_v2.py did
(loaders, chunker and duplicate-page handling from the file's own config), checks each
question's chunk ranges against the stored ones, and for every question still lacking the
model scores all chunks of its gold document. Bi-encoders encode the gold document's chunks
and the question with the model's own documented prompts (config_sentence_transformers.json);
no global top-k is computed for added models. Checkpoints atomically every --every questions;
rerunning continues where the last checkpoint stopped.

    python add_model.py results/raw/qasper_full.json.gz --model granite30m
    python add_model.py results/raw/nq_full.json.gz --model jina_turbo
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
sys.path.insert(0, str(REPO))
from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402

MODELS = {
    # key: (kind, repo id, backend, notes)
    "granite30m": ("dense", "ibm-granite/granite-embedding-30m-english", "torch", "IBM Granite Embedding, Dec 2024"),
    "arctic_s": ("dense", "Snowflake/snowflake-arctic-embed-s", "torch", "Snowflake Arctic Embed S, Apr 2024"),
    "bge_m3": ("dense", "BAAI/bge-m3", "torch", "BGE-M3 dense, Jan 2024"),
    "jina_turbo": ("cross", "jinaai/jina-reranker-v1-turbo-en", "onnx", "Jina reranker v1 turbo, Apr 2024"),
    "ce_msmarco_minilm": ("cross", "cross-encoder/ms-marco-MiniLM-L-6-v2", "torch", "study model"),
}


def save(path: Path, data: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def rebuild_chunks(cfg: dict, corpus: str) -> dict[str, list]:
    loader_name = "qasper" if corpus == "qasper" else "nq"
    documents, seen = [], set()
    for n, rf in enumerate(cfg["raw_files"]):
        loader = get_loader(loader_name)
        res = loader.load(REPO / rf["path"], split=rf["split"], limit=cfg["limit_per_file"])
        text_by_id = {d.doc_id: d.text for d in documents}
        for d in res.documents:
            if d.doc_id in seen:
                if text_by_id[d.doc_id] == d.text:
                    continue
                d = dataclasses.replace(d, doc_id=f"{d.doc_id}#file{n}")
            seen.add(d.doc_id)
            documents.append(d)
    chunks, stats = chunk_documents(documents, DocumentChunker(chunk_size=cfg["chunk_size"],
                                                               chunk_overlap=cfg["chunk_overlap"]))
    if stats.offset_mismatches or len(chunks) != cfg["n_chunks"]:
        raise SystemExit("chunks differ from the scored file")
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    return by_doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--only-ids", default=None, help="JSON list of question ids to score (subset runs)")
    args = ap.parse_args()
    kind, repo_id, backend, note = MODELS[args.model]
    path = Path(args.scored)
    data = json.load(gzip.open(path, "rt", encoding="utf-8"))
    cfg = data["config"]
    by_doc = rebuild_chunks(cfg, data["corpus"])
    only = set(json.loads(Path(args.only_ids).read_text())) if args.only_ids else None

    from sentence_transformers import CrossEncoder, SentenceTransformer
    kw = {"device": "cpu", "trust_remote_code": True}
    if backend == "onnx":
        kw["backend"] = "onnx"
    if kind == "dense":
        enc = SentenceTransformer(repo_id, **kw)
        qprompt = (enc.prompts or {}).get("query", "")
        dprompt = (enc.prompts or {}).get("document", (enc.prompts or {}).get("passage", ""))
    else:
        enc = CrossEncoder(repo_id, max_length=cfg["ce_max_length"], **kw)
        qprompt = dprompt = ""
    todo = [q for q in data["questions"] if args.model not in q["orders"] and (only is None or q["question_id"] in only)]
    print(f"{len(todo)} questions to score with {args.model} ({repo_id}, {backend}); "
          f"query prompt {qprompt!r}", flush=True)
    t0, done = time.time(), 0
    doc_cache: dict[str, np.ndarray] = {}
    for q in todo:
        chunks = by_doc[q["doc_id"]]
        if [[c.start_char, c.end_char] for c in chunks] != [list(r) for r in q["chunk_ranges"]]:
            raise SystemExit(f"{q['question_id']}: chunk ranges differ")
        if kind == "dense":
            if q["doc_id"] not in doc_cache:
                doc_cache = {q["doc_id"]: enc.encode([dprompt + c.text for c in chunks], batch_size=16,
                                                     normalize_embeddings=True, show_progress_bar=False)}
            qv = enc.encode([qprompt + q["question"]], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = doc_cache[q["doc_id"]] @ qv
        else:
            scores = np.asarray(enc.predict([(q["question"], c.text) for c in chunks], batch_size=16,
                                            show_progress_bar=False), dtype=np.float64).reshape(-1)
        q["orders"][args.model] = [int(i) for i in np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")]
        done += 1
        if done % args.every == 0:
            save(path, data)
            print(f"  {done}/{len(todo)} {time.time() - t0:.0f}s (checkpoint)", flush=True)
    target = [q for q in data["questions"] if only is None or q["question_id"] in only]
    if all(args.model in q["orders"] for q in target):
        snap = Path.home() / ".cache/huggingface/hub" / ("models--" + repo_id.replace("/", "--")) / "refs/main"
        cfg["models"][args.model] = {"identity": repo_id, "backend": backend, "note": note,
                                     "query_prompt": qprompt, "document_prompt": dprompt,
                                     "cached_revision": snap.read_text().strip() if snap.exists() else None,
                                     "scored_questions": len(target)}
        cfg["runtime_seconds"][args.model] = cfg["runtime_seconds"].get(args.model, 0) + round(time.time() - t0, 1)
    save(path, data)
    print(f"done: {done} scored in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
