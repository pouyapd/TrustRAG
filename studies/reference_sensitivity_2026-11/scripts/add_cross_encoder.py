#!/usr/bin/env python
"""Add a cross-encoder's within-document orderings to an existing score_v2 output, resumably.

Rebuilds the documents and chunks exactly as score_v2.py did (same loaders, chunker and
duplicate-page handling, read from the file's own config), checks that every question's
chunk ranges match the stored ones, scores the questions that do not yet have the model,
and checkpoints the file atomically every --every questions. Rerunning continues where the
last checkpoint stopped.

    python add_cross_encoder.py results/raw/qasper_full.json.gz --model ce_msmarco_minilm
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402

CROSS = {"ce_msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
         "bge_reranker_base": "BAAI/bge-reranker-base"}


def save(path: Path, data: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--model", required=True, choices=sorted(CROSS))
    ap.add_argument("--every", type=int, default=100)
    args = ap.parse_args()
    path = Path(args.scored)
    data = json.load(gzip.open(path, "rt", encoding="utf-8"))
    cfg = data["config"]
    loader_name = "qasper" if data["corpus"] == "qasper" else "nq"

    documents, seen = [], set()
    for n, rf in enumerate(cfg["raw_files"]):
        loader = get_loader(loader_name)
        res = loader.load(REPO / rf["path"] if not Path(rf["path"]).is_absolute() else Path(rf["path"]),
                          split=rf["split"], limit=cfg["limit_per_file"])
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

    from sentence_transformers import CrossEncoder
    enc = CrossEncoder(CROSS[args.model], device="cpu", max_length=cfg["ce_max_length"])
    todo = [q for q in data["questions"] if args.model not in q["orders"]]
    print(f"{len(todo)} of {len(data['questions'])} questions to score with {args.model}", flush=True)
    t0, done = time.time(), 0
    for q in todo:
        doc_chunks = by_doc[q["doc_id"]]
        if [[c.start_char, c.end_char] for c in doc_chunks] != [list(r) for r in q["chunk_ranges"]]:
            raise SystemExit(f"{q['question_id']}: chunk ranges differ")
        scores = np.asarray(enc.predict([(q["question"], c.text) for c in doc_chunks], batch_size=32,
                                        show_progress_bar=False), dtype=np.float64).reshape(-1)
        q["orders"][args.model] = [int(i) for i in np.argsort(-scores, kind="stable")]
        done += 1
        if done % args.every == 0:
            save(path, data)
            print(f"  {done}/{len(todo)} {time.time() - t0:.0f}s (checkpoint)", flush=True)
    if all(args.model in q["orders"] for q in data["questions"]):
        cfg["models"][args.model] = {"identity": CROSS[args.model], "cached_revision": None}
        snap = Path.home() / ".cache/huggingface/hub" / ("models--" + CROSS[args.model].replace("/", "--")) / "refs/main"
        if snap.exists():
            cfg["models"][args.model]["cached_revision"] = snap.read_text().strip()
        cfg["runtime_seconds"][args.model] = cfg["runtime_seconds"].get(args.model, 0) + round(time.time() - t0, 1)
    save(path, data)
    print(f"done: {done} scored in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
