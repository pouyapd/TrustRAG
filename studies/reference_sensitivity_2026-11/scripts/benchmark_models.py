#!/usr/bin/env python
"""Measure CPU throughput of candidate retrieval models on real chunks (feasibility study).

Encodes 48 QASPER chunks (256 cl100k tokens each, as in the study) with each bi-encoder and
scores 48 (question, chunk) pairs with each cross-encoder, after a warm-up batch. Backends:
PyTorch, and ONNX Runtime where the model repository ships an official ONNX export. Writes
results/feasibility/benchmark.json; failures are recorded with their error, not skipped.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
sys.path.insert(0, str(REPO))
from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402

DENSE = ["Snowflake/snowflake-arctic-embed-xs", "Snowflake/snowflake-arctic-embed-s",
         "ibm-granite/granite-embedding-30m-english", "Alibaba-NLP/gte-modernbert-base", "BAAI/bge-m3"]
CROSS = ["mixedbread-ai/mxbai-rerank-xsmall-v1", "jinaai/jina-reranker-v1-turbo-en",
         "Alibaba-NLP/gte-reranker-modernbert-base"]
REFERENCE = {"dense": "sentence-transformers/all-MiniLM-L6-v2", "cross": "cross-encoder/ms-marco-MiniLM-L-6-v2"}


def main() -> int:
    loader = get_loader("qasper")
    res = loader.load(REPO / "data/raw/qasper-dev-v0.3.json", split="dev", limit=40)
    chunks, _ = chunk_documents(res.documents, DocumentChunker(chunk_size=256, chunk_overlap=32))
    texts = [c.text for c in chunks[:48]]
    pairs = [(res.questions[0].question, t) for t in texts]
    out = {"platform": platform.platform(), "python": sys.version.split()[0], "n_items": len(texts), "results": []}
    from sentence_transformers import CrossEncoder, SentenceTransformer

    def run(kind, rid, backend):
        rec = {"model": rid, "kind": kind, "backend": backend}
        try:
            t0 = time.time()
            kw = {"device": "cpu", "trust_remote_code": True}
            if backend == "onnx":
                kw["backend"] = "onnx"
            m = SentenceTransformer(rid, **kw) if kind == "dense" else CrossEncoder(rid, max_length=512, **kw)
            rec["load_seconds"] = round(time.time() - t0, 1)
            if kind == "dense":
                m.encode(texts[:4])
                t = time.time(); m.encode(texts, batch_size=16)
                rec["items_per_second"] = round(len(texts) / (time.time() - t), 2)
                rec["max_seq_length"] = m.max_seq_length
            else:
                m.predict(pairs[:4])
                t = time.time(); m.predict(pairs, batch_size=16)
                rec["items_per_second"] = round(len(pairs) / (time.time() - t), 2)
        except Exception as e:  # recorded, not hidden
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            rec["trace_tail"] = traceback.format_exc()[-400:]
        print(json.dumps(rec), flush=True)
        out["results"].append(rec)

    run("dense", REFERENCE["dense"], "torch")
    run("cross", REFERENCE["cross"], "torch")
    for rid in DENSE:
        run("dense", rid, "torch")
        run("dense", rid, "onnx")
    for rid in CROSS:
        run("cross", rid, "torch")
        run("cross", rid, "onnx")
    dst = REV / "results/feasibility"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "benchmark.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
