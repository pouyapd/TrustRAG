#!/usr/bin/env python
"""Score within-document evidence localisation for the revised study (revision 2026-11).

For every answerable question with located evidence, every chunk of the question's gold
document is ranked against the question by each localiser, and the full best-first
ordering is stored together with every annotator's gold chunks and the chunk character
ranges. Any evidence reference (first annotator, union, NQ's official rule, plurality,
agreement, a single drawn annotator) can then be rescored from the stored file without
rerunning a model (see analyse_v2.py).

For BM25 and the bi-encoders the script also stores the global top-20 over the whole
corpus of the run, which gives reach (A), span coverage (C) and admitted chunks.

The chunker, BM25 parameters, model identities, prefixes and cross-encoder settings are
the original study's (scripts/localisation_probe.py, scripts/*_reference_sensitivity.py).
It imports the repository code read-only and writes only under --out.

    python score_v2.py --corpus qasper --raw data/raw/qasper-dev-v0.3.json:dev --limit 300 \
        --out results/repro_qasper_300.json.gz          # reproduction check
    python score_v2.py --corpus qasper --raw <dev>:dev --raw <test>:test --out ...
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_bm25_baseline import tokenize  # noqa: E402

from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.evaluation.references import gold_chunks  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402
from src.rag.embedders import EMBEDDERS  # noqa: E402

CROSS = {
    "ce_msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge_reranker_base": "BAAI/bge-reranker-base",
    # added in the revision: current (2025) ModernBERT-based models
    "gte_reranker_modernbert": "Alibaba-NLP/gte-reranker-modernbert-base",
}
EXTRA_DENSE = {
    # key: (repo_id, query_prefix, passage_prefix)
    "gte_modernbert": ("Alibaba-NLP/gte-modernbert-base", "", ""),
}
STUDY_MODELS = ("bm25", "minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base")
GLOBAL_K = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cached_revision(repo_id: str) -> str | None:
    snap = Path.home() / ".cache/huggingface/hub" / ("models--" + repo_id.replace("/", "--")) / "refs/main"
    return snap.read_text().strip() if snap.exists() else None


def bm25_matrix(token_lists: list[list[str]], k1: float = 1.5, b: float = 0.75):
    """Chunk x term matrix of BM25 weights; a query's scores are W @ query-term counts.

    Identical to the probe's scorer (corpus-level Robertson/Sparck-Jones IDF with the +1
    guard, repeated query terms counted again), vectorised.
    """
    from collections import Counter

    from scipy.sparse import csr_matrix

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    lengths = np.array([len(t) for t in token_lists], dtype=np.float64)
    avg = lengths.mean()
    df: Counter[str] = Counter()
    freqs = [Counter(t) for t in token_lists]
    for f in freqs:
        df.update(f.keys())
    n = len(token_lists)
    idf = {t: np.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    for i, f in enumerate(freqs):
        norm = k1 * (1 - b + b * lengths[i] / avg)
        for t, c in f.items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(idf[t] * c * (k1 + 1) / (c + norm))
    return csr_matrix((vals, (rows, cols)), shape=(n, len(vocab))), vocab


def query_vector(terms: list[str], vocab: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab))
    for t in terms:
        j = vocab.get(t)
        if j is not None:
            v[j] += 1
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("qasper", "nq"), required=True)
    ap.add_argument("--raw", action="append", required=True, help="path:split, repeatable")
    ap.add_argument("--limit", type=int, default=None, help="per raw file, as in the study")
    ap.add_argument("--models", default=",".join(STUDY_MODELS))
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--ce-max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    models = [m for m in args.models.split(",") if m]

    loader_name = "qasper" if args.corpus == "qasper" else "nq"
    documents, questions_raw, raw_files = [], [], []
    seen_docs: set[str] = set()
    for item in args.raw:
        path, split = item.rsplit(":", 1)
        loader = get_loader(loader_name)
        loader.keep_annotator_spans = True
        res = loader.load(Path(path), split=split, limit=args.limit)
        text_by_id = {d.doc_id: d.text for d in documents}
        rename: dict[str, str] = {}
        for d in res.documents:
            if d.doc_id in seen_docs:
                # The same page asked about in two files: one copy when the text is identical;
                # otherwise (text extraction differs between dataset files) both are kept as
                # separate documents, each with its own questions.
                if text_by_id[d.doc_id] == d.text:
                    continue
                new_id = f"{d.doc_id}#file{len(raw_files)}"
                rename[d.doc_id] = new_id
                d = dataclasses.replace(d, doc_id=new_id)
            seen_docs.add(d.doc_id)
            documents.append(d)
        questions_raw += [(split, q, rename) for q in res.questions]
        raw_files.append({"path": Path(path).as_posix(), "split": split, "sha256": sha256_file(Path(path)),
                          "questions_loaded": len(res.questions), "documents": len(res.documents)})
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks, stats = chunk_documents(documents, chunker)
    if stats.offset_mismatches:
        raise SystemExit(f"{stats.offset_mismatches} chunk offset mismatches")
    doc_chunks: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_chunks.setdefault(c.doc_id, []).append(i)
    doc_index = {d: n for n, d in enumerate(doc_chunks)}
    print(f"{len(questions_raw)} questions, {len(documents)} docs, {len(chunks)} chunks", flush=True)

    records = []
    for split, q, rename in questions_raw:
        spans = [dataclasses.replace(s, doc_id=rename.get(s.doc_id, s.doc_id)) for s in q.supporting_spans]
        spans = [s for s in spans if s.doc_id in doc_chunks]
        if not q.is_answerable or not spans:
            continue
        doc_id = sorted({s.doc_id for s in spans})[0]
        idxs = doc_chunks[doc_id]
        ranges = [(chunks[i].start_char, chunks[i].end_char) for i in idxs]
        study_gold = sorted(gold_chunks(ranges, [(s.start_char, s.end_char) for s in spans if s.doc_id == doc_id]))
        if args.corpus == "qasper":
            ann = q.metadata["annotator_evidence"]
            by_ann = {str(a["annotator"]): sorted(gold_chunks(ranges, [tuple(x) for x in a["spans"]]))
                      for a in ann if a.get("status") == "ok"}
        else:
            ann = q.metadata["annotator_long_answers"]
            by_ann = {str(a["annotator"]): sorted(gold_chunks(ranges, [(a["start_char"], a["end_char"])]))
                      for a in ann if a.get("status") == "ok"}
        records.append({"question_id": q.question_id, "split": split, "doc_id": doc_id, "D": len(idxs),
                        "chunk_ranges": ranges, "annotations": ann, "gold_by_annotator": by_ann,
                        "study_gold": study_gold, "question": q.question, "_idxs": idxs, "orders": {},
                        "global_top": {}})
    print(f"{len(records)} answerable questions with located evidence", flush=True)
    timing: dict[str, float] = {}

    def store(model: str, rec: dict, scores: np.ndarray) -> None:
        rec["orders"][model] = [int(i) for i in np.argsort(-scores, kind="stable")]

    def store_global(model: str, rec: dict, scores_all: np.ndarray) -> None:
        top = np.argsort(-scores_all, kind="stable")[:GLOBAL_K]
        pos_in_doc = {g: p for p, g in enumerate(rec["_idxs"])}
        rec["global_top"][model] = [[doc_index[chunks[int(g)].doc_id], pos_in_doc.get(int(g), -1)] for g in top]

    for model in models:
        t1 = time.time()
        if model == "bm25":
            W, vocab = bm25_matrix([tokenize(c.text) for c in chunks])
            for rec in records:
                s_all = W @ query_vector(tokenize(rec["question"]), vocab)
                store(model, rec, s_all[rec["_idxs"]])
                store_global(model, rec, s_all)
        elif model in EMBEDDERS or model in EXTRA_DENSE:
            from sentence_transformers import SentenceTransformer
            if model in EMBEDDERS:
                spec = EMBEDDERS[model]
                repo_id, qp, pp = spec.repo_id, spec.query_prefix, spec.passage_prefix
            else:
                repo_id, qp, pp = EXTRA_DENSE[model]
            enc = SentenceTransformer(repo_id, device="cpu")
            emb = enc.encode([pp + c.text for c in chunks], batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            qemb = enc.encode([qp + r["question"] for r in records], batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            for n, rec in enumerate(records):
                s_all = emb @ qemb[n]
                store(model, rec, s_all[rec["_idxs"]])
                store_global(model, rec, s_all)
        elif model in CROSS:
            from sentence_transformers import CrossEncoder
            enc = CrossEncoder(CROSS[model], device="cpu", max_length=args.ce_max_length)
            for n, rec in enumerate(records, 1):
                pairs = [(rec["question"], chunks[i].text) for i in rec["_idxs"]]
                store(model, rec, np.asarray(enc.predict(pairs, batch_size=32, show_progress_bar=False),
                                             dtype=np.float64).reshape(-1))
                if n % 200 == 0:
                    print(f"  {model}: {n}/{len(records)} {time.time() - t1:.0f}s", flush=True)
        else:
            raise SystemExit(f"unknown model {model}")
        timing[model] = round(time.time() - t1, 1)
        print(f"{model}: {timing[model]}s", flush=True)
        dump(records, [m for m in models if m in timing], timing, raw_files, documents, chunks,
             doc_chunks, args, t0)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


def dump(records, models, timing, raw_files, documents, chunks, doc_chunks, args, t0) -> None:
    """Write everything scored so far (atomically), so a late failure loses nothing."""
    repo_ids = {m: ("BM25 (Okapi, k1=1.5, b=0.75, corpus-level IDF)" if m == "bm25" else
                    EMBEDDERS[m].repo_id if m in EMBEDDERS else
                    EXTRA_DENSE[m][0] if m in EXTRA_DENSE else CROSS[m]) for m in models}
    out = {
        "description": "Within-document best-first chunk orderings per question and localiser, every "
                       "annotator's gold chunk positions, and global top-20 (doc index, position in gold "
                       "document or -1) for BM25 and bi-encoders. Positions index the gold document's "
                       "chunks in order. No dataset text except the question is stored.",
        "corpus": args.corpus,
        "config": {"raw_files": raw_files, "limit_per_file": args.limit, "chunk_size": args.chunk_size,
                   "chunk_overlap": args.chunk_overlap, "ce_max_length": args.ce_max_length,
                   "n_documents": len(documents), "n_chunks": len(chunks), "global_k": GLOBAL_K,
                   "models": {m: {"identity": repo_ids[m],
                                  "cached_revision": None if m == "bm25" else cached_revision(repo_ids[m])}
                              for m in models},
                   "runtime_seconds": dict(timing), "total_seconds": round(time.time() - t0, 1),
                   "python": sys.version.split()[0], "platform": platform.platform(),
                   "torch_threads": __import__("torch").get_num_threads()},
        "doc_ids": list(doc_chunks),
        "questions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in records],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, args.out)
    print(f"  saved {args.out} ({', '.join(models)})", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
