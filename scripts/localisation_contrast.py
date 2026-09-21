#!/usr/bin/env python
"""Corpus contrasts behind the localisation study (results/localisation/corpus_contrast.json).

Why might localisation behave differently on QASPER and NQ?  This computes, from the same
chunks and the stored MiniLM vectors used in the localisation study:

  - intra-document homogeneity: mean pairwise cosine between chunks of the same document
  - query-gold margin: cosine(query, best gold chunk) - cosine(query, best non-gold chunk)
  - lexical overlap between the question and its gold chunk vs the best non-gold chunk
  - question length, gold span length and spans per question

The figures are descriptive; they suggest hypotheses and do not test them.  A question is
included only if its gold document has at least one gold and one non-gold chunk (the
margins are undefined otherwise); on NQ this excludes one of the 300 questions.

    python scripts/localisation_contrast.py --out results/localisation/corpus_contrast.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.corpus import chunk_documents  # noqa: E402
from src.data.loaders import get_loader  # noqa: E402
from src.rag.chunking import DocumentChunker  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "be",
    "by", "with", "as", "at", "from", "that", "this", "which", "what", "how", "do", "does", "did",
    "it", "its", "their", "they", "them", "we", "our", "there", "these", "those", "than", "then",
    "not", "no", "who", "whom", "whose", "when", "where", "why",
}

CORPORA = {
    "qasper": ("data/raw/qasper-dev-v0.3.json", "dev", "data/build/index_qasper_dev_300_emb_minilm", "exp_qasper_dev"),
    "nq": ("data/raw/nq-validation-0.parquet", "validation", "data/build/index_nq_val_300_fixed", "exp_nq_validation"),
}


def toks(s: str) -> set[str]:
    return {t for t in TOKEN.findall(s.lower()) if t not in STOP and len(t) > 1}


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def contrast(dataset: str, raw: str, split: str, index_dir: str, collection: str, limit: int) -> dict:
    import chromadb
    from sentence_transformers import SentenceTransformer

    loaded = get_loader(dataset).load(ROOT / raw, split=split, limit=limit)
    chunks, _ = chunk_documents(loaded.documents, DocumentChunker(256, 32))
    col = chromadb.PersistentClient(path=str(ROOT / index_dir)).get_collection(collection)
    got = col.get(include=["embeddings"])
    by_id = dict(zip(got["ids"], got["embeddings"], strict=True))
    emb = np.asarray([by_id[c.chunk_id] for c in chunks], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    doc_idx: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_idx.setdefault(c.doc_id, []).append(i)
    homog = []
    for idxs in doc_idx.values():
        if len(idxs) < 2:
            continue
        m = emb[idxs] @ emb[idxs].T
        n = len(idxs)
        homog.append((m.sum() - n) / (n * (n - 1)))

    qs = [q for q in loaded.questions if q.is_answerable and q.supporting_spans and q.supporting_spans[0].doc_id in doc_idx]
    qemb = model.encode([q.question for q in qs], batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    margins, lex_gold, lex_other, qlen, slen, excluded = [], [], [], [], [], 0
    for q, qe in zip(qs, qemb, strict=True):
        d = q.supporting_spans[0].doc_id
        idxs = doc_idx[d]
        gold = [i for i in idxs if any(s.doc_id == d and overlap(chunks[i].start_char, chunks[i].end_char, s.start_char, s.end_char) > 0
                                       for s in q.supporting_spans)]
        other = [i for i in idxs if i not in gold]
        if not gold or not other:
            excluded += 1
            continue
        sims = emb[idxs] @ qe
        margins.append(max(sims[idxs.index(i)] for i in gold) - max(sims[idxs.index(i)] for i in other))
        qt = toks(q.question)
        lex_gold.append(max(len(qt & toks(chunks[i].text)) / max(1, len(qt)) for i in gold))
        lex_other.append(max(len(qt & toks(chunks[i].text)) / max(1, len(qt)) for i in other))
        qlen.append(len(TOKEN.findall(q.question.lower())))
        spans_here = [s for s in q.supporting_spans if s.doc_id == d]
        slen.append(sum(s.end_char - s.start_char for s in spans_here) / max(1, len(spans_here)))
    margins = np.array(margins)
    lg, lo = np.array(lex_gold), np.array(lex_other)
    return {
        "n_questions": int(len(margins)),
        "n_excluded_no_gold_or_no_nongold_chunk": excluded,
        "intra_document_mean_cosine": round(float(np.mean(homog)), 4),
        "query_gold_margin_mean": round(float(margins.mean()), 4),
        "query_gold_margin_share_positive": round(float((margins > 0).mean()), 4),
        "query_gold_margin_share_within_0.02": round(float((np.abs(margins) <= 0.02).mean()), 4),
        "lexical_overlap_gold_mean": round(float(lg.mean()), 4),
        "lexical_overlap_best_other_mean": round(float(lo.mean()), 4),
        "share_gold_strictly_most_lexical": round(float((lg > lo).mean()), 4),
        "share_gold_zero_lexical_overlap": round(float((lg == 0).mean()), 4),
        "question_length_words_median": float(np.median(qlen)),
        "gold_span_chars_median": float(np.median(slen)),
        "spans_per_question_mean": round(float(np.mean([len(q.supporting_spans) for q in qs])), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {ds: contrast(ds, *spec, limit=args.limit) for ds, spec in CORPORA.items()}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
