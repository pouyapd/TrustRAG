#!/usr/bin/env python
"""Robustness checks for the within-document localisation study.

Five things a reviewer asks about the study that the main report does not answer:

1. QASPER's questions are clustered in documents (290 questions in 111 papers, up to ten per
   paper), and the paired McNemar tests treat questions as independent. ``--bootstrap``
   resamples *documents* with replacement and reports a percentile interval for every
   pairwise hit@1 difference, so the main conclusions can be checked against clustering.
2. A chunk is "gold" when it overlaps a gold span by at least one character, and
   all-MiniLM-L6-v2 reads only the first 256 wordpieces of a chunk. ``--audit-chunks``
   measures how long the chunks are in that tokenizer, how often gold evidence lies beyond
   the cut, and how small the smallest counted overlaps are.
3. Model revisions were not pinned when the study ran. ``--model-metadata`` records the
   revision hashes and input limits of every model as they sit in the local Hugging Face
   cache, so the paper can report what was actually run.
4. The summary files report hit@1/3/5 only. ``--hit-at-k`` re-reads the stored per-question
   gold ranks and reports hit@k for k in {1, 3, 5, 10, 20} for the seven models and the
   three variants, so the deeper cut-offs quoted in the paper trace to a committed file.
5. Reach and localisation are reported separately. ``--reach-association`` cross-tabulates,
   per retriever, whether the gold document was reached at k = 5 against whether the gold
   chunk was ranked first inside it, with Fisher's exact test (QASPER only; NQ reach is
   saturated).

Each mode writes one JSON file; nothing here changes a result of the study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ["bm25", "minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base"]

HF_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/e5-small-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "ce_msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge_reranker_base": "BAAI/bge-reranker-base",
    "reader_qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "reader_smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

# What the model cards say about fine-tuning data, as read on 2026-09-19. "undocumented"
# means the card gives no dataset list; nothing is inferred beyond the card.
TRAINING_DATA = {
    "minilm": "1B sentence pairs incl. MS MARCO (9.1M triplets) and Natural Questions (100,231 pairs); model card lists the datasets",
    "bge": "model card cites MS MARCO and NQ among fine-tuning sets (BGE technical report)",
    "e5": "model card cites MS MARCO and NQ among fine-tuning sets (E5 technical report)",
    "mpnet": "1B sentence pairs incl. MS MARCO (9.1M triplets) and Natural Questions (100,231 pairs); model card lists the datasets",
    "ce_msmarco_minilm": "MS MARCO passage ranking",
    "bge_reranker_base": "undocumented: model card states only 'multilingual pair data'",
    "reader_qwen": "not documented at dataset level",
    "reader_smollm": "not documented at dataset level in the card",
}


# --------------------------------------------------------------------------- rows
def load_rows(results_dir: Path, dataset: str) -> dict[str, dict[str, tuple[str, int]]]:
    """question_id -> (doc_id, hit@1) for every model, from the per-question row files."""
    rows: dict[str, dict[str, tuple[str, int]]] = {}
    probe_files = [results_dir / f"within_document_{dataset}.json"]
    if dataset == "nq":
        probe_files.append(results_dir / "within_document_nq_dense.json")
    for f in probe_files:
        data = json.loads(f.read_text(encoding="utf-8"))["within_document_rows"]
        for model, model_rows in data.items():
            if model in MODELS:
                rows[model] = {r["question_id"]: (r["doc_id"], int(r["hit@1"])) for r in model_rows}
    for f, key in [
        (results_dir / f"localisers_{dataset}.json", "ce_msmarco_minilm"),
        (results_dir / f"reranker_bge_base_{dataset}.json", "bge_reranker_base"),
    ]:
        data = json.loads(f.read_text(encoding="utf-8"))
        for value in data.values():
            if isinstance(value, dict) and isinstance(value.get("cross_encoder"), list):
                rows[key] = {r["question_id"]: (r["doc_id"], int(r["hit@1"])) for r in value["cross_encoder"]}
    missing = [m for m in MODELS if m not in rows]
    if missing:
        raise SystemExit(f"rows missing for {missing}")
    return rows


def cluster_bootstrap(diffs_by_cluster: list[list[int]], n_boot: int = 4000, seed: int = 0) -> dict:
    """Percentile bootstrap of a mean paired difference, resampling clusters with replacement.

    ``diffs_by_cluster`` holds, per document, the per-question differences hit@1(A) - hit@1(B).
    The observed statistic is the mean over all questions; each resample draws documents
    with replacement and pools their questions, so the interval reflects between-document
    variation as well as within-document variation.
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(diffs_by_cluster)
    flat = np.array([d for cluster in diffs_by_cluster for d in cluster], dtype=float)
    observed = float(flat.mean())
    sums = np.array([sum(c) for c in diffs_by_cluster], dtype=float)
    sizes = np.array([len(c) for c in diffs_by_cluster], dtype=float)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boots = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two_sided = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {
        "delta_hit@1": round(observed, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "bootstrap_p_two_sided": round(min(1.0, p_two_sided), 4),
        "n_clusters": n_clusters,
        "n_questions": int(sizes.sum()),
        "n_boot": n_boot,
        "seed": seed,
    }


VARIANTS = ["bm25_local_idf", "rrf_bm25+bge", "oracle_union_bm25+dense"]


def load_ranks(results_dir: Path, dataset: str) -> dict[str, dict[str, int]]:
    """question_id -> gold rank for the seven models and the three variants."""
    ranks: dict[str, dict[str, int]] = {}
    probe_files = [results_dir / f"within_document_{dataset}.json"]
    if dataset == "nq":
        probe_files.append(results_dir / "within_document_nq_dense.json")
    for f in probe_files:
        data = json.loads(f.read_text(encoding="utf-8"))["within_document_rows"]
        for model, model_rows in data.items():
            if model in MODELS:
                ranks[model] = {r["question_id"]: int(r["rank"]) for r in model_rows}
    extra = json.loads((results_dir / f"localisers_{dataset}.json").read_text(encoding="utf-8"))["rows"]
    ranks["ce_msmarco_minilm"] = {r["question_id"]: int(r["rank"]) for r in extra["cross_encoder"]}
    # the variants are reported with BGE-small as the dense partner; on NQ they live in a
    # separate file because the first NQ run paired BM25 with MiniLM
    partner = results_dir / f"localisers_{dataset}_bge_partner.json"
    if partner.exists():
        extra = json.loads(partner.read_text(encoding="utf-8"))["rows"]
    for v in VARIANTS:
        ranks[v] = {r["question_id"]: int(r["rank"]) for r in extra[v]}
    rer = json.loads((results_dir / f"reranker_bge_base_{dataset}.json").read_text(encoding="utf-8"))["rows"]
    ranks["bge_reranker_base"] = {r["question_id"]: int(r["rank"]) for r in rer["cross_encoder"]}
    return ranks


def run_hit_at_k(results_dir: Path, dataset: str, ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    ranks = load_ranks(results_dir, dataset)
    common = set.intersection(*(set(r) for r in ranks.values()))
    out = {"dataset": dataset, "n_questions": len(common),
           "method": "hit@k = share of the common questions whose stored gold rank is <= k",
           "models": {}, "variants": {}}
    for name, r in ranks.items():
        vals = [r[q] for q in sorted(common)]
        entry = {str(k): round(sum(1 for v in vals if v <= k) / len(vals), 3) for k in ks}
        entry["max_rank"] = max(vals)
        (out["variants"] if name in VARIANTS else out["models"])[name] = entry
    return out


def run_reach_association(results_dir: Path, dataset: str) -> dict:
    from scipy.stats import fisher_exact

    data = json.loads((results_dir / f"within_document_{dataset}.json").read_text(encoding="utf-8"))
    out = {"dataset": dataset,
           "method": "2x2 table per retriever: gold document reached at k=5 (global top-k run) x gold chunk "
                     "ranked first inside its document (within-document run); Fisher's exact test, two-sided",
           "retrievers": {}}
    for model, global_rows in data["global_rows"].items():
        within = {r["question_id"]: r for r in data["within_document_rows"][model]}
        table = [[0, 0], [0, 0]]  # [reached][ranked first]
        for g in global_rows:
            table[int(bool(g["A"]))][int(within[g["question_id"]]["rank"] == 1)] += 1
        _, p = fisher_exact(table)
        missed, reached = table
        out["retrievers"][model] = {
            "missed_doc": {"gold_not_first": missed[0], "gold_first": missed[1],
                           "hit@1": round(missed[1] / sum(missed), 3) if sum(missed) else None},
            "reached_doc": {"gold_not_first": reached[0], "gold_first": reached[1],
                            "hit@1": round(reached[1] / sum(reached), 3) if sum(reached) else None},
            "fisher_exact_p": round(float(p), 4),
        }
    return out


def run_bootstrap(results_dir: Path, dataset: str, n_boot: int, seed: int) -> dict:
    rows = load_rows(results_dir, dataset)
    out = {"dataset": dataset, "method": "document-level cluster bootstrap, percentile 95% interval",
           "n_boot": n_boot, "seed": seed, "pairs": {}}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            common = sorted(set(rows[a]) & set(rows[b]))
            by_doc: dict[str, list[int]] = {}
            for q in common:
                by_doc.setdefault(rows[a][q][0], []).append(rows[a][q][1] - rows[b][q][1])
            out["pairs"][f"{a}_vs_{b}"] = cluster_bootstrap(list(by_doc.values()), n_boot, seed)
    return out


# --------------------------------------------------------------------- chunk audit
def run_chunk_audit(dataset: str, raw: Path, split: str, limit: int) -> dict:
    """Chunk lengths in MiniLM's tokenizer, gold evidence beyond its 256-wordpiece cut, and the
    size of the largest gold overlap per question."""
    from transformers import AutoTokenizer

    from src.data.corpus import chunk_documents
    from src.data.loaders import get_loader
    from src.rag.chunking import DocumentChunker

    limit_wp = 256
    tok = AutoTokenizer.from_pretrained(HF_MODELS["minilm"])
    loaded = get_loader(dataset).load(raw, split=split, limit=limit)
    chunks, _ = chunk_documents(loaded.documents, DocumentChunker(256, 32))
    lengths = np.array([len(tok(c.text, add_special_tokens=True)["input_ids"]) for c in chunks])
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c.doc_id, []).append(i)

    def overlap(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0))

    n = fully_hidden = partly_hidden = 0
    largest_overlaps = []
    for q in loaded.questions:
        if not (q.is_answerable and q.supporting_spans):
            continue
        doc = q.supporting_spans[0].doc_id
        if doc not in by_doc:
            continue
        visible, largest = [], 0
        for i in by_doc[doc]:
            c = chunks[i]
            pieces = [(max(c.start_char, s.start_char), min(c.end_char, s.end_char))
                      for s in q.supporting_spans
                      if s.doc_id == doc and overlap(c.start_char, c.end_char, s.start_char, s.end_char) > 0]
            if not pieces:
                continue
            offsets = tok(c.text, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
            cut = offsets[limit_wp - 1][1] if len(offsets) > limit_wp else len(c.text)
            visible.append(any(o0 - c.start_char < cut for o0, _ in pieces))
            largest = max(largest, max(o1 - o0 for o0, o1 in pieces))
        if not visible:
            continue
        n += 1
        fully_hidden += not any(visible)
        partly_hidden += not all(visible)
        largest_overlaps.append(largest)
    largest_overlaps = np.array(largest_overlaps)
    return {
        "dataset": dataset,
        "tokenizer": HF_MODELS["minilm"],
        "wordpiece_limit": limit_wp,
        "n_chunks": int(len(chunks)),
        "wordpieces_per_chunk": {"median": float(np.median(lengths)), "p90": float(np.percentile(lengths, 90)),
                                  "max": int(lengths.max()),
                                  "share_over_256": round(float((lengths > 256).mean()), 4),
                                  "share_over_384": round(float((lengths > 384).mean()), 4),
                                  "share_over_512": round(float((lengths > 512).mean()), 4)},
        "n_questions": n,
        "questions_all_gold_beyond_cut": {"n": int(fully_hidden), "share": round(fully_hidden / n, 4)},
        "questions_some_gold_beyond_cut": {"n": int(partly_hidden), "share": round(partly_hidden / n, 4)},
        "largest_gold_overlap_chars": {"min": int(largest_overlaps.min()), "median": float(np.median(largest_overlaps)),
                                       "n_below_50": int((largest_overlaps < 50).sum()),
                                       "share_below_50": round(float((largest_overlaps < 50).mean()), 4)},
    }


# ------------------------------------------------------------------ model metadata
def run_model_metadata() -> dict:
    from huggingface_hub import constants

    cache = Path(constants.HF_HUB_CACHE)
    out = {"hf_hub_cache": str(cache), "read_on": "2026-09-19", "models": {}}
    for key, name in HF_MODELS.items():
        d = cache / ("models--" + name.replace("/", "--"))
        entry: dict = {"model_id": name, "training_data_per_model_card": TRAINING_DATA[key]}
        if not d.exists():
            entry["revision"] = None
            entry["note"] = "not in the local cache"
            out["models"][key] = entry
            continue
        ref = d / "refs" / "main"
        entry["revision"] = ref.read_text().strip() if ref.exists() else None
        entry["cached_snapshots"] = sorted(p.name for p in (d / "snapshots").iterdir())
        snap = d / "snapshots" / (entry["revision"] or entry["cached_snapshots"][-1])
        if len(entry["cached_snapshots"]) > 1:
            # more than one revision was fetched over the project's life; say whether they differ
            digests = {
                s: hashlib.sha256((d / "snapshots" / s / "model.safetensors").read_bytes()).hexdigest()[:16]
                for s in entry["cached_snapshots"]
                if (d / "snapshots" / s / "model.safetensors").exists()
            }
            entry["weights_sha256_by_snapshot"] = digests
            entry["weights_identical_across_snapshots"] = len(set(digests.values())) == 1
        for fname in ("sentence_bert_config.json", "config.json", "tokenizer_config.json"):
            p = snap / fname
            if not p.exists():
                continue
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if fname == "sentence_bert_config.json":
                entry["max_seq_length"] = cfg.get("max_seq_length")
            elif fname == "config.json":
                entry["architecture"] = (cfg.get("architectures") or [None])[0]
                entry["max_position_embeddings"] = cfg.get("max_position_embeddings")
                entry["hidden_size"] = cfg.get("hidden_size")
                entry["num_layers"] = cfg.get("num_hidden_layers")
            else:
                entry["tokenizer_class"] = cfg.get("tokenizer_class")
                entry["tokenizer_model_max_length"] = cfg.get("model_max_length")
        out["models"][key] = entry
    # cross-encoders were scored with max_length=512 in localisation_extra.py
    for key in ("ce_msmarco_minilm", "bge_reranker_base"):
        out["models"][key]["effective_input_length"] = 512
        out["models"][key]["effective_input_length_source"] = "CrossEncoder(..., max_length=512) in scripts/localisation_extra.py"
    for key in ("minilm", "bge", "e5", "mpnet"):
        out["models"][key]["effective_input_length"] = out["models"][key].get("max_seq_length")
        out["models"][key]["effective_input_length_source"] = "sentence_bert_config.json max_seq_length"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/localisation")
    ap.add_argument("--bootstrap", choices=["qasper", "nq"], help="document-cluster bootstrap for this corpus")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-chunks", choices=["qasper", "nq"], help="chunk-length / overlap audit (needs raw data)")
    ap.add_argument("--raw")
    ap.add_argument("--split")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model-metadata", action="store_true")
    ap.add_argument("--hit-at-k", choices=["qasper", "nq"], help="hit@{1,3,5,10,20} from the stored gold ranks")
    ap.add_argument("--reach-association", choices=["qasper", "nq"],
                    help="Fisher's exact test of document reach at k=5 against rank-1 localisation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir
    if args.bootstrap:
        out = run_bootstrap(results_dir, args.bootstrap, args.n_boot, args.seed)
    elif args.audit_chunks:
        out = run_chunk_audit(args.audit_chunks, ROOT / args.raw, args.split, args.limit)
    elif args.model_metadata:
        out = run_model_metadata()
    elif args.hit_at_k:
        out = run_hit_at_k(results_dir, args.hit_at_k)
    elif args.reach_association:
        out = run_reach_association(results_dir, args.reach_association)
    else:
        ap.error("choose --bootstrap, --audit-chunks, --model-metadata, --hit-at-k or --reach-association")
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
