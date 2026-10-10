#!/usr/bin/env python
"""NQ within-document localisation under alternative evidence references (exploratory).

Not part of the submitted manuscript. The manuscript's NQ numbers take the long answer of
the first annotator who marked one as the only gold evidence (``_first_long_answer`` in
src/data/loaders/nq_parquet.py), although the NQ validation split is five-way annotated.
This script asks how far those numbers depend on that choice. It reads the study's stored
ranks in results/localisation/ and never writes there.

Step 1, ``--score``. Needs the raw NQ file, the stored chunk vectors under data/build and
the models in the local Hugging Face cache. Loads the same 300 questions with every
annotator's long answer, chunks exactly as the study did, ranks every chunk of each gold
document under the seven localisers, and stores, per question and localiser, the rank of
the first chunk that overlaps each annotator's long answer. Ranks under the first
annotator are checked against the study's stored ranks; any mismatch stops the run.

Step 2, ``--summarise``. Seconds, no data or models: rebuilds every reference scheme of
src/evaluation/references.py from the stored per-annotator ranks and reports hit@k, MRR,
rank distributions, chance levels, Wilson intervals, Holm-corrected exact McNemar tests,
document-cluster bootstrap intervals and single-annotator resampling.

    python scripts/nq_reference_sensitivity.py --score \\
        --raw data/raw/nq-validation-0.parquet \\
        --load-embeddings data/build/nq_val_300_vectors \\
        --minilm-chroma data/build/index_nq_val_300_fixed:exp_nq_validation \\
        --out-dir results/exploratory/nq_reference_sensitivity
    python scripts/nq_reference_sensitivity.py --summarise \\
        --out-dir results/exploratory/nq_reference_sensitivity
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from localisation_probe import random_expectations  # noqa: E402
from localisation_report import MODELS as MODEL_INFO  # noqa: E402
from localisation_report import holm  # noqa: E402
from localisation_robustness import MODELS, cluster_bootstrap, load_ranks  # noqa: E402

from src.evaluation.provenance import collect_provenance  # noqa: E402
from src.evaluation.references import (  # noqa: E402
    SCHEMES,
    draw_single,
    first_gold_rank,
    gold_chunks,
    non_null_count,
    reference_annotators,
    reference_rank,
    usable_annotations,
)
from src.evaluation.statistics import mcnemar_exact, wilson_proportion_ci  # noqa: E402

PER_QUESTION = "per_question.json"
SCORE_CONFIG = "score_config.json"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10), (">10", 11, 10 ** 9)]
DENSE = ("minilm", "bge", "e5", "mpnet")
CROSS = ("ce_msmarco_minilm", "bge_reranker_base")
DEFAULT_SEED = 20261010


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """One digest over every file under ``root``: relative path and content, sorted by path."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def cached_revision(repo_id: str) -> str | None:
    """The snapshot the local Hugging Face cache resolves for ``repo_id`` (refs/main)."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    ref = hub / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else None


# --------------------------------------------------------------------------- scoring

def score(args: argparse.Namespace) -> int:
    # Use the cached model revisions only; a download would change what is being compared.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from run_bm25_baseline import BM25, tokenize

    from src.data.corpus import chunk_documents
    from src.data.loaders import get_loader
    from src.rag.chunking import DocumentChunker
    from src.rag.embedders import EMBEDDERS

    t0 = time.time()
    # Captured before anything is written, so the dirty flag describes the code that ran.
    provenance = collect_provenance(script="scripts/nq_reference_sensitivity.py --score")
    frozen_dir = Path(args.frozen_dir)
    frozen = load_ranks(frozen_dir, "nq")
    frozen_probe = json.loads((frozen_dir / "within_document_nq.json").read_text(encoding="utf-8"))
    frozen_geometry = {r["question_id"]: (r["D"], r["m"])
                       for r in frozen_probe["within_document_rows"]["bm25"]}

    loader = get_loader("nq")
    loader.keep_annotator_spans = True
    loaded = loader.load(Path(args.raw), split="validation", limit=args.limit)
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks, stats = chunk_documents(loaded.documents, chunker)
    if stats.offset_mismatches:
        raise SystemExit(f"{stats.offset_mismatches} chunk offset mismatches")
    doc_chunks: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        doc_chunks.setdefault(chunk.doc_id, []).append(i)
    print(f"{len(loaded.questions)} questions, {len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)

    questions = []
    for q in loaded.questions:
        spans = [s for s in q.supporting_spans if s.doc_id in doc_chunks]
        if not q.is_answerable or not spans:
            continue
        doc_id = sorted({s.doc_id for s in spans})[0]
        idxs = doc_chunks[doc_id]
        ranges = [(chunks[i].start_char, chunks[i].end_char) for i in idxs]
        annotations = q.metadata["annotator_long_answers"]
        per_annotator = {a["annotator"]: sorted(gold_chunks(ranges, [(a["start_char"], a["end_char"])]))
                         for a in usable_annotations(annotations)}
        if any(not chunk_set for chunk_set in per_annotator.values()):
            raise SystemExit(f"{q.question_id}: an annotator's long answer overlaps no chunk")
        first = reference_annotators(annotations, "first")[0]
        first_span = next(a for a in annotations if a["annotator"] == first)
        if (first_span["start_char"], first_span["end_char"]) != (spans[0].start_char, spans[0].end_char):
            raise SystemExit(f"{q.question_id}: first annotator's span is not the loader's evidence span")
        questions.append({"q": q, "doc_id": doc_id, "idxs": idxs, "annotations": annotations,
                          "per_annotator": per_annotator, "first": first})

    # The question set, document sizes and first-annotator gold chunk counts must be the
    # study's own, or nothing below is comparable with the manuscript.
    ids = [x["q"].question_id for x in questions]
    if ids != list(frozen_geometry):
        raise SystemExit("question set or order differs from results/localisation/within_document_nq.json")
    geometry_mismatch = [x["q"].question_id for x in questions
                         if (len(x["idxs"]), len(x["per_annotator"][x["first"]]))
                         != frozen_geometry[x["q"].question_id]]
    if geometry_mismatch:
        raise SystemExit(f"D or m differs from the study for {len(geometry_mismatch)} questions")
    invariant = {x["q"].question_id for x in questions
                 if len({tuple(v) for v in x["per_annotator"].values()}) == 1}
    print(f"{len(questions)} questions match the study; {len(questions) - len(invariant)} have "
          f"annotators whose spans cover different chunks", flush=True)

    # ---- localisers: each returns within-document scores for one question ----
    bm25 = BM25([tokenize(c.text) for c in chunks])

    def bm25_scores(question: str, idxs: list[int]) -> np.ndarray:
        # The probe's scorer (corpus-level IDF, k1 = 1.5, b = 0.75), unchanged.
        terms = tokenize(question)
        out = np.zeros(len(idxs))
        for pos, i in enumerate(idxs):
            total = 0.0
            freq = bm25.freqs[i]
            for term in terms:
                idf, f = bm25.idf.get(term), freq.get(term)
                if idf is None or not f:
                    continue
                denom = f + bm25.k1 * (1 - bm25.b + bm25.b * bm25.lengths[i] / bm25.avg_len)
                total += idf * f * (bm25.k1 + 1) / denom
            out[pos] = total
        return out

    from sentence_transformers import CrossEncoder, SentenceTransformer

    dense = {}
    vector_inputs = {}
    for key in DENSE:
        spec = EMBEDDERS[key]
        if key == "minilm":
            import chromadb
            # rsplit: a Windows path has a drive-letter colon; a collection name has none.
            index_dir, collection = args.minilm_chroma.rsplit(":", 1)
            got = chromadb.PersistentClient(path=index_dir).get_collection(collection).get(
                include=["metadatas", "embeddings"])
            by_id = {i: (np.asarray(e, dtype=np.float32), m)
                     for i, e, m in zip(got["ids"], got["embeddings"], got["metadatas"], strict=True)}
            emb = np.zeros((len(chunks), len(next(iter(by_id.values()))[0])), dtype=np.float32)
            for i, chunk in enumerate(chunks):
                vector, meta = by_id[chunk.chunk_id]
                if (int(meta["start_char"]), int(meta["end_char"])) != (chunk.start_char, chunk.end_char):
                    raise SystemExit(f"stored MiniLM index disagrees with chunk {chunk.chunk_id}")
                emb[i] = vector
            vector_inputs[key] = {"chroma_collection": collection,
                                  "chroma_index_sha256_tree": sha256_tree(Path(index_dir))}
        else:
            npy = Path(args.load_embeddings) / f"{key}.npy"
            emb = np.load(npy).astype(np.float32)
            if emb.shape[0] != len(chunks):
                raise SystemExit(f"{npy}: {emb.shape[0]} rows for {len(chunks)} chunks")
            vector_inputs[key] = {"npy": npy.as_posix(), "sha256": sha256_file(npy)}
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        model = SentenceTransformer(spec.repo_id, device="cpu")
        qemb = model.encode([spec.query_prefix + x["q"].question for x in questions], batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        dense[key] = (emb, {x["q"].question_id: qemb[n] for n, x in enumerate(questions)})
        print(f"  {key}: {spec.repo_id}", flush=True)

    scope = dict(item.split("=", 1) for item in args.ce_scope.split(",") if item)
    ranks: dict[str, dict[str, dict[str, int]]] = {m: {} for m in MODELS}
    source: dict[str, dict[str, str]] = {m: {} for m in MODELS}
    runtime: dict[str, float] = {}

    def record(model: str, x: dict, scores: np.ndarray) -> None:
        order = list(np.argsort(-scores, kind="stable"))
        qid = x["q"].question_id
        ranks[model][qid] = {str(a): first_gold_rank(order, set(g)) for a, g in x["per_annotator"].items()}
        source[model][qid] = "recomputed"

    for model in ("bm25", *DENSE):
        t1 = time.time()
        for x in questions:
            if model == "bm25":
                scores = bm25_scores(x["q"].question, x["idxs"])
            else:
                emb, qmap = dense[model]
                scores = emb[x["idxs"]] @ qmap[x["q"].question_id]
            record(model, x, scores)
        runtime[model] = round(time.time() - t1, 1)

    for model in CROSS:
        t1 = time.time()
        encoder = CrossEncoder(MODEL_INFO[model][0], device="cpu", max_length=512)
        todo = questions if scope.get(model, "all") == "all" else [
            x for x in questions if x["q"].question_id not in invariant]
        for n, x in enumerate(todo, 1):
            pairs = [(x["q"].question, chunks[i].text) for i in x["idxs"]]
            record(model, x, np.asarray(encoder.predict(pairs, batch_size=32, show_progress_bar=False)))
            if n % 25 == 0:
                print(f"  {model}: {n}/{len(todo)}  {time.time() - t1:.0f}s", flush=True)
        # On a question whose annotators all cover the same chunks every reference is the
        # same gold set, so the study's stored rank is the rank under every scheme.
        for x in questions:
            qid = x["q"].question_id
            if qid not in ranks[model]:
                ranks[model][qid] = {str(a): frozen[model][qid] for a in x["per_annotator"]}
                source[model][qid] = "stored_rank_reference_invariant"
        runtime[model] = round(time.time() - t1, 1)

    # ---- the check everything rests on: the first annotator reproduces the study ----
    mismatches = {m: [qid for qid, src in source[m].items() if src == "recomputed"
                      and ranks[m][qid][str(next(x["first"] for x in questions if x["q"].question_id == qid))]
                      != frozen[m][qid]] for m in MODELS}
    recomputed = {m: sum(1 for s in source[m].values() if s == "recomputed") for m in MODELS}
    if any(mismatches.values()):
        raise SystemExit("first-annotator ranks differ from the study: "
                         + json.dumps({m: len(v) for m, v in mismatches.items()}))
    print("first-annotator ranks reproduce the study's stored ranks: "
          + ", ".join(f"{m} {recomputed[m]}/{len(questions)}" for m in MODELS), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_question = {
        "description": "Per question and localiser, the 1-based rank of the first chunk of the gold "
                       "document that overlaps each annotator's long answer. Character ranges index the "
                       "cleaned page text built by src/data/loaders/nq_parquet.py; no dataset text is stored.",
        "localisers": list(MODELS),
        "questions": [{
            "question_id": x["q"].question_id,
            "example_id": x["q"].metadata["example_id"],
            "doc_id": x["doc_id"],
            "D": len(x["idxs"]),
            "annotations": [{**a, **({"gold_chunks": x["per_annotator"][a["annotator"]]}
                                     if a["status"] == "ok" else {})} for a in x["annotations"]],
            "ranks": {m: ranks[m][x["q"].question_id] for m in MODELS},
            "rank_source": {m: source[m][x["q"].question_id] for m in MODELS},
        } for x in questions],
    }
    (out_dir / PER_QUESTION).write_text(json.dumps(per_question, indent=1) + "\n", encoding="utf-8")

    models = {m: {"identity": MODEL_INFO[m][0], "family": MODEL_INFO[m][1],
                  "cached_revision": cached_revision(MODEL_INFO[m][0]) if m != "bm25" else None}
              for m in MODELS}
    config = {
        "status": "exploratory; not part of the submitted manuscript",
        "raw_file": {"path": Path(args.raw).as_posix(), "sha256": sha256_file(Path(args.raw))},
        "limit": args.limit, "split": "validation",
        "chunking": {"size": args.chunk_size, "overlap": args.chunk_overlap, "n_chunks": len(chunks)},
        "localisers": models, "vector_inputs": vector_inputs,
        "cross_encoder_scope": {m: scope.get(m, "all") for m in CROSS},
        "checks": {
            "question_ids_and_order_match_study": True,
            "D_and_first_annotator_m_match_study": True,
            "first_annotator_ranks_match_study": {m: f"{recomputed[m]} recomputed, 0 mismatches" for m in MODELS},
            "questions_where_annotators_cover_different_chunks": len(questions) - len(invariant),
        },
        "runtime_seconds": {**runtime, "total": round(time.time() - t0, 1)},
        "provenance": provenance,
    }
    (out_dir / SCORE_CONFIG).write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / PER_QUESTION} and {out_dir / SCORE_CONFIG}", flush=True)
    return 0


# --------------------------------------------------------------------------- summarising

def annotator_ranks(question: dict, model: str) -> dict[int, int]:
    return {int(a): r for a, r in question["ranks"][model].items()}


def scheme_rows(questions: list[dict], models: list[str], scheme: str) -> list[dict]:
    """One row per question that has a gold long answer under ``scheme``."""
    rows = []
    for q in questions:
        annotators = reference_annotators(q["annotations"], scheme)
        if annotators is None:
            continue
        rows.append(_row(q, models, annotators))
    return rows


def _row(question: dict, models: list[str], annotators: list[int]) -> dict:
    by_annotator = {a["annotator"]: a for a in question["annotations"]}
    gold = set().union(*(set(by_annotator[a]["gold_chunks"]) for a in annotators))
    return {"question_id": question["question_id"], "doc_id": question["doc_id"],
            "D": question["D"], "m": len(gold),
            "rank": {m: reference_rank(annotator_ranks(question, m), annotators) for m in models}}


def reference_invariant(question: dict) -> bool:
    """True when every annotator's span covers the same chunks, so every scheme agrees."""
    return len({tuple(a["gold_chunks"]) for a in usable_annotations(question["annotations"])}) == 1


def model_metrics(rows: list[dict], model: str) -> dict:
    ranks = np.array([r["rank"][model] for r in rows])
    n = len(ranks)
    h1, h5 = int((ranks <= 1).sum()), int((ranks <= 5).sum())
    chance = [random_expectations(r["D"], r["m"]) for r in rows]
    chance1 = float(np.mean([c["hit@1"] for c in chance]))
    w1, w5 = wilson_proportion_ci(h1, n), wilson_proportion_ci(h5, n)
    return {
        "n": n,
        "hit@1": round(h1 / n, 4), "hit@1_wilson95": [round(w1.lower, 4), round(w1.upper, 4)],
        "hit@5": round(h5 / n, 4), "hit@5_wilson95": [round(w5.lower, 4), round(w5.upper, 4)],
        "mrr": round(float(np.mean(1.0 / ranks)), 4),
        "median_rank": float(np.median(ranks)),
        "rank_buckets": {name: round(float(((ranks >= lo) & (ranks <= hi)).mean()), 4)
                         for name, lo, hi in BUCKETS},
        "chance_hit@1": round(chance1, 4),
        "chance_hit@5": round(float(np.mean([c["hit@5"] for c in chance])), 4),
        "lift@1_over_chance": round((h1 / n) / chance1, 3) if chance1 else None,
    }


def pairwise(rows: list[dict], models: list[str], n_boot: int, seed: int) -> dict:
    """Exact McNemar on hit@1 with Holm over all pairs, as in localisation_report.py, plus a
    document-cluster bootstrap interval for each hit@1 difference."""
    out, raw = {}, []
    pairs = list(itertools.combinations(models, 2))
    for a, b in pairs:
        a_only = sum(1 for r in rows if r["rank"][a] == 1 and r["rank"][b] != 1)
        b_only = sum(1 for r in rows if r["rank"][b] == 1 and r["rank"][a] != 1)
        test = mcnemar_exact(a_only, b_only)
        raw.append(test.p_value if test.p_value is not None else 1.0)
        clusters: dict[str, list[int]] = {}
        for r in rows:
            clusters.setdefault(r["doc_id"], []).append(int(r["rank"][a] == 1) - int(r["rank"][b] == 1))
        boot = cluster_bootstrap(list(clusters.values()), n_boot=n_boot, seed=seed)
        out[f"{a}_vs_{b}"] = {
            "hit@1_diff": round((sum(r["rank"][a] == 1 for r in rows) - sum(r["rank"][b] == 1 for r in rows))
                                / len(rows), 4),
            "hit@1_only_a": a_only, "hit@1_only_b": b_only,
            "mcnemar_exact_p": None if test.p_value is None else round(test.p_value, 4),
            "cluster_bootstrap_ci95": boot["ci95"], "cluster_bootstrap_excludes_zero": boot["excludes_zero"],
        }
    for (a, b), adjusted in zip(pairs, holm(raw), strict=True):
        out[f"{a}_vs_{b}"]["mcnemar_holm_adjusted_p"] = round(adjusted, 4)
    return out


def _holm_significant(rows: list[dict], models: list[str]) -> dict[str, int]:
    """Per pair: +1 if A beats B at Holm-adjusted p < 0.05, -1 if B beats A, 0 otherwise."""
    pairs = list(itertools.combinations(models, 2))
    raw, signs = [], []
    for a, b in pairs:
        a_only = sum(1 for r in rows if r["rank"][a] == 1 and r["rank"][b] != 1)
        b_only = sum(1 for r in rows if r["rank"][b] == 1 and r["rank"][a] != 1)
        test = mcnemar_exact(a_only, b_only)
        raw.append(test.p_value if test.p_value is not None else 1.0)
        signs.append(1 if a_only > b_only else -1 if b_only > a_only else 0)
    return {f"{a}_vs_{b}": (sign if adjusted < 0.05 else 0)
            for (a, b), sign, adjusted in zip(pairs, signs, holm(raw), strict=True)}


def single_annotator_resampling(questions: list[dict], models: list[str], n_draws: int, seed: int,
                                reference_signs: dict[str, int]) -> dict:
    """Draw one usable annotator per question, ``n_draws`` times, and track the scores."""
    rng = random.Random(seed)
    hits = {m: [] for m in models}
    best = {m: 0.0 for m in models}
    same_conclusion = {pair: 0 for pair in reference_signs}
    for _ in range(n_draws):
        rows = [_row(q, models, draw_single(q["annotations"], rng)) for q in questions]
        h1 = {m: sum(r["rank"][m] == 1 for r in rows) / len(rows) for m in models}
        for m in models:
            hits[m].append(h1[m])
        top = max(h1.values())
        winners = [m for m in models if h1[m] == top]
        for m in winners:
            best[m] += 1 / len(winners)
        for pair, sign in _holm_significant(rows, models).items():
            same_conclusion[pair] += sign == reference_signs[pair]
    return {
        "n_draws": n_draws, "seed": seed,
        "hit@1": {m: {"mean": round(float(np.mean(v)), 4),
                      "p2.5": round(float(np.percentile(v, 2.5)), 4),
                      "p97.5": round(float(np.percentile(v, 97.5)), 4)} for m, v in hits.items()},
        "share_of_draws_best_at_hit@1": {m: round(v / n_draws, 4) for m, v in best.items()},
        "share_of_draws_with_same_holm_conclusion_as_first": {
            pair: round(count / n_draws, 4) for pair, count in same_conclusion.items()},
    }


def summarise_questions(questions: list[dict], models: list[str], n_boot: int = 4000,
                        n_draws: int = 2000, seed: int = DEFAULT_SEED) -> dict:
    references = {scheme: scheme_rows(questions, models, scheme) for scheme in SCHEMES}
    invariant_ids = {q["question_id"] for q in questions if reference_invariant(q)}
    references["agreement_subset"] = [r for r in references["first"] if r["question_id"] in invariant_ids]
    summary = {"schemes": {}, "annotation_profile": annotation_profile(questions)}
    for name, rows in references.items():
        summary["schemes"][name] = {
            "n_questions": len(rows),
            "models": {m: model_metrics(rows, m) for m in models},
            "pairwise": pairwise(rows, models, n_boot=n_boot, seed=0),
            "best_at_hit@1": max(models, key=lambda m, rows=rows: sum(r["rank"][m] == 1 for r in rows)),
        }
    first = {r["question_id"]: r for r in references["first"]}
    summary["union_vs_first"] = {m: {
        "questions_with_better_rank": sum(1 for r in references["union"] if r["rank"][m] < first[r["question_id"]]["rank"][m]),
        "hit@1_gained": sum(1 for r in references["union"]
                            if r["rank"][m] == 1 and first[r["question_id"]]["rank"][m] != 1),
        "hit@5_gained": sum(1 for r in references["union"]
                            if r["rank"][m] <= 5 and first[r["question_id"]]["rank"][m] > 5),
    } for m in models}
    signs = _holm_significant(references["first"], models)
    summary["holm_significant_pairs"] = {name: {pair: sign for pair, sign in _holm_significant(rows, models).items() if sign}
                                         for name, rows in references.items()}
    summary["single_annotator_resampling"] = single_annotator_resampling(questions, models, n_draws, seed, signs)
    return summary


def span_relation(first: tuple[int, int], other: tuple[int, int]) -> str:
    """How another annotator's long answer relates to the first annotator's."""
    (a0, a1), (b0, b1) = first, other
    if (a0, a1) == (b0, b1):
        return "identical"
    if min(a1, b1) - max(a0, b0) <= 0:
        return "disjoint"
    if a0 <= b0 and b1 <= a1:
        return "inside_first"
    if b0 <= a0 and a1 <= b1:
        return "contains_first"
    return "partial_overlap"


#: For a question-level label, the most consequential relation wins.
_RELATION_ORDER = ("disjoint", "partial_overlap", "contains_first", "inside_first", "identical")


def annotation_profile(questions: list[dict]) -> dict:
    pair_relations: dict[str, int] = {}
    question_relations: dict[str, int] = {}
    for q in questions:
        usable = usable_annotations(q["annotations"])
        first_id = reference_annotators(q["annotations"], "first")[0]
        first = next((a["start_char"], a["end_char"]) for a in usable if a["annotator"] == first_id)
        relations = [span_relation(first, (a["start_char"], a["end_char"]))
                     for a in usable if a["annotator"] != first_id]
        for relation in relations:
            pair_relations[relation] = pair_relations.get(relation, 0) + 1
        label = next((r for r in _RELATION_ORDER if r in relations), "single_annotator")
        question_relations[label] = question_relations.get(label, 0) + 1
    nonnull = [non_null_count(q["annotations"]) for q in questions]
    usable = [len(usable_annotations(q["annotations"])) for q in questions]
    distinct_spans = [len({(a["start_char"], a["end_char"]) for a in usable_annotations(q["annotations"])})
                      for q in questions]
    return {
        "n_questions": len(questions),
        "annotators_marking_a_long_answer": {str(k): nonnull.count(k) for k in sorted(set(nonnull))},
        "annotators_with_a_usable_span": {str(k): usable.count(k) for k in sorted(set(usable))},
        "questions_with_more_than_one_distinct_span": sum(1 for d in distinct_spans if d > 1),
        "questions_where_spans_cover_different_chunks": sum(1 for q in questions if not reference_invariant(q)),
        "questions_below_the_official_threshold": sum(1 for k in nonnull if k < 2),
        "other_annotator_vs_first_span_pairs": dict(sorted(pair_relations.items())),
        "questions_by_most_consequential_relation": dict(sorted(question_relations.items())),
    }


def summarise(args: argparse.Namespace) -> int:
    provenance = collect_provenance(script="scripts/nq_reference_sensitivity.py --summarise")
    out_dir = Path(args.out_dir)
    per_question = out_dir / PER_QUESTION
    data = json.loads(per_question.read_text(encoding="utf-8"))
    models = [m for m in MODELS if m in data["localisers"]]
    summary = {
        "status": "exploratory; not part of the submitted manuscript",
        "schemes_defined_in": "src/evaluation/references.py",
        "input": {"path": per_question.as_posix(), "sha256": sha256_file(per_question)},
        "settings": {"n_boot": args.n_boot, "bootstrap_seed": 0, "n_draws": args.n_draws,
                     "resampling_seed": args.seed},
        **summarise_questions(data["questions"], models, args.n_boot, args.n_draws, args.seed),
        "manuscript_check": manuscript_check(data["questions"], models, Path(args.frozen_dir)),
        "provenance": provenance,
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out_dir / SUMMARY_MD).write_text(render_markdown(summary, models), encoding="utf-8")
    print(f"wrote {out_dir / SUMMARY_JSON} and {out_dir / SUMMARY_MD}")
    return 0


def manuscript_check(questions: list[dict], models: list[str], frozen_dir: Path) -> dict:
    """Under the first annotator the numbers must be the manuscript's, to the last digit."""
    stored = json.loads((frozen_dir / "summary_nq.json").read_text(encoding="utf-8"))
    rows = scheme_rows(questions, models, "first")
    ours = {m: model_metrics(rows, m) for m in models}
    tests = pairwise(rows, models, n_boot=10, seed=0)
    metric_diffs = {f"{m}.{k}": [ours[m][k], stored["models"][m][k]] for m in models
                    for k in ("hit@1", "hit@5", "mrr") if ours[m][k] != stored["models"][m][k]}
    test_diffs = {f"{pair}.{k}": [tests[pair][k], stored["pairwise"][pair][k]] for pair in tests
                  for k in ("mcnemar_exact_p", "mcnemar_holm_adjusted_p")
                  if tests[pair][k] != stored["pairwise"][pair][k]}
    return {"compared_with": (frozen_dir / "summary_nq.json").as_posix(),
            "metrics_identical": not metric_diffs, "pairwise_tests_identical": not test_diffs,
            "differences": {**metric_diffs, **test_diffs}}


def render_markdown(summary: dict, models: list[str]) -> str:
    names = {"first": "first annotator (manuscript)", "union": "union of annotators",
             "official": "NQ official (>= 2 annotators, union)", "plurality": "plurality span",
             "agreement_subset": "questions where annotators agree"}
    lines = ["# NQ localisation under alternative evidence references", "",
             "**Exploratory. Not part of the submitted manuscript.** Generated by "
             "`scripts/nq_reference_sensitivity.py --summarise` from `per_question.json`.", "",
             "## hit@1 (hit@5) by reference", "",
             "| Localiser | " + " | ".join(f"{names[s]} (n={summary['schemes'][s]['n_questions']})"
                                         for s in names) + " |",
             "|---|" + "---:|" * len(names)]
    for m in models:
        cells = [f"{summary['schemes'][s]['models'][m]['hit@1']:.3f} ({summary['schemes'][s]['models'][m]['hit@5']:.3f})"
                 for s in names]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "| | " + " | ".join(names[s] for s in names) + " |", "|---|" + "---:|" * len(names),
              "| chance hit@1 (random order) | " + " | ".join(
                  f"{summary['schemes'][s]['models'][models[0]]['chance_hit@1']:.3f}" for s in names) + " |",
              "| best at hit@1 | " + " | ".join(summary["schemes"][s]["best_at_hit@1"] for s in names) + " |",
              "| Holm-significant pairs | " + " | ".join(
                  str(len(summary["holm_significant_pairs"][s])) for s in names) + " |", "",
              "## Holm-significant pairs (hit@1, exact McNemar, alpha 0.05)", ""]
    for s in names:
        pairs = summary["holm_significant_pairs"][s]
        rendered = ", ".join(f"{p.split('_vs_')[0] if v > 0 else p.split('_vs_')[1]} > "
                             f"{p.split('_vs_')[1] if v > 0 else p.split('_vs_')[0]}" for p, v in pairs.items())
        lines.append(f"- **{names[s]}**: {rendered or 'none'}")
    res = summary["single_annotator_resampling"]
    lines += ["", f"## One annotator drawn at random per question ({res['n_draws']} draws, seed {res['seed']})", "",
              "| Localiser | mean hit@1 | 95% of draws | share of draws best |", "|---|---:|---:|---:|"]
    for m in models:
        h = res["hit@1"][m]
        lines.append(f"| {m} | {h['mean']:.3f} | {h['p2.5']:.3f}–{h['p97.5']:.3f} | "
                     f"{res['share_of_draws_best_at_hit@1'][m]:.3f} |")
    check = summary["manuscript_check"]
    lines += ["", "## Check against the manuscript", "",
              f"Under the first annotator, hit@1, hit@5 and MRR identical to `summary_nq.json`: "
              f"{check['metrics_identical']}; McNemar and Holm p-values identical: {check['pairwise_tests_identical']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--score", action="store_true")
    step.add_argument("--summarise", action="store_true")
    ap.add_argument("--raw", default="data/raw/nq-validation-0.parquet")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--load-embeddings", default="data/build/nq_val_300_vectors")
    ap.add_argument("--minilm-chroma", default="data/build/index_nq_val_300_fixed:exp_nq_validation")
    ap.add_argument("--ce-scope", default="ce_msmarco_minilm=all,bge_reranker_base=all",
                    help="per cross-encoder, 'all' or 'affected' (only questions whose annotators "
                         "cover different chunks; elsewhere every scheme gives the stored rank)")
    ap.add_argument("--frozen-dir", default="results/localisation")
    ap.add_argument("--out-dir", default="results/exploratory/nq_reference_sensitivity")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    if Path(args.out_dir).resolve().is_relative_to((REPO / "results" / "localisation").resolve()):
        raise SystemExit("refusing to write into results/localisation/, which holds the manuscript's results")
    return score(args) if args.score else summarise(args)


if __name__ == "__main__":
    raise SystemExit(main())
