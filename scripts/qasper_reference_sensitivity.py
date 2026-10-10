#!/usr/bin/env python
"""QASPER within-document localisation under alternative evidence references (exploratory).

Not part of the submitted manuscript. The manuscript's QASPER numbers count every
annotator's evidence paragraphs as gold (``_evidence_spans`` in src/data/loaders/qasper.py
merges them), while its NQ numbers count only the first annotator's long answer. This
script asks how far the QASPER numbers depend on that choice; with
results/exploratory/nq_reference_sensitivity it also puts the two corpora on one footing
(``scripts/reference_sensitivity_claims.py``). It reads the study's stored ranks in
results/localisation/ and never writes there.

Step 1, ``--score``. Needs the raw QASPER dev file, copies of the four stored QASPER Chroma
indexes and the models in the local Hugging Face cache. Loads the same 290 questions with
each annotator's evidence, chunks exactly as the study did, ranks every chunk of each gold
document under the seven localisers, and stores, per question and localiser, the rank of
the first chunk that overlaps each distinct evidence paragraph. Ranks under the union of
annotators are checked against the study's stored ranks; any mismatch stops the run. The
cross-encoders run only on questions where some reference differs from the union
(``--ce-scope``): everywhere else every scheme's gold chunks are the union's, so the stored
rank is exact.

Step 2, ``--summarise``. Seconds to minutes, no data or models: rebuilds the evidence-set
schemes of src/evaluation/references.py from the stored ranks and reports hit@k, MRR, rank
distributions, chance levels, Wilson intervals, Holm-corrected exact McNemar tests,
document-cluster bootstrap intervals and single-annotator resampling.

    python scripts/qasper_reference_sensitivity.py --score \\
        --raw data/raw/qasper-dev-v0.3.json \\
        --chroma-map "minilm=COPY/index_qasper_dev_300_emb_minilm:exp_qasper_dev,..." \\
        --ce-scope ce_msmarco_minilm=affected,bge_reranker_base=affected \\
        --out-dir results/exploratory/qasper_reference_sensitivity
    python scripts/qasper_reference_sensitivity.py --summarise \\
        --out-dir results/exploratory/qasper_reference_sensitivity
"""
from __future__ import annotations

import argparse
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

from localisation_report import MODELS as MODEL_INFO  # noqa: E402
from localisation_robustness import MODELS, load_ranks  # noqa: E402
from nq_reference_sensitivity import (  # noqa: E402
    CROSS,
    DEFAULT_SEED,
    DENSE,
    _holm_significant,
    cached_revision,
    model_metrics,
    pairwise,
    sha256_file,
    sha256_lf,
    sha256_tree,
    write_lf,
)

from src.evaluation.provenance import collect_provenance  # noqa: E402
from src.evaluation.references import (  # noqa: E402
    EVIDENCE_SET_SCHEMES,
    draw_single_evidence,
    evidence_spans,
    first_gold_rank,
    gold_chunks,
    reference_spans,
    usable_annotations,
)

PER_QUESTION = "per_question.json"
SCORE_CONFIG = "score_config.json"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
NEURAL = ("minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base")
DEFAULT_CHROMA_MAP = ",".join(
    f"{key}=data/build/index_qasper_dev_300_emb_{key}:exp_qasper_dev" for key in ("minilm", "mpnet", "bge", "e5"))


# --------------------------------------------------------------------------- references

def span_gold(question: dict, spans: list[tuple[int, int]]) -> frozenset[int]:
    """The chunks of the gold document that overlap any of ``spans``."""
    by_span = {(s["start_char"], s["end_char"]): s["gold_chunks"] for s in question["spans"]}
    return frozenset(chunk for span in spans for chunk in by_span[span])


def candidate_references(annotations: list[dict]) -> list[list[tuple[int, int]]]:
    """Every reference any scheme can produce: the four schemes and each single annotator."""
    out = [reference_spans(annotations, scheme) for scheme in EVIDENCE_SET_SCHEMES]
    out += [sorted(evidence_spans(a)) for a in usable_annotations(annotations)]
    return [spans for spans in out if spans is not None]


def reference_invariant(question: dict) -> bool:
    """True when every scheme and every single annotator give the union's gold chunks."""
    union = span_gold(question, reference_spans(question["annotations"], "union"))
    return all(span_gold(question, spans) == union for spans in candidate_references(question["annotations"]))


def question_rank(question: dict, model: str, spans: list[tuple[int, int]]) -> int:
    """Rank of the first gold chunk when ``spans`` are the reference."""
    ranks = question["ranks"][model]
    if span_gold(question, spans) == span_gold(question, reference_spans(question["annotations"], "union")):
        return ranks["union"]
    if ranks["spans"] is None:
        raise ValueError(f"{question['question_id']}: {model} has no per-paragraph ranks for a "
                         "reference that differs from the union")
    position = {(s["start_char"], s["end_char"]): i for i, s in enumerate(question["spans"])}
    return min(ranks["spans"][position[span]] for span in spans)


def _row(question: dict, models: list[str], spans: list[tuple[int, int]]) -> dict:
    return {"question_id": question["question_id"], "doc_id": question["doc_id"],
            "D": question["D"], "m": len(span_gold(question, spans)), "n_spans": len(spans),
            "rank": {m: question_rank(question, m, spans) for m in models}}


def scheme_rows(questions: list[dict], models: list[str], scheme: str) -> list[dict]:
    """One row per question that has evidence under ``scheme``."""
    rows = []
    for q in questions:
        spans = reference_spans(q["annotations"], scheme)
        if spans is not None:
            rows.append(_row(q, models, spans))
    return rows


# --------------------------------------------------------------------------- scoring

def parse_chroma_map(text: str) -> dict[str, tuple[str, str]]:
    """``key=index_dir:collection,...``; split on the last colon, so Windows paths work."""
    out = {}
    for item in (x for x in text.split(",") if x):
        key, rest = item.split("=", 1)
        index_dir, collection = rest.rsplit(":", 1)
        out[key] = (index_dir, collection)
    return out


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
    provenance = collect_provenance(script="scripts/qasper_reference_sensitivity.py --score")
    frozen_dir = Path(args.frozen_dir)
    frozen = load_ranks(frozen_dir, "qasper")
    frozen_probe = json.loads((frozen_dir / "within_document_qasper.json").read_text(encoding="utf-8"))
    frozen_geometry = {r["question_id"]: (r["D"], r["m"])
                       for r in frozen_probe["within_document_rows"]["bm25"]}

    loader = get_loader("qasper")
    loader.keep_annotator_spans = True
    loaded = loader.load(Path(args.raw), split="dev", limit=args.limit)
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
        loader_spans = [s for s in q.supporting_spans if s.doc_id in doc_chunks]
        if not q.is_answerable or not loader_spans:
            continue
        doc_id = sorted({s.doc_id for s in loader_spans})[0]
        idxs = doc_chunks[doc_id]
        ranges = [(chunks[i].start_char, chunks[i].end_char) for i in idxs]
        annotations = q.metadata["annotator_evidence"]
        union = reference_spans(annotations, "union")
        if set(union) != {(s.start_char, s.end_char) for s in loader_spans}:
            raise SystemExit(f"{q.question_id}: the annotators' evidence is not the loader's evidence")
        spans = [{"start_char": s, "end_char": e,
                  "annotators": [a["annotator"] for a in usable_annotations(annotations)
                                 if (s, e) in evidence_spans(a)],
                  "gold_chunks": sorted(gold_chunks(ranges, [(s, e)]))} for s, e in union]
        if any(not s["gold_chunks"] for s in spans):
            raise SystemExit(f"{q.question_id}: an evidence paragraph overlaps no chunk")
        record = {"q": q, "doc_id": doc_id, "idxs": idxs,
                  "question": {"question_id": q.question_id, "annotations": annotations, "spans": spans}}
        record["union_gold"] = set(span_gold(record["question"], union))
        record["invariant"] = reference_invariant(record["question"])
        questions.append(record)

    # The question set, document sizes and union gold chunk counts must be the study's own,
    # or nothing below is comparable with the manuscript.
    ids = [x["q"].question_id for x in questions]
    if ids != list(frozen_geometry):
        raise SystemExit("question set or order differs from results/localisation/within_document_qasper.json")
    geometry_mismatch = [x["q"].question_id for x in questions
                         if (len(x["idxs"]), len(x["union_gold"])) != frozen_geometry[x["q"].question_id]]
    if geometry_mismatch:
        raise SystemExit(f"D or m differs from the study for {len(geometry_mismatch)} questions")
    affected = sum(1 for x in questions if not x["invariant"])
    print(f"{len(questions)} questions match the study; on {affected} some reference differs "
          f"from the union", flush=True)

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

    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    chroma_map = parse_chroma_map(args.chroma_map)
    dense = {}
    vector_inputs = {}
    for key in DENSE:
        spec = EMBEDDERS[key]
        index_dir, collection = chroma_map[key]
        # Hash before opening: Chroma rewrites part of chroma.sqlite3 when it opens an index,
        # so point --chroma-map at copies rather than at data/build itself.
        index_hash = sha256_tree(Path(index_dir))
        got = chromadb.PersistentClient(path=index_dir).get_collection(collection).get(
            include=["metadatas", "embeddings"])
        by_id = {i: (np.asarray(e, dtype=np.float32), m)
                 for i, e, m in zip(got["ids"], got["embeddings"], got["metadatas"], strict=True)}
        emb = np.zeros((len(chunks), len(next(iter(by_id.values()))[0])), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            vector, meta = by_id[chunk.chunk_id]
            if (int(meta["start_char"]), int(meta["end_char"])) != (chunk.start_char, chunk.end_char):
                raise SystemExit(f"stored {key} index disagrees with chunk {chunk.chunk_id}")
            emb[i] = vector
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        vector_inputs[key] = {"chroma_index": Path(index_dir).name, "chroma_collection": collection,
                              "chroma_index_sha256_tree_before_opening": index_hash}
        model = SentenceTransformer(spec.repo_id, device="cpu")
        qemb = model.encode([spec.query_prefix + x["q"].question for x in questions], batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        dense[key] = (emb, {x["q"].question_id: qemb[n] for n, x in enumerate(questions)})
        print(f"  {key}: {spec.repo_id}", flush=True)

    scope = dict(item.split("=", 1) for item in args.ce_scope.split(",") if item)
    ranks: dict[str, dict[str, dict]] = {m: {} for m in MODELS}
    source: dict[str, dict[str, str]] = {m: {} for m in MODELS}
    runtime: dict[str, float] = {}

    def record(model: str, x: dict, scores: np.ndarray) -> None:
        order = list(np.argsort(-scores, kind="stable"))
        span_ranks = [first_gold_rank(order, set(s["gold_chunks"])) for s in x["question"]["spans"]]
        union_rank = first_gold_rank(order, x["union_gold"])
        if union_rank != min(span_ranks):
            raise SystemExit(f"{x['q'].question_id}: union rank is not the best paragraph rank")
        ranks[model][x["q"].question_id] = {"union": union_rank, "spans": span_ranks}
        source[model][x["q"].question_id] = "recomputed"

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
        todo = questions if scope.get(model, "all") == "all" else [x for x in questions if not x["invariant"]]
        for n, x in enumerate(todo, 1):
            pairs = [(x["q"].question, chunks[i].text) for i in x["idxs"]]
            record(model, x, np.asarray(encoder.predict(pairs, batch_size=32, show_progress_bar=False)))
            if n % 25 == 0:
                print(f"  {model}: {n}/{len(todo)}  {time.time() - t1:.0f}s", flush=True)
        # Where every reference is the union's gold set, the study's stored rank is the
        # rank under every scheme.
        for x in questions:
            qid = x["q"].question_id
            if qid not in ranks[model]:
                ranks[model][qid] = {"union": frozen[model][qid], "spans": None}
                source[model][qid] = "stored_rank_reference_invariant"
        runtime[model] = round(time.time() - t1, 1)

    # ---- the check everything rests on: the union reproduces the study ----
    mismatches = {m: [qid for qid, src in source[m].items()
                      if src == "recomputed" and ranks[m][qid]["union"] != frozen[m][qid]] for m in MODELS}
    recomputed = {m: sum(1 for s in source[m].values() if s == "recomputed") for m in MODELS}
    if any(mismatches.values()):
        raise SystemExit("union ranks differ from the study: "
                         + json.dumps({m: len(v) for m, v in mismatches.items()}))
    print("union ranks reproduce the study's stored ranks: "
          + ", ".join(f"{m} {recomputed[m]}/{len(questions)}" for m in MODELS), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_question = {
        "description": "Per question and localiser, the 1-based rank of the first chunk of the gold "
                       "document that overlaps the union of the annotators' evidence ('union') and each "
                       "distinct evidence paragraph ('spans', aligned with the question's 'spans'; null "
                       "where a cross-encoder was not rerun because every reference is the union's gold). "
                       "Character ranges index the paper body built by src/data/loaders/qasper.py; no "
                       "dataset text is stored.",
        "localisers": list(MODELS),
        "questions": [{
            "question_id": x["q"].question_id,
            "paper_id": x["q"].metadata["paper_id"],
            "doc_id": x["doc_id"],
            "D": len(x["idxs"]),
            "annotations": x["question"]["annotations"],
            "spans": x["question"]["spans"],
            "reference_invariant": x["invariant"],
            "ranks": {m: ranks[m][x["q"].question_id] for m in MODELS},
            "rank_source": {m: source[m][x["q"].question_id] for m in MODELS},
        } for x in questions],
    }
    write_lf(out_dir / PER_QUESTION, json.dumps(per_question, indent=1) + "\n")

    models = {m: {"identity": MODEL_INFO[m][0], "family": MODEL_INFO[m][1],
                  "cached_revision": cached_revision(MODEL_INFO[m][0]) if m != "bm25" else None}
              for m in MODELS}
    config = {
        "status": "exploratory; not part of the submitted manuscript",
        "raw_file": {"path": Path(args.raw).as_posix(), "sha256": sha256_file(Path(args.raw))},
        "limit": args.limit, "split": "dev",
        "chunking": {"size": args.chunk_size, "overlap": args.chunk_overlap, "n_chunks": len(chunks)},
        "localisers": models, "vector_inputs": vector_inputs,
        "cross_encoder_scope": {m: scope.get(m, "all") for m in CROSS},
        "checks": {
            "question_ids_and_order_match_study": True,
            "D_and_union_m_match_study": True,
            "union_of_annotator_evidence_equals_loader_evidence": True,
            "union_ranks_match_study": {m: f"{recomputed[m]} recomputed, 0 mismatches" for m in MODELS},
            "questions_where_some_reference_differs_from_union": affected,
        },
        "runtime_seconds": {**runtime, "total": round(time.time() - t0, 1)},
        "provenance": provenance,
    }
    write_lf(out_dir / SCORE_CONFIG, json.dumps(config, indent=1) + "\n")
    print(f"wrote {out_dir / PER_QUESTION} and {out_dir / SCORE_CONFIG}", flush=True)
    return 0


# --------------------------------------------------------------------------- summarising

def scheme_summary(rows: list[dict], models: list[str], n_boot: int) -> dict:
    hit1 = {m: sum(r["rank"][m] == 1 for r in rows) for m in models}
    neural = [m for m in models if m in NEURAL]
    return {
        "n_questions": len(rows),
        "mean_gold_spans_per_question": round(float(np.mean([r["n_spans"] for r in rows])), 3),
        "mean_gold_chunks_per_question": round(float(np.mean([r["m"] for r in rows])), 3),
        "models": {m: model_metrics(rows, m) for m in models},
        "pairwise": pairwise(rows, models, n_boot=n_boot, seed=0),
        "best_at_hit@1": max(models, key=lambda m: hit1[m]),
        "worst_neural_at_hit@1": min(neural, key=lambda m: hit1[m]),
        # the manuscript's union analysis: some localiser ranks the gold chunk first
        "any_localiser_hit@1": round(sum(any(r["rank"][m] == 1 for m in models) for r in rows) / len(rows), 4),
        "no_localiser_hit@1": round(sum(all(r["rank"][m] != 1 for m in models) for r in rows) / len(rows), 4),
        "gold_outside_top5_for_every_localiser": round(
            sum(all(r["rank"][m] > 5 for m in models) for r in rows) / len(rows), 4),
    }


def single_annotator_resampling(questions: list[dict], models: list[str], n_draws: int, seed: int,
                                reference_signs: dict[str, int]) -> dict:
    """Draw one usable annotator per question, ``n_draws`` times, and track the scores."""
    rng = random.Random(seed)
    # An annotator's row is fixed, so each is built once, keyed by its evidence; a draw
    # only picks rows.
    options = [{tuple(sorted(evidence_spans(a))): _row(q, models, sorted(evidence_spans(a)))
                for a in usable_annotations(q["annotations"])} for q in questions]
    neural = [m for m in models if m in NEURAL]
    hits = {m: [] for m in models}
    best = {m: 0.0 for m in models}
    worst_neural = {m: 0.0 for m in neural}
    same_conclusion = {pair: 0 for pair in reference_signs}
    for _ in range(n_draws):
        rows = [opts[tuple(draw_single_evidence(q["annotations"], rng))]
                for q, opts in zip(questions, options, strict=True)]
        h1 = {m: sum(r["rank"][m] == 1 for r in rows) / len(rows) for m in models}
        for m in models:
            hits[m].append(h1[m])
        top = max(h1.values())
        winners = [m for m in models if h1[m] == top]
        for m in winners:
            best[m] += 1 / len(winners)
        bottom = min(h1[m] for m in neural)
        losers = [m for m in neural if h1[m] == bottom]
        for m in losers:
            worst_neural[m] += 1 / len(losers)
        for pair, sign in _holm_significant(rows, models).items():
            same_conclusion[pair] += sign == reference_signs[pair]
    return {
        "n_draws": n_draws, "seed": seed,
        "hit@1": {m: {"mean": round(float(np.mean(v)), 4),
                      "p2.5": round(float(np.percentile(v, 2.5)), 4),
                      "p97.5": round(float(np.percentile(v, 97.5)), 4)} for m, v in hits.items()},
        "share_of_draws_best_at_hit@1": {m: round(v / n_draws, 4) for m, v in best.items()},
        "share_of_draws_worst_neural_at_hit@1": {m: round(v / n_draws, 4) for m, v in worst_neural.items()},
        "share_of_draws_with_same_holm_conclusion_as_union": {
            pair: round(count / n_draws, 4) for pair, count in same_conclusion.items()},
    }


def summarise_questions(questions: list[dict], models: list[str], n_boot: int = 4000,
                        n_draws: int = 2000, seed: int = DEFAULT_SEED) -> dict:
    stored_flags = [q["reference_invariant"] for q in questions]
    if stored_flags != [reference_invariant(q) for q in questions]:
        raise ValueError("stored reference_invariant flags disagree with the annotations")
    references = {scheme: scheme_rows(questions, models, scheme) for scheme in EVIDENCE_SET_SCHEMES}
    invariant_ids = {q["question_id"] for q in questions if q["reference_invariant"]}
    references["agreement_subset"] = [r for r in references["union"] if r["question_id"] in invariant_ids]
    summary = {"schemes": {name: scheme_summary(rows, models, n_boot) for name, rows in references.items()},
               "annotation_profile": annotation_profile(questions)}
    union = {r["question_id"]: r for r in references["union"]}
    summary["first_vs_union"] = {m: {
        "questions_with_worse_rank": sum(1 for r in references["first"] if r["rank"][m] > union[r["question_id"]]["rank"][m]),
        "hit@1_lost": sum(1 for r in references["first"]
                          if r["rank"][m] != 1 and union[r["question_id"]]["rank"][m] == 1),
        "hit@5_lost": sum(1 for r in references["first"]
                          if r["rank"][m] > 5 and union[r["question_id"]]["rank"][m] <= 5),
    } for m in models}
    signs = _holm_significant(references["union"], models)
    summary["holm_significant_pairs"] = {name: {pair: sign for pair, sign in _holm_significant(rows, models).items() if sign}
                                         for name, rows in references.items()}
    summary["single_annotator_resampling"] = single_annotator_resampling(questions, models, n_draws, seed, signs)
    return summary


def annotation_profile(questions: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for q in questions:
        for a in q["annotations"]:
            statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    usable = [len(usable_annotations(q["annotations"])) for q in questions]
    multi = [q for q in questions if len(usable_annotations(q["annotations"])) > 1]
    # For questions with two or more usable annotators: does any paragraph appear in two
    # annotators' evidence, and do their evidence sets cover the same chunks?
    shared = sum(1 for q in multi if reference_spans(q["annotations"], "agreement") is not None)
    same_chunks = sum(1 for q in multi if len({span_gold(q, sorted(evidence_spans(a)))
                                               for a in usable_annotations(q["annotations"])}) == 1)
    spans_per_annotator = [len(evidence_spans(a)) for q in questions for a in usable_annotations(q["annotations"])]
    first_excluded = sum(1 for q in questions if reference_spans(q["annotations"], "first") is None)
    return {
        "n_questions": len(questions),
        "annotator_status_counts": dict(sorted(statuses.items())),
        "annotators_with_usable_evidence": {str(k): usable.count(k) for k in sorted(set(usable))},
        "mean_evidence_paragraphs_per_usable_annotator": round(float(np.mean(spans_per_annotator)), 3),
        "questions_with_two_or_more_usable_annotators": len(multi),
        "of_which_share_a_paragraph": shared,
        "of_which_cover_the_same_chunks": same_chunks,
        "questions_where_some_reference_differs_from_union": sum(1 for q in questions if not q["reference_invariant"]),
        "questions_without_first_annotator_evidence_in_text": first_excluded,
    }


def summarise(args: argparse.Namespace) -> int:
    provenance = collect_provenance(script="scripts/qasper_reference_sensitivity.py --summarise")
    out_dir = Path(args.out_dir)
    per_question = out_dir / PER_QUESTION
    data = json.loads(per_question.read_text(encoding="utf-8"))
    models = [m for m in MODELS if m in data["localisers"]]
    summary = {
        "status": "exploratory; not part of the submitted manuscript",
        "schemes_defined_in": "src/evaluation/references.py (evidence sets)",
        "input": {"path": per_question.as_posix(), "sha256_lf": sha256_lf(per_question)},
        "settings": {"n_boot": args.n_boot, "bootstrap_seed": 0, "n_draws": args.n_draws,
                     "resampling_seed": args.seed},
        **summarise_questions(data["questions"], models, args.n_boot, args.n_draws, args.seed),
        "manuscript_check": manuscript_check(data["questions"], models, Path(args.frozen_dir)),
        "provenance": provenance,
    }
    write_lf(out_dir / SUMMARY_JSON, json.dumps(summary, indent=1) + "\n")
    write_lf(out_dir / SUMMARY_MD, render_markdown(summary, models))
    print(f"wrote {out_dir / SUMMARY_JSON} and {out_dir / SUMMARY_MD}")
    return 0


def manuscript_check(questions: list[dict], models: list[str], frozen_dir: Path) -> dict:
    """Under the union of annotators the numbers must be the manuscript's, to the last digit."""
    stored = json.loads((frozen_dir / "summary_qasper.json").read_text(encoding="utf-8"))
    rows = scheme_rows(questions, models, "union")
    ours = {m: model_metrics(rows, m) for m in models}
    tests = pairwise(rows, models, n_boot=10, seed=0)
    metric_diffs = {f"{m}.{k}": [ours[m][k], stored["models"][m][k]] for m in models
                    for k in ("hit@1", "hit@5", "mrr") if ours[m][k] != stored["models"][m][k]}
    test_diffs = {f"{pair}.{k}": [tests[pair][k], stored["pairwise"][pair][k]] for pair in tests
                  for k in ("mcnemar_exact_p", "mcnemar_holm_adjusted_p")
                  if tests[pair][k] != stored["pairwise"][pair][k]}
    any_hit = round(sum(any(r["rank"][m] == 1 for m in models) for r in rows) / len(rows), 4)
    if any_hit != stored["union"]["any_hit@1"]:
        metric_diffs["any_localiser_hit@1"] = [any_hit, stored["union"]["any_hit@1"]]
    return {"compared_with": (frozen_dir / "summary_qasper.json").as_posix(),
            "metrics_identical": not metric_diffs, "pairwise_tests_identical": not test_diffs,
            "differences": {**metric_diffs, **test_diffs}}


def render_markdown(summary: dict, models: list[str]) -> str:
    names = {"union": "union of annotators (manuscript)", "first": "first annotator",
             "plurality": "plurality paragraphs", "agreement": "paragraphs marked twice",
             "agreement_subset": "questions where the reference does not matter"}
    schemes = summary["schemes"]
    lines = ["# QASPER localisation under alternative evidence references", "",
             "**Exploratory. Not part of the submitted manuscript.** Generated by "
             "`scripts/qasper_reference_sensitivity.py --summarise` from `per_question.json`.", "",
             "## hit@1 (hit@5) by reference", "",
             "| Localiser | " + " | ".join(f"{names[s]} (n={schemes[s]['n_questions']})" for s in names) + " |",
             "|---|" + "---:|" * len(names)]
    for m in models:
        cells = [f"{schemes[s]['models'][m]['hit@1']:.3f} ({schemes[s]['models'][m]['hit@5']:.3f})" for s in names]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "| | " + " | ".join(names[s] for s in names) + " |", "|---|" + "---:|" * len(names),
              "| gold paragraphs per question | " + " | ".join(
                  f"{schemes[s]['mean_gold_spans_per_question']:.2f}" for s in names) + " |",
              "| gold chunks per question | " + " | ".join(
                  f"{schemes[s]['mean_gold_chunks_per_question']:.2f}" for s in names) + " |",
              "| chance hit@1 (random order) | " + " | ".join(
                  f"{schemes[s]['models'][models[0]]['chance_hit@1']:.3f}" for s in names) + " |",
              "| best at hit@1 | " + " | ".join(schemes[s]["best_at_hit@1"] for s in names) + " |",
              "| worst neural at hit@1 | " + " | ".join(schemes[s]["worst_neural_at_hit@1"] for s in names) + " |",
              "| some localiser at rank 1 | " + " | ".join(
                  f"{schemes[s]['any_localiser_hit@1']:.3f}" for s in names) + " |",
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
              "| Localiser | mean hit@1 | 95% of draws | share of draws best | share of draws worst neural |",
              "|---|---:|---:|---:|---:|"]
    for m in models:
        h = res["hit@1"][m]
        worst = res["share_of_draws_worst_neural_at_hit@1"].get(m)
        lines.append(f"| {m} | {h['mean']:.3f} | {h['p2.5']:.3f}–{h['p97.5']:.3f} | "
                     f"{res['share_of_draws_best_at_hit@1'][m]:.3f} | "
                     f"{'—' if worst is None else f'{worst:.3f}'} |")
    check = summary["manuscript_check"]
    lines += ["", "## Check against the manuscript", "",
              f"Under the union of annotators, hit@1, hit@5 and MRR identical to `summary_qasper.json`: "
              f"{check['metrics_identical']}; McNemar and Holm p-values identical: {check['pairwise_tests_identical']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--score", action="store_true")
    step.add_argument("--summarise", action="store_true")
    ap.add_argument("--raw", default="data/raw/qasper-dev-v0.3.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--chunk-overlap", type=int, default=32)
    ap.add_argument("--chroma-map", default=DEFAULT_CHROMA_MAP,
                    help="key=index_dir:collection for minilm, mpnet, bge and e5; point it at copies")
    ap.add_argument("--ce-scope", default="ce_msmarco_minilm=all,bge_reranker_base=all",
                    help="per cross-encoder, 'all' or 'affected' (only questions where some reference "
                         "differs from the union; elsewhere every scheme gives the stored rank)")
    ap.add_argument("--frozen-dir", default="results/localisation")
    ap.add_argument("--out-dir", default="results/exploratory/qasper_reference_sensitivity")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    if Path(args.out_dir).resolve().is_relative_to((REPO / "results" / "localisation").resolve()):
        raise SystemExit("refusing to write into results/localisation/, which holds the manuscript's results")
    return score(args) if args.score else summarise(args)


if __name__ == "__main__":
    raise SystemExit(main())
