#!/usr/bin/env python
"""Assemble the subset files on which every model, including those too slow for the full samples,
is compared on the same questions:

  results/raw/qasper_sub290.json.gz  the earlier study's 290 QASPER questions: BM25, MiniLM, CE (from
                                     the full run), BGE-M3 (scored on this file), and Granite, Jina,
                                     ColBERT copied from the full run
  results/raw/nq_sub300.json.gz      the earlier study's 300 NQ questions (the first 300 rows of the
                                     full run): every full-run model plus ColBERT (first 300 orderings)

Orderings are copied unchanged; BM25's are those of the full run (corpus-level IDF over the full
sample). A model is included only if every question of the subset has its ordering."""
import gzip
import json
import os
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from localisation_robustness import load_ranks  # noqa: E402


def load(name):
    return json.load(gzip.open(REV / f"results/raw/{name}.json.gz", "rt", encoding="utf-8"))


def save(name, data):
    p = REV / f"results/raw/{name}.json.gz"
    tmp = Path(str(p) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def complete(qs, m):
    return all(m in q["orders"] for q in qs)


def main(which=("qasper", "nq")):
    if "qasper" in which:
        qasper()
    if "nq" in which:
        nq_part()


def qasper():
    # QASPER: start from the subset file (holds BGE-M3), add models from the full run
    full = load("qasper_full")
    sub = load("qasper_sub290")
    byid = {q["question_id"]: q for q in full["questions"]}
    ids = set(load_ranks(REPO / "results/localisation", "qasper")["bm25"])
    assert {q["question_id"] for q in sub["questions"]} == ids
    for m, meta in full["config"]["models"].items():
        if m not in sub["config"]["models"] and complete([byid[i] for i in ids], m):
            for q in sub["questions"]:
                q["orders"][m] = byid[q["question_id"]]["orders"][m]
            sub["config"]["models"][m] = meta
    save("qasper_sub290", sub)
    print("qasper_sub290:", list(sub["config"]["models"]))


def nq_part():
    # NQ: first 300 rows of the full run, plus ColBERT
    nq = load("nq_full")
    ids = set(load_ranks(REPO / "results/localisation", "nq")["bm25"])
    qs = nq["questions"][:300]
    assert {q["question_id"] for q in qs} == ids
    models = {m: meta for m, meta in nq["config"]["models"].items() if complete(qs, m)}
    # partially scored models (e.g. a resumed job) may already cover the subset
    for m in {k for q in qs for k in q["orders"]} - set(models):
        if complete(qs, m):
            models[m] = {"identity": m, "note": "scored on this subset from a partially completed full run"}
    cb = REV / "data/colbert/nq_full.orders.json"
    if cb.exists():
        o = json.loads(cb.read_text())
        if all(q["question_id"] in o for q in qs):
            for q in qs:
                q["orders"]["colbert_small"] = o[q["question_id"]]
            models["colbert_small"] = {"identity": "answerdotai/answerai-colbert-small-v1", "backend": "pylate (separate venv)"}
    for q in qs:
        q["orders"] = {m: q["orders"][m] for m in models}
    out = dict(nq)
    out["questions"] = qs
    out["config"] = dict(nq["config"]); out["config"]["models"] = models
    out["description"] = nq["description"] + " Subset: the earlier study's 300 NQ questions."
    save("nq_sub300", out)
    print("nq_sub300:", list(models))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("qasper", "nq"))
