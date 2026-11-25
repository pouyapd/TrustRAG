#!/usr/bin/env python
"""Bridge for the late-interaction (ColBERT) run, which needs PyLate in a separate environment.

  export (main venv):  python colbert_bridge.py export qasper_full
      writes data/colbert/<file>.texts.jsonl.gz (questions + gold-document chunk texts; dataset
      text, kept private, never published)
  score  (.venv_colbert): python colbert_bridge.py score qasper_full
      PyLate's ColBERT (answerdotai/answerai-colbert-small-v1); documents encoded once per
      document (is_query=False), queries with is_query=True; MaxSim score = sum over query tokens
      of the max dot product with document tokens (PyLate's embeddings are L2-normalised).
      Checkpoints data/colbert/<file>.orders.json every 100 questions; resumable.
  merge (main venv):   python colbert_bridge.py merge qasper_full
      adds the orders to results/raw/<file>.json.gz as model "colbert_small".
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
DIR = REV / "data/colbert"
MODEL = "answerdotai/answerai-colbert-small-v1"


def export(name):
    sys.path.insert(0, str(REV / "scripts"))
    sys.path.insert(0, str(REV.parents[1]))
    from add_model import rebuild_chunks
    data = json.load(gzip.open(REV / f"results/raw/{name}.json.gz", "rt", encoding="utf-8"))
    by_doc = rebuild_chunks(data["config"], data["corpus"])
    DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(DIR / f"{name}.texts.jsonl.gz", "wt", encoding="utf-8") as f:
        for q in data["questions"]:
            ch = by_doc[q["doc_id"]]
            assert [[c.start_char, c.end_char] for c in ch] == [list(r) for r in q["chunk_ranges"]]
            f.write(json.dumps({"qid": q["question_id"], "doc": q["doc_id"], "q": q["question"],
                                "chunks": [c.text for c in ch]}) + "\n")
    print("exported", len(data["questions"]))


def score(name):
    import numpy as np
    from pylate import models
    rows = [json.loads(l) for l in gzip.open(DIR / f"{name}.texts.jsonl.gz", "rt", encoding="utf-8")]
    out_p = DIR / f"{name}.orders.json"
    orders = json.loads(out_p.read_text()) if out_p.exists() else {}
    m = models.ColBERT(model_name_or_path=MODEL, device="cpu")
    t0, done, cache = time.time(), 0, {}
    for r in rows:
        if r["qid"] in orders:
            continue
        if r["doc"] not in cache:
            cache = {r["doc"]: m.encode(r["chunks"], is_query=False, batch_size=16, show_progress_bar=False)}
        qe = m.encode([r["q"]], is_query=True, show_progress_bar=False)[0]
        s = [float((np.asarray(qe) @ np.asarray(d).T).max(axis=1).sum()) for d in cache[r["doc"]]]
        orders[r["qid"]] = [int(i) for i in np.argsort(-np.asarray(s), kind="stable")]
        done += 1
        if done % 100 == 0:
            tmp = Path(str(out_p) + ".tmp"); tmp.write_text(json.dumps(orders)); os.replace(tmp, out_p)
            print(f"  {done} {time.time() - t0:.0f}s", flush=True)
    tmp = Path(str(out_p) + ".tmp"); tmp.write_text(json.dumps(orders)); os.replace(tmp, out_p)
    import pylate
    print(f"done {done} in {time.time() - t0:.0f}s; pylate {pylate.__version__}", flush=True)


def merge(name):
    p = REV / f"results/raw/{name}.json.gz"
    data = json.load(gzip.open(p, "rt", encoding="utf-8"))
    orders = json.loads((DIR / f"{name}.orders.json").read_text())
    if len(orders) != len(data["questions"]):
        raise SystemExit(f"incomplete: {len(orders)} of {len(data['questions'])}")
    for q in data["questions"]:
        o = orders[q["question_id"]]
        assert sorted(o) == list(range(q["D"]))
        q["orders"]["colbert_small"] = o
    snap = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "refs/main"
    data["config"]["models"]["colbert_small"] = {"identity": MODEL, "backend": "pylate (separate venv)",
                                                 "note": "Answer.AI ColBERT small v1, Aug 2024; MaxSim",
                                                 "cached_revision": snap.read_text().strip() if snap.exists() else None}
    tmp = Path(str(p) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, p)
    print("merged", len(orders))


if __name__ == "__main__":
    {"export": export, "score": score, "merge": merge}[sys.argv[1]](sys.argv[2])
