#!/usr/bin/env python
"""Compare score_v2 ranks with the original study's stored ranks (results/localisation).

The study's reference is QASPER: union of annotators; NQ: first annotator's long answer,
which score_v2 stores as ``study_gold``. Any rank mismatch is reported per model.
"""
import gzip, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO))
from localisation_robustness import load_ranks  # noqa: E402

def first_rank(order, gold):
    gold = set(gold)
    return next(p for p, i in enumerate(order, 1) if i in gold)

def main(path, corpus):
    data = json.load(gzip.open(path, "rt", encoding="utf-8"))
    frozen = load_ranks(REPO / "results/localisation", corpus)
    out = {}
    for model in data["config"]["models"]:
        if model not in frozen:
            continue
        rows = [q for q in data["questions"] if q["question_id"] in frozen[model]]
        mism = [q["question_id"] for q in rows if first_rank(q["orders"][model], q["study_gold"]) != frozen[model][q["question_id"]]]
        out[model] = {"compared": len(rows), "study_questions": len(frozen[model]), "mismatches": len(mism), "examples": mism[:5]}
    print(json.dumps(out, indent=1))
    return out

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
