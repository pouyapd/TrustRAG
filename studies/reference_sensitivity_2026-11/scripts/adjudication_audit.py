#!/usr/bin/env python
"""Automated consistency audit of the author's gold-span derivability adjudication (133 units).

NOT a second annotator. An NLI cross-encoder (cross-encoder/nli-deberta-v3-base) scores, for
each unit, whether some retrieved chunk entails "The answer to the question "<q>" is: <a>"
for some reference answer. The maximum entailment probability is compared with the author's
YES/NO label to flag possible inconsistencies, and is computed for the 36 units nobody read.
Sensitivity bounds for the under-coverage rate are derived from the flags.

Inputs: the deposited annotation package (full-context sheet, CC BY 4.0, extracted under
data/annotation_package/), the repository's proxy partition and the author's labels.
Output: results/audit/adjudication_audit.json (no dataset text).
"""
from __future__ import annotations

import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
SHEET = REV / "data/annotation_package/trustrag-annotation-package-v1.0.0/reports/annotation/qasper_dev_300_full_context/annotation_sheet.jsonl"
HI, LO = 0.9, 0.1


def labels() -> dict[str, str]:
    out = {}
    for p in ("results/goldspan_adjudication/first_pass/answers.csv", "results/goldspan_adjudication/remaining37/answers.csv"):
        with open(REPO / p, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].startswith("unit_"):
                    out[row[0]] = row[1].strip().upper()
    return out


def main() -> int:
    from sentence_transformers import CrossEncoder
    sem = json.loads((REPO / "results/goldspan_adjudication/gold_span_semantic.json").read_text(encoding="utf-8"))
    bucket = {u["annotation_id"]: u["bucket"] for u in sem["units"]}
    sheet = {r["annotation_id"]: r for r in map(json.loads, SHEET.open(encoding="utf-8"))}
    lab = labels()
    model_id = "cross-encoder/nli-deberta-v3-base"
    nli = CrossEncoder(model_id, device="cpu", max_length=512)
    ent_idx = [i for i, l in nli.model.config.id2label.items() if l.lower().startswith("entail")][0]
    t0 = time.time()
    units = []
    for uid in sorted(bucket):
        r = sheet[uid]
        answers = r["reference_answers"] if isinstance(r["reference_answers"], list) else ast.literal_eval(r["reference_answers"])
        ctx = r["retrieved_context"] if isinstance(r["retrieved_context"], list) else ast.literal_eval(r["retrieved_context"])
        chunks = [c["text"] for c in ctx]
        pairs = [(c, f'The answer to the question "{r["question"]}" is: {a}.') for c in chunks for a in answers]
        logits = np.asarray(nli.predict(pairs, batch_size=16, show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        units.append({"annotation_id": uid, "bucket": bucket[uid], "author_label": lab.get(uid),
                      "max_entailment": float(probs[:, ent_idx].max()), "n_pairs": len(pairs)})
        print(uid, lab.get(uid), round(units[-1]["max_entailment"], 3), flush=True)
    judged = [u for u in units if u["author_label"]]
    unread = [u for u in units if not u["author_label"]]
    yes = [u for u in judged if u["author_label"] == "YES"]
    no = [u for u in judged if u["author_label"] == "NO"]
    flag_no = [u["annotation_id"] for u in no if u["max_entailment"] >= HI]
    flag_yes = [u["annotation_id"] for u in yes if u["max_entailment"] < LO]
    unread_hi = [u["annotation_id"] for u in unread if u["max_entailment"] >= HI]

    def auc(pos, neg):
        if not pos or not neg:
            return None
        return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

    n_pop = len(units)
    rate = len(yes) / n_pop
    out = {
        "status": "automated consistency audit; not a human annotation; no labels changed",
        "nli_model": model_id, "hypothesis_template": 'The answer to the question "<q>" is: <a>.',
        "thresholds": {"high": HI, "low": LO}, "n_units": n_pop, "n_judged": len(judged), "n_unread": len(unread),
        "author_yes": len(yes), "author_no": len(no),
        "auc_nli_vs_author": auc([u["max_entailment"] for u in yes], [u["max_entailment"] for u in no]),
        "median_entailment_yes": float(np.median([u["max_entailment"] for u in yes])),
        "median_entailment_no": float(np.median([u["max_entailment"] for u in no])),
        "flag_author_no_but_nli_high": flag_no, "flag_author_yes_but_nli_low": flag_yes,
        "unread_with_nli_high": unread_hi,
        "under_coverage": {
            "author_count": rate,
            "if_flagged_no_were_yes": (len(yes) + len(flag_no)) / n_pop,
            "if_flagged_yes_were_no": (len(yes) - len(flag_yes)) / n_pop,
            "plus_unread_nli_high": (len(yes) + len(unread_hi)) / n_pop,
            "upper_bound_all_flags_and_unread_high": (len(yes) + len(flag_no) + len(unread_hi)) / n_pop,
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "units": units,
    }
    dst = REV / "results/audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "adjudication_audit.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "units"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
