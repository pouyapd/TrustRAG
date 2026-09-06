#!/usr/bin/env python
"""Audit a completed human annotation pass against the guidelines and the evidence.

This never writes to the annotation file. It reads the human labels, recomputes the
objective facts each label depends on -- answerability, whether any retrieved chunk
covers a gold span, what the generator was given -- and reports, per unit, how well
the written decision procedure in docs/ANNOTATION_GUIDELINES.md supports the label
that was chosen.

The verdicts are deliberately conservative. `strongly_supported` is reserved for
labels whose preconditions are objectively checkable and hold; `likely_inconsistent`
is used only where a label contradicts an explicit rule in the guidelines, not where
it merely differs from what the system or the reference pass concluded.

    python scripts/audit_human_annotations.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --annotator human \
        --reference reports/annotation/qasper_dev_300_full_context/annotator_a/completed.jsonl \
        --rows reports/experiments/qasper_dev_300/rows.jsonl \
        --records reports/experiments/qasper_dev_300/inference.jsonl \
        --out reports/annotation/qasper_dev_300_full_context/audit
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.statistics import cohens_kappa, confusion_matrix  # noqa: E402

STEP3_LABELS = {"ok", "partial_answer", "incorrect_answer",
                "refusal_when_answerable", "hallucination"}
UNANSWERABLE_LABELS = {"ok_abstained", "answered_when_unanswerable"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def span_coverage(unit: dict) -> tuple[int, int]:
    """Gold spans covered by some retrieved chunk, recomputed from the sheet.

    Independent of the stored run: half-open interval overlap on char_range,
    within the same document.
    """
    covered = 0
    gold = unit.get("gold_evidence") or []
    for g in gold:
        gs, ge = g["char_range"]
        for c in unit.get("retrieved_context") or []:
            if c["doc_id"] != g["doc_id"]:
                continue
            cs, ce = c["char_range"]
            if min(ge, ce) - max(gs, cs) > 0:
                covered += 1
                break
    return covered, len(gold)


def audit_unit(unit: dict, human: dict, row: dict | None) -> dict:
    """One unit: the facts, then how well the guidelines support the label."""
    label = human["human_label"]
    answerable = bool(unit["corpus_can_answer"])
    n_retrieved = len(unit.get("retrieved_context") or [])
    covered, n_gold = span_coverage(unit)
    evidence_present = covered > 0

    facts = {
        "corpus_can_answer": answerable,
        "n_retrieved_chunks": n_retrieved,
        "n_gold_spans": n_gold,
        "n_gold_spans_covered": covered,
        "evidence_reached_generator": evidence_present,
        "evidence_status_in_run": (row or {}).get("evidence_status"),
        "abstained_in_run": (row or {}).get("abstained"),
        "answer_f1_normalized": (row or {}).get("answer_f1_normalized"),
        "key_fact_recall": (row or {}).get("key_fact_recall"),
    }

    verdict, reasons = "plausibly_supported", []

    # --- Step 1: answerability gates everything else -----------------------
    if not answerable:
        if label in UNANSWERABLE_LABELS:
            abstained = facts["abstained_in_run"]
            expected = "ok_abstained" if abstained else "answered_when_unanswerable"
            if abstained is None:
                verdict = "ambiguous"
                reasons.append("step 1: corpus_can_answer=false and the label is "
                               "admissible, but the run records no abstention signal "
                               "to distinguish the two categories")
            elif label == expected:
                verdict = "strongly_supported"
                reasons.append(f"step 1: corpus_can_answer=false and the system "
                               f"{'declined' if abstained else 'produced a substantive answer'}, "
                               f"which is what {label} requires")
            else:
                verdict = "likely_inconsistent"
                reasons.append(f"step 1: corpus_can_answer=false, but the system "
                               f"{'declined' if abstained else 'produced a substantive answer'}, "
                               f"so the guidelines direct {expected} rather than {label}")
        else:
            verdict = "likely_inconsistent"
            reasons.append("step 1 of the guidelines admits only ok_abstained or "
                           "answered_when_unanswerable when corpus_can_answer=false")
        return {"verdict": verdict, "reasons": reasons, "facts": facts}

    if label in UNANSWERABLE_LABELS:
        verdict = "likely_inconsistent"
        reasons.append("label is reserved for corpus_can_answer=false, but this "
                       "question is marked answerable")
        return {"verdict": verdict, "reasons": reasons, "facts": facts}

    # --- Step 2: did the evidence reach the system? ------------------------
    if label == "no_retrieval":
        if n_retrieved == 0:
            verdict = "strongly_supported"
            reasons.append("step 2: retrieved_context is empty")
        else:
            verdict = "likely_inconsistent"
            reasons.append(f"the guidelines define no_retrieval as an empty "
                           f"retrieved_context; this unit has {n_retrieved} chunks")
            if not evidence_present:
                reasons.append("wrong_retrieval is the category the guidelines assign "
                               "to this situation (chunks arrived, none of them useful)")
        return {"verdict": verdict, "reasons": reasons, "facts": facts}

    if label == "wrong_retrieval":
        if not evidence_present:
            verdict = "strongly_supported"
            reasons.append("step 2: no retrieved chunk covers any gold span")
        else:
            verdict = "ambiguous"
            reasons.append(f"{covered}/{n_gold} gold span(s) were covered by a "
                           "retrieved chunk, so the evidence was present by offset; "
                           "the guidelines still allow this label if the retrieved "
                           "text does not in fact carry the needed information")
        return {"verdict": verdict, "reasons": reasons, "facts": facts}

    # --- Step 3: only reached when the evidence was present ----------------
    if label in STEP3_LABELS:
        if not evidence_present:
            verdict = "likely_inconsistent"
            reasons.append("step 2 directs a retrieval label when no gold span reached "
                           "the system, explicitly even when the answer is correct; a "
                           "step-3 label skips that gate")
            return {"verdict": verdict, "reasons": reasons, "facts": facts}

        reasons.append("step 2 passed: a gold span was covered, so step 3 applies")
        kfr = facts["key_fact_recall"]
        f1 = facts["answer_f1_normalized"]
        abstained = facts["abstained_in_run"]
        if label == "refusal_when_answerable":
            verdict = "strongly_supported" if abstained else "likely_inconsistent"
            reasons.append("the stored run records an abstention" if abstained
                           else "the stored run records no abstention")
        elif label == "ok" and kfr is not None and kfr >= 1.0:
            verdict = "strongly_supported"
            reasons.append(f"all reference key facts present (key_fact_recall={kfr:.2f})")
        elif label == "incorrect_answer" and kfr is not None and kfr <= 0.2:
            verdict = "strongly_supported"
            reasons.append(f"almost no reference key facts present (key_fact_recall={kfr:.2f})")
        elif label in {"ok", "partial_answer", "incorrect_answer"}:
            verdict = "plausibly_supported"
            reasons.append(f"answer-quality judgement; run signals: key_fact_recall={kfr}, "
                           f"answer_f1={f1}")
        elif label == "hallucination":
            verdict = "ambiguous"
            reasons.append("hallucination requires invented specifics absent from both "
                           "context and reference; not checkable from stored signals alone")
    return {"verdict": verdict, "reasons": reasons, "facts": facts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--annotator", default="human")
    ap.add_argument("--reference", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--records", default="")
    ap.add_argument("--key", default="",
                    help="proposed_labels_key.jsonl mapping annotation_id -> question_id; "
                         "defaults to the one inside --package. Only the question_id column "
                         "is read, never the proposed label.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg = Path(args.package)
    sheet = {u["annotation_id"]: u for u in read_jsonl(pkg / "annotation_sheet.jsonl")}
    human = {r["annotation_id"]: r
             for r in read_jsonl(pkg / f"annotator_{args.annotator}" / "completed.jsonl")}

    rows_by_unit: dict[str, dict] = {}
    if args.rows and args.records:
        key_path = Path(args.key) if args.key else pkg / "proposed_labels_key.jsonl"
        key = {k["annotation_id"]: k.get("question_id") for k in read_jsonl(key_path)}
        records = read_jsonl(Path(args.records))
        rows = read_jsonl(Path(args.rows))
        index = {(r.get("metadata") or {}).get("question_id"): i
                 for i, r in enumerate(records)}
        for unit_id, qid in key.items():
            i = index.get(qid)
            if i is not None and i < len(rows):
                rows_by_unit[unit_id] = rows[i]

    reference = {}
    if args.reference:
        reference = {r["annotation_id"]: r.get("human_label", "")
                     for r in read_jsonl(Path(args.reference))}

    units = []
    for unit_id in sorted(human):
        unit = sheet[unit_id]
        result = audit_unit(unit, human[unit_id], rows_by_unit.get(unit_id))
        row = rows_by_unit.get(unit_id, {})
        units.append({
            "annotation_id": unit_id,
            "question": unit["question"],
            "human_label": human[unit_id]["human_label"],
            "human_confidence": human[unit_id].get("human_confidence", ""),
            "human_notes": human[unit_id].get("human_notes", ""),
            "reference_label": reference.get(unit_id, ""),
            "taxonomy_document_gated": row.get("failure_mode_v2"),
            "taxonomy_evidence_gated": row.get("failure_mode_evidence"),
            "agrees_with_reference": reference.get(unit_id) == human[unit_id]["human_label"],
            **result,
        })

    verdicts = Counter(u["verdict"] for u in units)
    shared = [u for u in units if u["reference_label"]]
    h = [u["human_label"] for u in shared]
    r = [u["reference_label"] for u in shared]

    report = {
        "package": str(pkg),
        "annotator": args.annotator,
        "n_units": len(units),
        "note": "Verdicts describe support from the written guidelines and the stored "
                "evidence. They are not corrections and no human label was modified.",
        "label_distribution": dict(Counter(u["human_label"] for u in units).most_common()),
        "confidence_distribution": dict(Counter(u["human_confidence"] for u in units).most_common()),
        "verdicts": dict(verdicts),
        "verdicts_by_label": {
            label: dict(Counter(u["verdict"] for u in units if u["human_label"] == label))
            for label in sorted({u["human_label"] for u in units})
        },
        "vs_reference": {
            "reference_file": args.reference,
            "reference_provenance": "automated pass produced by a language-model "
                                    "annotator; not human ground truth",
            "n": len(shared),
            "raw_agreement": round(sum(1 for a, b in zip(h, r, strict=True) if a == b) / len(shared), 4)
                             if shared else None,
            "cohens_kappa": cohens_kappa(r, h).as_dict() if shared else None,
            "confusion_matrix": confusion_matrix(r, h) if shared else None,
        },
        "units": units,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "human_annotation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n": len(units), "verdicts": dict(verdicts),
                      "raw_agreement": report["vs_reference"]["raw_agreement"],
                      "kappa": (report["vs_reference"]["cohens_kappa"] or {}).get("kappa")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
