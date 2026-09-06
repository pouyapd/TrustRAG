#!/usr/bin/env python
"""Derive the final human-reviewed dataset from the original pass and the review pass.

Two human passes exist over the same 200 units: the original annotation, and a
second review of the units an audit flagged. This combines them -- original label
for the units never flagged, review decision for the units that were -- into a new
derived dataset, and records the full chain for every unit:

    original label -> flagged (reason) -> review decision -> final label

Both source passes are opened read-only and neither is modified. The derived file
is a new artifact; the historical record stays exactly where it was.

    python scripts/build_final_human_dataset.py \
        --original reports/annotation/qasper_dev_300_full_context --original-annotator human \
        --review reports/annotation/review_43_flagged --review-annotator review \
        --audit reports/annotation/qasper_dev_300_full_context/audit/human_annotation_audit.json \
        --out reports/annotation/qasper_dev_300_full_context/final_human_reviewed
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HUMAN_FIELDS = ("human_label", "human_confidence", "human_notes")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--original-annotator", default="human")
    ap.add_argument("--review", required=True)
    ap.add_argument("--review-annotator", default="review")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--review-audit", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    orig_pkg, rev_pkg, out = Path(args.original), Path(args.review), Path(args.out)
    orig_file = orig_pkg / f"annotator_{args.original_annotator}" / "completed.jsonl"
    rev_file = rev_pkg / f"annotator_{args.review_annotator}" / "completed.jsonl"

    original = {r["annotation_id"]: r for r in read_jsonl(orig_file)}
    review = {r["annotation_id"]: r for r in read_jsonl(rev_file)}
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    flagged = {u["annotation_id"]: u for u in audit["units"]
               if u["verdict"] == "likely_inconsistent"}
    rev_audit = {}
    if args.review_audit:
        rev_audit = {u["annotation_id"]: u for u in
                     json.loads(Path(args.review_audit).read_text(encoding="utf-8"))["units"]}

    if set(review) != set(flagged):
        print("the review pass does not cover exactly the flagged units - refusing")
        return 1

    rows, chain = [], []
    n_changed = n_upheld = 0
    for unit_id in sorted(original):
        row = dict(original[unit_id])
        was_flagged = unit_id in flagged
        if was_flagged:
            source = "second_review"
            for field in HUMAN_FIELDS:
                row[field] = review[unit_id].get(field, "")
            changed = original[unit_id]["human_label"] != review[unit_id]["human_label"]
            n_changed += changed
            n_upheld += not changed
        else:
            source = "original_pass"
            changed = False

        rows.append(row)
        entry = {
            "annotation_id": unit_id,
            "original_label": original[unit_id]["human_label"],
            "original_confidence": original[unit_id].get("human_confidence", ""),
            "flagged_by_audit": was_flagged,
            "flag_reason": flagged[unit_id]["reasons"][0] if was_flagged else None,
            "second_review_label": review[unit_id]["human_label"] if was_flagged else None,
            "second_review_confidence": review[unit_id].get("human_confidence", "") if was_flagged else None,
            "label_changed_in_review": changed,
            "final_label": row["human_label"],
            "final_confidence": row.get("human_confidence", ""),
            "final_label_source": source,
        }
        if was_flagged and unit_id in rev_audit:
            entry["post_review_audit_verdict"] = rev_audit[unit_id]["verdict"]
            entry["post_review_audit_reason"] = rev_audit[unit_id]["reasons"][0]
        chain.append(entry)

    out.mkdir(parents=True, exist_ok=True)
    (out / "completed.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "provenance_chain.json").write_text(
        json.dumps(chain, indent=2, ensure_ascii=False), encoding="utf-8")

    dist = Counter(r["human_label"] for r in rows)
    orig_dist = Counter(r["human_label"] for r in original.values())
    still_flagged = sum(1 for e in chain
                        if e.get("post_review_audit_verdict") == "likely_inconsistent")

    manifest = {
        "kind": "final human-reviewed dataset (derived)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_units": len(rows),
        "composition": {
            "from_original_pass": sum(1 for e in chain if e["final_label_source"] == "original_pass"),
            "from_second_review": sum(1 for e in chain if e["final_label_source"] == "second_review"),
        },
        "review_outcome": {
            "flagged_and_reviewed": len(flagged),
            "label_changed_in_review": n_changed,
            "label_upheld_in_review": n_upheld,
            "still_guideline_inconsistent_after_review": still_flagged or None,
        },
        "sources": {
            "original_pass": {"file": orig_file.as_posix(), "sha256": sha256(orig_file)},
            "second_review": {"file": rev_file.as_posix(), "sha256": sha256(rev_file)},
            "audit": {"file": Path(args.audit).as_posix(), "sha256": sha256(Path(args.audit))},
        },
        "final_label_distribution": dict(dist.most_common()),
        "original_label_distribution": dict(orig_dist.most_common()),
        "provenance_note": "Derived artifact. Both source passes are preserved unmodified at "
                           "the paths and checksums above. Every unit's chain -- original "
                           "label, flag reason, review decision, final label -- is in "
                           "provenance_chain.json.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"final dataset : {out.as_posix()}/completed.jsonl")
    print(f"units         : {len(rows)} "
          f"({manifest['composition']['from_original_pass']} original + "
          f"{manifest['composition']['from_second_review']} reviewed)")
    print(f"review changed: {n_changed} label(s); upheld {n_upheld}")
    print(f"distribution  : {dict(dist.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
