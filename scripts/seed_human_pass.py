#!/usr/bin/env python
"""Open a human annotation pass that starts from labels already collected.

The 22 units the project owner labelled by hand live in an earlier annotation
package. This copies those labels — and nothing else — into a fresh annotator
slot on the current package, marks them locked so the interface cannot
overwrite them by accident, and leaves every other unit empty for the annotator
to fill in.

It never invents a label. A unit is carried over only when the source file
holds a non-empty label for it and both packages agree on which question that
`annotation_id` refers to; anything else is reported and skipped.

    python scripts/seed_human_pass.py \\
        --package reports/annotation/qasper_dev_300_full_context \\
        --annotator human \\
        --from reports/annotation/qasper_dev_300/annotator_a/completed.jsonl \\
        --ids-from reports/annotation/qasper_dev_300/annotator_a/PROVENANCE.md

Refuses to run if the target slot already holds labels, so a second invocation
cannot wipe work in progress.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HUMAN_FIELDS = ("human_label", "human_confidence", "human_notes")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ids_from_provenance(path: Path, heading: str) -> list[str]:
    """The unit ids listed under one heading of a PROVENANCE.md."""
    text = path.read_text(encoding="utf-8")
    if heading not in text:
        raise SystemExit(f"{path} has no section headed {heading!r}")
    section = text.split(heading, 1)[1].split("\n##", 1)[0]
    return list(dict.fromkeys(re.findall(r"unit_\d{4}", section)))


def question_ids(package: Path) -> dict[str, str]:
    """annotation_id -> question_id, from the package's withheld key.

    Used only to confirm two packages mean the same question by the same id.
    The proposed labels in that file are not read.
    """
    key = package / "proposed_labels_key.jsonl"
    return {r["annotation_id"]: r.get("question_id", "") for r in read_jsonl(key)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, help="package to annotate")
    parser.add_argument("--annotator", default="human", help="new annotator slot id")
    parser.add_argument("--from", dest="source", required=True,
                        help="completed.jsonl holding the labels to carry over")
    parser.add_argument("--ids-from", default="",
                        help="PROVENANCE.md listing which ids to carry over; without "
                             "it, every labelled row in --from is carried")
    parser.add_argument("--heading", default="## Human-annotated",
                        help="heading in --ids-from whose id list to read")
    args = parser.parse_args()

    package = Path(args.package)
    master = package / "annotation_sheet.jsonl"
    if not master.exists():
        print(f"no annotation sheet at {master}", file=sys.stderr)
        return 1

    target = package / f"annotator_{args.annotator}"
    output = target / "completed.jsonl"
    if output.exists() and any(r.get("human_label") for r in read_jsonl(output)):
        print(f"{output} already holds labels — refusing to overwrite them",
              file=sys.stderr)
        return 1

    sheet = read_jsonl(master)
    by_id = {u["annotation_id"]: u for u in sheet}

    source_path = Path(args.source)
    source = {r["annotation_id"]: r for r in read_jsonl(source_path)}

    if args.ids_from:
        wanted = ids_from_provenance(Path(args.ids_from), args.heading)
    else:
        wanted = [i for i, r in source.items() if str(r.get("human_label", "")).strip()]

    # Same id must mean the same question in both packages, or carrying a label
    # across would silently attach it to a different unit.
    here = question_ids(package)
    there = question_ids(source_path.parent.parent)

    carried, skipped = {}, []
    for unit_id in wanted:
        row = source.get(unit_id)
        if row is None:
            skipped.append((unit_id, "not in the source file"))
            continue
        label = str(row.get("human_label", "")).strip()
        if not label:
            skipped.append((unit_id, "no label in the source file"))
            continue
        if unit_id not in by_id:
            skipped.append((unit_id, "not in this package"))
            continue
        if here.get(unit_id) != there.get(unit_id):
            skipped.append((unit_id, "different question under the same id"))
            continue
        carried[unit_id] = {f: row.get(f, "") for f in HUMAN_FIELDS}

    target.mkdir(parents=True, exist_ok=True)
    (target / "annotation_sheet.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in sheet) + "\n",
        encoding="utf-8",
    )

    rows = []
    for unit in sheet:
        row = dict(unit)
        got = carried.get(unit["annotation_id"])
        for field in HUMAN_FIELDS:
            row[field] = got[field] if got else ""
        rows.append(row)
    output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    (target / ".locked_ids.json").write_text(
        json.dumps(
            {
                "note": "Annotations protected from accidental overwrite. Editing one "
                        "through the interface requires an explicit unlock and removes "
                        "it from this list.",
                "carried_from": str(source_path),
                "annotation_ids": sorted(carried),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"package   : {package}")
    print(f"slot      : {target}")
    print(f"carried   : {len(carried)} label(s) from {source_path}")
    print(f"locked    : {len(carried)} unit(s)")
    print(f"remaining : {len(sheet) - len(carried)} unit(s) to annotate")
    if skipped:
        print(f"skipped   : {len(skipped)}")
        for unit_id, why in skipped[:10]:
            print(f"            {unit_id} — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
