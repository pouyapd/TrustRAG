"""Audit whether an annotation package showed annotators the whole retrieved chunk.

Step 2 of `docs/ANNOTATION_GUIDELINES.md` asks the annotator whether the
evidence reached the system. That question is unanswerable if the sheet shows a
prefix of each chunk: text past the cut is indistinguishable from text that was
never retrieved, and the natural reading of a chunk that stops mid-sentence is
"the fact did not arrive". The first `qasper_dev_300` package stored
`chunk.text[:600]`, so 1418 of its 1500 chunks were excerpts.

This script does not guess. For every chunk in a package it looks up the chunk
the retriever actually returned, keyed by document and character offsets, in the
source `inference.jsonl`, and reports one of:

    already_complete   the package held the whole chunk to begin with
    recovered          the package held a prefix; the full chunk exists in the
                       source records and a rebuilt package now shows it
    unreconstructable  no chunk with those offsets is in the source records, so
                       what the annotator saw cannot be checked

A chunk shorter than 600 characters is not evidence of truncation and a chunk
longer than 600 is not evidence of the opposite; the only test that means
anything is the stored text against the recorded `char_range`.

    python scripts/audit_annotation_truncation.py \
        --records reports/experiments/qasper_dev_300/inference.jsonl \
        --old-package reports/annotation/qasper_dev_300 \
        --new-package reports/annotation/qasper_dev_300_full_context \
        --out reports/annotation/qasper_dev_300_full_context/TRUNCATION_AUDIT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance

#: The constant the original builder sliced with. Recorded so the audit can say
#: which chunks were cut by *that* rule rather than merely being short.
LEGACY_LIMIT = 600


def norm(text: str) -> str:
    """Collapse whitespace so a probe is not defeated by line wrapping alone."""
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_chunks(records_path: Path) -> dict[tuple[str, int, int], str]:
    """Every retrieved chunk in the source run, keyed by document and offsets.

    Offsets are the key rather than the text: a document that repeats itself
    makes text lookup ambiguous, and the offsets are exactly what the package
    records.
    """
    index: dict[tuple[str, int, int], str] = {}
    for record in read_jsonl(records_path):
        for chunk in record.get("retrieved", []):
            start, end = chunk.get("start_char"), chunk.get("end_char")
            if start is None or end is None:
                continue
            index[(str(chunk.get("doc_id")), int(start), int(end))] = str(
                chunk.get("text", "")
            )
    return index


def audit_chunk(chunk: dict, source: dict, new_chunk: dict | None) -> dict:
    """One chunk's before/after, with the reason it is classified that way."""
    span = chunk.get("char_range") or [None, None]
    old_len = len(str(chunk.get("text", "")))
    entry = {
        "rank": chunk.get("rank"),
        "doc_id": chunk.get("doc_id"),
        "char_range": span,
        "char_range_length": None,
        "old_displayed_chars": old_len,
        "new_displayed_chars": None,
        "was_truncated": None,
        "cut_at_legacy_limit": False,
        "status": "unreconstructable",
    }
    if span[0] is None or span[1] is None:
        entry["status"] = "unreconstructable"
        entry["reason"] = "chunk has no char_range, so completeness cannot be judged"
        return entry

    length = int(span[1]) - int(span[0])
    entry["char_range_length"] = length
    entry["was_truncated"] = old_len < length
    entry["cut_at_legacy_limit"] = old_len == LEGACY_LIMIT and length > LEGACY_LIMIT

    key = (str(chunk.get("doc_id")), int(span[0]), int(span[1]))
    full = source.get(key)
    if full is None:
        entry["reason"] = "no chunk with these offsets in the source records"
        return entry

    entry["source_chars"] = len(full)
    if new_chunk is not None:
        entry["new_displayed_chars"] = len(str(new_chunk.get("text", "")))
    elif not entry["was_truncated"]:
        entry["new_displayed_chars"] = old_len

    if not entry["was_truncated"]:
        entry["status"] = "already_complete"
        entry["reason"] = "stored text already covered the whole char_range"
    elif entry["new_displayed_chars"] == length:
        entry["status"] = "recovered"
        entry["reason"] = (
            f"prefix of {old_len} chars replaced by the complete {length}-char chunk"
        )
    else:
        entry["status"] = "unreconstructable"
        entry["reason"] = (
            "the full chunk exists in the source records but the rebuilt package "
            "does not show it"
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit annotation-package truncation")
    parser.add_argument("--records", required=True, help="source inference.jsonl")
    parser.add_argument("--old-package", required=True, help="package to audit")
    parser.add_argument("--new-package", default="", help="rebuilt package, if any")
    parser.add_argument("--out", required=True, help="where to write the audit JSON")
    parser.add_argument("--verify-unit", action="append", default=[],
                        help="annotation_id to report in full detail (repeatable)")
    args = parser.parse_args()

    records_path = Path(args.records)
    old_sheet_path = Path(args.old_package) / "annotation_sheet.jsonl"
    for path in (records_path, old_sheet_path):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    source = source_chunks(records_path)
    old_units = {u["annotation_id"]: u for u in read_jsonl(old_sheet_path)}
    new_units: dict[str, dict] = {}
    if args.new_package:
        new_sheet = Path(args.new_package) / "annotation_sheet.jsonl"
        if not new_sheet.exists():
            print(f"missing {new_sheet}", file=sys.stderr)
            return 1
        new_units = {u["annotation_id"]: u for u in read_jsonl(new_sheet)}

    per_unit, statuses = [], Counter()
    affected_ids, unreconstructable_ids = [], []
    total_chunks = 0
    old_chars = new_chars = 0

    for unit_id, unit in sorted(old_units.items()):
        new_unit = new_units.get(unit_id)
        new_by_rank = {
            c.get("rank"): c for c in (new_unit or {}).get("retrieved_context", [])
        }
        chunk_entries = []
        for chunk in unit.get("retrieved_context", []):
            entry = audit_chunk(chunk, source, new_by_rank.get(chunk.get("rank")))
            chunk_entries.append(entry)
            statuses[entry["status"]] += 1
            total_chunks += 1
            old_chars += entry["old_displayed_chars"]
            new_chars += entry["new_displayed_chars"] or entry["old_displayed_chars"]
        truncated = [c for c in chunk_entries if c["was_truncated"]]
        if truncated:
            affected_ids.append(unit_id)
        if any(c["status"] == "unreconstructable" for c in chunk_entries):
            unreconstructable_ids.append(unit_id)
        per_unit.append(
            {
                "annotation_id": unit_id,
                "question_matches_rebuild": (
                    None if new_unit is None else new_unit["question"] == unit["question"]
                ),
                "n_chunks": len(chunk_entries),
                "n_truncated": len(truncated),
                "chunks": chunk_entries,
            }
        )

    verified = {}
    for unit_id in args.verify_unit:
        unit = old_units.get(unit_id)
        if unit is None:
            verified[unit_id] = {"error": "not in the audited package"}
            continue
        new_unit = new_units.get(unit_id)
        detail = next(u for u in per_unit if u["annotation_id"] == unit_id)
        old_display = {
            c.get("rank"): norm(str(c.get("text", "")))
            for c in unit["retrieved_context"]
        }
        full_text = {
            c.get("rank"): norm(str(c.get("text", "")))
            for c in (new_unit or {}).get("retrieved_context", [])
        }
        probes = [("system_answer", str(unit.get("system_answer", "")))]
        probes += [("reference_answer", str(r)) for r in unit.get("reference_answers", [])]

        findings = []
        for kind, raw in probes:
            needle = norm(raw)[:160]
            if len(needle) < 25:
                continue
            in_old = [r for r, t in old_display.items() if needle and needle in t]
            in_full = [r for r, t in full_text.items() if needle and needle in t]
            findings.append(
                {
                    "probe_kind": kind,
                    "probe": needle,
                    "found_in_old_display_ranks": in_old,
                    "found_in_full_chunk_ranks": in_full,
                    "hidden_by_legacy_limit": bool(in_full) and not in_old,
                }
            )
        verified[unit_id] = {
            "chunks": [
                {
                    "rank": c["rank"],
                    "doc_id": c["doc_id"],
                    "char_range": c["char_range"],
                    "old_displayed_chars": c["old_displayed_chars"],
                    "new_displayed_chars": c["new_displayed_chars"],
                    "status": c["status"],
                }
                for c in detail["chunks"]
            ],
            "probes": findings,
            "evidence_hidden_by_600_char_limit": any(
                f["hidden_by_legacy_limit"] for f in findings
            ),
            "hidden_probe_kinds": sorted(
                {f["probe_kind"] for f in findings if f["hidden_by_legacy_limit"]}
            ),
        }

    report = {
        "records": str(records_path),
        "audited_package": str(Path(args.old_package)),
        "rebuilt_package": str(Path(args.new_package)) if args.new_package else None,
        "legacy_display_limit": LEGACY_LIMIT,
        "totals": {
            "annotation_units": len(old_units),
            "retrieved_chunks": total_chunks,
            "chunks_previously_truncated": sum(
                1 for u in per_unit for c in u["chunks"] if c["was_truncated"]
            ),
            "chunks_cut_at_legacy_limit": sum(
                1 for u in per_unit for c in u["chunks"] if c["cut_at_legacy_limit"]
            ),
            "chunks_already_complete": statuses["already_complete"],
            "chunks_now_complete": statuses["already_complete"] + statuses["recovered"],
            "chunks_recovered": statuses["recovered"],
            "chunks_unreconstructable": statuses["unreconstructable"],
            "affected_units": len(affected_ids),
            "unreconstructable_units": len(unreconstructable_ids),
            "displayed_chars_before": old_chars,
            "displayed_chars_after": new_chars,
            "chars_previously_hidden": new_chars - old_chars,
        },
        "affected_unit_ids": affected_ids,
        "unreconstructable_unit_ids": unreconstructable_ids,
        "verified_units": verified,
        "per_unit": per_unit,
        "provenance": collect_provenance(source_records=str(records_path)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = report["totals"]
    print(f"audited {t['annotation_units']} units, {t['retrieved_chunks']} chunks")
    print(f"  previously truncated : {t['chunks_previously_truncated']}"
          f" (cut at the {LEGACY_LIMIT}-char limit: {t['chunks_cut_at_legacy_limit']})")
    print(f"  now complete         : {t['chunks_now_complete']}"
          f" (recovered {t['chunks_recovered']})")
    print(f"  unreconstructable    : {t['chunks_unreconstructable']}")
    print(f"  affected units       : {t['affected_units']}")
    print(f"  characters recovered : {t['chars_previously_hidden']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
