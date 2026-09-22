#!/usr/bin/env python
"""Build the adjudication package for the gold-span units the first pass left open.

The first adjudication (``reports/annotation/goldspan_adjudication``) settled 60 of the
87 proxy-unresolved units by hand.  Two groups of the 133 zero-coverage units were never
judged by a person:

  27 unresolved units that the stratified sample of 60 did not draw, and
  10 units that both proxies called "answer present outside the gold span" (bucket B),
     which the estimator counts as under-coverage on the proxies' word alone.

Those 37 units are what this script packages.  Completing them turns the unresolved part
of the population into a census -- no sampling error left -- and replaces the bucket-B
assumption, which the published sensitivity analysis shows is the larger of the two
uncertainties, with observed labels.  The 36 bucket-A units ("answer absent" under both
proxies) stay unchecked and remain an assumption either way.

Nothing here is new annotation policy: the question, the allowed answers, the blinding
and the sheet layout are taken unchanged from ``build_goldspan_adjudication.py``, and the
original package is not touched.  There is no sampling, so there is no seed: every
remaining unit is included.

    python scripts/build_goldspan_remainder.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --audit reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json \
        --first-pass reports/annotation/goldspan_adjudication \
        --out reports/annotation/goldspan_adjudication_remaining37
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_goldspan_adjudication import (  # noqa: E402
    ANSWERS,
    UNRESOLVED,
    read_jsonl,
    render_unit,
    sha256,
)

PROXY_PRESENT = "B_supported_outside_gold_span"


def remaining_ids(audit: dict, already: set[str]) -> dict[str, list[str]]:
    """The units still without a human judgement, grouped by their proxy bucket."""
    groups: dict[str, list[str]] = {}
    for u in audit["units"]:
        bucket, uid = u["bucket"], u["annotation_id"]
        if uid in already:
            continue
        if bucket in UNRESOLVED or bucket == PROXY_PRESENT:
            groups.setdefault(bucket, []).append(uid)
    return {b: sorted(v) for b, v in sorted(groups.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="the full-context annotation package")
    ap.add_argument("--audit", required=True, help="audit/gold_span_semantic.json")
    ap.add_argument("--first-pass", required=True, help="the completed 60-unit package")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg, out, first = Path(args.package), Path(args.out), Path(args.first_pass)
    if out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty - refusing to overwrite")
        return 1

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    already = {i for ids in first_manifest["selected_ids_by_stratum"].values() for i in ids}

    groups = remaining_ids(audit, already)
    selected = [i for ids in groups.values() for i in ids]
    if set(selected) & already:
        print("overlap with the first pass - refusing to build")
        return 1
    if len(selected) != len(set(selected)):
        print("duplicate ids - refusing to build")
        return 1

    # Presentation order is the unit id, not the bucket: the sheet must not let the
    # reviewer infer which group a unit came from (the proxies are what is under test).
    selected = sorted(selected)

    sheet_master = {u["annotation_id"]: u
                    for u in read_jsonl(pkg / "annotation_sheet.jsonl")}
    missing = [i for i in selected if i not in sheet_master]
    if missing:
        print(f"{len(missing)} unit(s) not found in the annotation sheet: {missing[:5]}")
        return 1
    units = [sheet_master[i] for i in selected]

    incomplete = [u["annotation_id"] for u in units
                  if any(c.get("text_complete") is False for c in (u.get("retrieved_context") or []))
                  or any(g.get("text_complete") is False for g in (u.get("gold_evidence") or []))]
    if incomplete:
        print(f"incomplete text in {len(incomplete)} unit(s): {incomplete[:5]}")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    n = len(units)

    body = [render_unit(i, n, u) for i, u in enumerate(units, 1)]
    (out / "review_sheet.txt").write_text(
        f"GOLD-SPAN ADJUDICATION — remaining {n} units\n"
        "Same question, same answers, same rules as the first 60. Read README.md first.\n\n"
        + "\n".join(body), encoding="utf-8")

    lines = [f"annotation_id,answer  # one of {' / '.join(ANSWERS)}"]
    lines += [f"{i}," for i in selected]
    (out / "answers.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    visible = ("annotation_id", "question", "reference_answers", "corpus_can_answer",
               "gold_evidence", "retrieved_context")
    (out / "units.jsonl").write_text(
        "\n".join(json.dumps({k: u.get(k) for k in visible}, ensure_ascii=False)
                  for u in units) + "\n", encoding="utf-8")

    chunks = [c for u in units for c in (u.get("retrieved_context") or [])]
    spans = [g for u in units for g in (u.get("gold_evidence") or [])]
    manifest = {
        "kind": "gold-span adjudication — remaining units (census, not a sample)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "question_put_to_the_annotator":
            "Is the reference answer derivable from the retrieved text alone, without "
            "relying on the annotated gold span?",
        "allowed_answers": list(ANSWERS),
        "relation_to_first_pass": {
            "first_pass": first.as_posix(),
            "first_pass_units": len(already),
            "overlap_with_first_pass": 0,
            "note": "The same reviewer (the author) judged the first pass. This package is "
                    "an extension of that pass, not an independent second annotation, and "
                    "yields no inter-annotator agreement.",
        },
        "population": {
            "zero_gold_span_coverage_units": audit["n_units"],
            "already_adjudicated": len(already),
            "remaining_without_a_human_judgement": n,
            "group_sizes": {b: len(v) for b, v in groups.items()},
            "still_not_adjudicated_after_this_pass": {
                "A_genuinely_unsupported": len([u for u in audit["units"]
                                                if u["bucket"] == "A_genuinely_unsupported"]),
                "note": "Units both proxies called 'answer absent'. Counted as no "
                        "under-coverage; this remains an assumption after this pass.",
            },
        },
        "selection": {
            "design": "census of every zero-coverage unit that has no human judgement, "
                      "i.e. the 27 unresolved units the first sample did not draw plus "
                      "the 10 bucket-B units counted on the proxies' word",
            "sampling": "none - all remaining units are included, so there is no seed",
            "presentation_order": "by unit id, so group membership is not inferable",
            "reproduce": f"python scripts/build_goldspan_remainder.py --package "
                         f"{pkg.as_posix()} --audit {Path(args.audit).as_posix()} "
                         f"--first-pass {first.as_posix()} --out {out.as_posix()}",
        },
        "ids_by_group": groups,
        "blinding": {
            "fields_shown": list(visible),
            "withheld": ["human_label", "human_confidence", "human_notes",
                         "reference/automated annotation label", "taxonomy labels",
                         "proxy scores", "bucket name", "first-pass answers"],
            "note": "Group membership is not shown and units are ordered by id, so a unit's "
                    "proxy bucket - including whether the proxies called it under-coverage - "
                    "cannot be inferred from the sheet.",
        },
        "integrity": {
            "retrieved_chunks": len(chunks),
            "retrieved_chunks_complete": sum(1 for c in chunks if c.get("text_complete")),
            "gold_spans": len(spans),
            "gold_spans_complete": sum(1 for g in spans if g.get("text_complete")),
            "source_sheet_sha256": sha256(pkg / "annotation_sheet.jsonl"),
            "audit_file_sha256": sha256(Path(args.audit)),
        },
        "does_not_modify": [
            (first / "answers.csv").as_posix(),
            (first / "manifest.json").as_posix(),
            (pkg / "annotator_human" / "completed.jsonl").as_posix(),
            (pkg / "final_human_reviewed" / "completed.jsonl").as_posix(),
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    readme = f"""# Gold-span adjudication — the remaining {n} units

## What this is

The first pass settled 60 of the 87 proxy-unresolved units. This package holds every
zero-coverage unit that still has no human judgement:

- **27** unresolved units the stratified sample did not draw;
- **10** units both proxies called "answer present outside the gold span", which the
  published estimate counts as under-coverage without anyone having checked them.

Which of the two groups a unit belongs to is **not** shown, and the units are ordered by
id, so the sheet gives nothing away.

## The question — unchanged from the first pass

> **Is the reference answer derivable from the retrieved text alone,
> without relying on the annotated gold span?**

- **YES** — someone reading only the retrieved text could produce the reference answer.
- **NO** — the retrieved text does not contain what is needed.
- **CANNOT_TELL** — you genuinely cannot decide. Use it freely; it is handled properly.

Judge from the retrieved text only. The gold span is shown for context and was *not*
given to the system; if the answer is only in the gold span, that is a **NO**.
"Derivable" means a careful reader could get there — stated outright or a short
inference away — not guessable from topic or prior knowledge.

## How to do it

1. Open `review_sheet.txt` — all {n} units, full text, no truncation.
2. Open `answers.csv` — the {n} ids are already listed in the same order.
3. Write `YES`, `NO` or `CANNOT_TELL` after each comma. Nothing else to fill in.

```
unit_0004,NO
unit_0011,YES
```

Stop and resume whenever you like.

## What happens next

```bash
python scripts/score_goldspan_adjudication.py \\
    --package {first.as_posix()} \\
    --remainder {out.as_posix()} \\
    --audit {Path(args.audit).as_posix()} \\
    --out {out.as_posix()}/estimate_combined.json
```

With these {n} answers in place the 87 unresolved units become a **census**: the
sampling error disappears, and the bucket-B assumption is replaced by observed labels.
What remains assumed is the 36 units both proxies called "answer absent", which no pass
has checked.

## What this is not

The same person judged the first 60 and will judge these {n}. This is an extension of
the author's own adjudication — **not** an independent second annotation, and it yields
no inter-annotator agreement. The single-annotator limitation stands.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"wrote {out} — {n} units")
    for b, ids in groups.items():
        print(f"  {b}: {len(ids)}")
    print(f"  retrieved chunks complete: {manifest['integrity']['retrieved_chunks_complete']}"
          f"/{manifest['integrity']['retrieved_chunks']}")
    print(f"  gold spans complete:       {manifest['integrity']['gold_spans_complete']}"
          f"/{manifest['integrity']['gold_spans']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
