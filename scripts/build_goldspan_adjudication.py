#!/usr/bin/env python
"""Build the human adjudication package for the unresolved gold-span cases.

The semantic audit left 87 of 133 zero-coverage units unresolved: two weak proxies
either disagreed or both sat mid-range. Only a human can settle whether the answer
was derivable from what the retriever actually returned. This draws a stratified
sample of those 87 and writes a self-contained review package.

The annotator sees the question, the reference answers, the full retrieved text and
the gold spans -- and nothing else. No previous label, no taxonomy verdict, no proxy
score, no bucket name reaches the sheet: those are exactly the signals whose validity
is under test, and showing them would contaminate the judgement.

    python scripts/build_goldspan_adjudication.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --audit reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json \
        --n 60 --seed 20260907 \
        --out reports/annotation/goldspan_adjudication
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
from datetime import UTC, datetime
from pathlib import Path

UNRESOLVED = ("C_possibly_inferable", "D_ambiguous_lexical_only", "D_ambiguous")
ANSWERS = ("YES", "NO", "CANNOT_TELL")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation that sums exactly to `total`."""
    pool = sum(counts.values())
    exact = {k: v * total / pool for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: exact[k] - alloc[k], reverse=True):
        if sum(alloc.values()) >= total:
            break
        alloc[k] += 1
    return alloc


def render_unit(n: int, total: int, unit: dict) -> str:
    """One unit, as the annotator sees it. Full text, no truncation."""
    L: list[str] = []
    add = L.append
    add(f"{'=' * 92}")
    add(f"UNIT {n} of {total}   ·   id: {unit['annotation_id']}")
    add(f"{'=' * 92}\n")
    add("QUESTION")
    add(textwrap.fill(unit["question"], 92, initial_indent="  ", subsequent_indent="  "))
    add("")
    add("REFERENCE ANSWER(S) — what the dataset says the answer is")
    for a in unit.get("reference_answers") or ["(none recorded)"]:
        add(textwrap.fill(str(a), 92, initial_indent="  - ", subsequent_indent="    "))
    add("")
    add("-" * 92)
    add("RETRIEVED TEXT — this is what the system actually returned. Judge from THIS.")
    add("-" * 92)
    for c in unit.get("retrieved_context") or []:
        add(f"\n  [retrieved chunk, rank {c.get('rank')}] {c['doc_id']} "
            f"chars [{c['char_range'][0]}, {c['char_range'][1]}) "
            f"— {len(c.get('text', ''))} chars, complete={c.get('text_complete')}")
        add(textwrap.fill(c.get("text", ""), 92, initial_indent="  ", subsequent_indent="  "))
    add("")
    add("-" * 92)
    add("ANNOTATED GOLD SPAN(S) — the dataset's marked evidence. NOT retrieved for this")
    add("unit. Shown for context only; do NOT use it to answer the question below.")
    add("-" * 92)
    for g in unit.get("gold_evidence") or []:
        add(f"\n  [gold span, NOT retrieved] {g['doc_id']} "
            f"chars [{g['char_range'][0]}, {g['char_range'][1]}) "
            f"— {len(g.get('text', ''))} chars, complete={g.get('text_complete')}")
        add(textwrap.fill(g.get("text", ""), 92, initial_indent="  ", subsequent_indent="  "))
    add("")
    add("-" * 92)
    add("  QUESTION FOR YOU:")
    add("  Is the reference answer derivable from the RETRIEVED TEXT alone,")
    add("  without relying on the annotated gold span?")
    add("")
    add("  YES / NO / CANNOT_TELL")
    add("-" * 92)
    add("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg, out = Path(args.package), Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty - refusing to overwrite")
        return 1

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    unresolved = [u for u in audit["units"] if u["bucket"] in UNRESOLVED]
    by_bucket: dict[str, list[str]] = {}
    for u in unresolved:
        by_bucket.setdefault(u["bucket"], []).append(u["annotation_id"])

    sizes = {b: len(v) for b, v in by_bucket.items()}
    alloc = largest_remainder(sizes, args.n)

    rng = random.Random(args.seed)
    selected: list[str] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket])
        selected.extend(rng.sample(pool, alloc[bucket]))
    # Present in a shuffled order so bucket membership is not inferable from position.
    rng.shuffle(selected)

    sheet_master = {u["annotation_id"]: u
                    for u in read_jsonl(pkg / "annotation_sheet.jsonl")}
    units = [sheet_master[i] for i in selected]

    out.mkdir(parents=True, exist_ok=True)

    # --- the readable sheet ------------------------------------------------
    body = [render_unit(i, len(units), u) for i, u in enumerate(units, 1)]
    (out / "review_sheet.txt").write_text(
        "GOLD-SPAN ADJUDICATION — 60 units\n"
        "Read the instructions in README.md first. Record answers in answers.csv.\n\n"
        + "\n".join(body), encoding="utf-8")

    # --- the answer file the annotator fills in ----------------------------
    lines = ["annotation_id,answer  # one of YES / NO / CANNOT_TELL"]
    lines += [f"{i}," for i in selected]
    (out / "answers.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- machine-readable copy of exactly what was shown -------------------
    visible = ("annotation_id", "question", "reference_answers", "corpus_can_answer",
               "gold_evidence", "retrieved_context")
    (out / "units.jsonl").write_text(
        "\n".join(json.dumps({k: u.get(k) for k in visible}, ensure_ascii=False)
                  for u in units) + "\n", encoding="utf-8")

    # --- manifest ----------------------------------------------------------
    chunks = [c for u in units for c in (u.get("retrieved_context") or [])]
    spans = [g for u in units for g in (u.get("gold_evidence") or [])]
    manifest = {
        "kind": "gold-span adjudication sample",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "question_put_to_the_annotator":
            "Is the reference answer derivable from the retrieved text alone, without "
            "relying on the annotated gold span?",
        "allowed_answers": list(ANSWERS),
        "population": {
            "zero_gold_span_coverage_units": audit["n_units"],
            "resolved_by_proxy_agreement": audit["n_units"] - len(unresolved),
            "unresolved_eligible_for_sampling": len(unresolved),
            "stratum_sizes": sizes,
        },
        "sampling": {
            "design": "stratified by proxy-agreement bucket, proportional allocation "
                      "with largest-remainder rounding, simple random draw within "
                      "stratum, then shuffled for presentation",
            "seed": args.seed,
            "n_requested": args.n,
            "n_selected": len(selected),
            "allocation": alloc,
            "reproduce": f"python scripts/build_goldspan_adjudication.py --package "
                         f"{pkg.as_posix()} --audit {Path(args.audit).as_posix()} "
                         f"--n {args.n} --seed {args.seed} --out {out.as_posix()}",
        },
        "selected_ids_by_stratum": {
            b: sorted(i for i in selected if i in set(by_bucket[b])) for b in sorted(by_bucket)
        },
        "blinding": {
            "fields_shown": list(visible),
            "withheld": ["human_label", "human_confidence", "human_notes",
                         "reference/automated annotation label", "taxonomy labels",
                         "proxy scores", "bucket name"],
            "note": "Bucket membership is not shown and unit order is shuffled, so the "
                    "stratum cannot be inferred from the sheet.",
        },
        "integrity": {
            "retrieved_chunks": len(chunks),
            "retrieved_chunks_complete": sum(1 for c in chunks if c.get("text_complete") is True),
            "gold_spans": len(spans),
            "gold_spans_complete": sum(1 for g in spans if g.get("text_complete") is True),
            "source_sheet_sha256": sha256(pkg / "annotation_sheet.jsonl"),
            "audit_file_sha256": sha256(Path(args.audit)),
        },
        "does_not_modify": [
            (pkg / "annotator_human" / "completed.jsonl").as_posix(),
            (pkg / "final_human_reviewed" / "completed.jsonl").as_posix(),
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"package : {out.as_posix()}")
    print(f"selected: {len(selected)} of {len(unresolved)} unresolved units")
    print(f"strata  : {alloc}")
    print(f"context : {manifest['integrity']['retrieved_chunks_complete']}/"
          f"{manifest['integrity']['retrieved_chunks']} retrieved chunks complete, "
          f"{manifest['integrity']['gold_spans_complete']}/"
          f"{manifest['integrity']['gold_spans']} gold spans complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
