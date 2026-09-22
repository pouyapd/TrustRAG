#!/usr/bin/env python
"""Render a gold-span adjudication package as a human-reviewable document.

``build_goldspan_remainder.py`` writes a fixed-width ``review_sheet.txt``. This script
renders the same units as Markdown with a label and a rationale field under each one, and
adds two things the plain sheet does not carry:

  * where each gold span sits relative to what was retrieved (same document or not,
    character overlap, nearest gap), and
  * the document text surrounding each gold span, so the reviewer can see what else the
    paper says near the annotated evidence.

That surrounding text is reference material, not evidence. The question put to the
reviewer is unchanged -- is the reference answer derivable from the *retrieved* text
alone -- and the neighbourhood section is fenced off and marked accordingly, because a
unit is a YES only when the retrieved chunks support the answer.

Nothing is inferred, scored or pre-filled: every label field is written blank. Unit order
is taken from ``units.jsonl`` unchanged, which is by unit id, so the proxy bucket a unit
came from stays unguessable.

    python scripts/render_goldspan_review.py \
        --package reports/annotation/goldspan_adjudication_remaining37 \
        --corpus data/raw/qasper-dev-v0.3.json \
        --out reports/annotation/goldspan_adjudication_remaining37/HUMAN_REVIEW.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.loaders.qasper import QasperLoader  # noqa: E402

WINDOW = 1500          # characters of document text shown either side of a gold span
GOLD_OPEN = "⟦GOLD SPAN STARTS⟧"
GOLD_CLOSE = "⟦GOLD SPAN ENDS⟧"
BLANK = "MY LABEL: __________"


def load_units(pkg: Path) -> list[dict]:
    return [json.loads(line) for line
            in (pkg / "units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def fence(text: str) -> str:
    """Fence a block of document text, widening the fence if the text contains one."""
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}text\n{text}\n{ticks}"


def positions(unit: dict) -> list[str]:
    """How each gold span relates to the chunks that were actually retrieved."""
    out: list[str] = []
    chunks = unit.get("retrieved_context") or []
    for n, g in enumerate(unit.get("gold_evidence") or [], 1):
        glo, ghi = g["char_range"]
        same = [c for c in chunks if c["doc_id"] == g["doc_id"]]
        if not same:
            docs = sorted({c["doc_id"] for c in chunks})
            shown = ", ".join(f"`{d}`" for d in docs) or "nothing"
            out.append(f"- **Gold span {n}** is in `{g['doc_id']}` at chars [{glo}, {ghi}). "
                       f"No retrieved chunk comes from that document (retrieval returned "
                       f"{shown}), so nothing retrieved overlaps it.")
            continue
        parts = []
        for c in same:
            lo, hi = c["char_range"]
            overlap = max(0, min(ghi, hi) - max(glo, lo))
            if overlap:
                parts.append(f"rank {c['rank']} overlaps it by {overlap} chars")
            elif lo > ghi:
                parts.append(f"rank {c['rank']} starts {lo - ghi} chars after it ends")
            else:
                parts.append(f"rank {c['rank']} ends {glo - hi} chars before it starts")
        out.append(f"- **Gold span {n}** is in `{g['doc_id']}` at chars [{glo}, {ghi}). "
                   f"Chunks from the same document: " + "; ".join(parts) + ".")
    return out


def neighbourhoods(unit: dict, docs: dict) -> list[tuple[str, int, int, str]]:
    """Merged windows of document text around this unit's gold spans, spans marked."""
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for g in unit.get("gold_evidence") or []:
        by_doc.setdefault(g["doc_id"], []).append((g["char_range"][0], g["char_range"][1]))

    blocks: list[tuple[str, int, int, str]] = []
    for doc_id, spans in sorted(by_doc.items()):
        doc = docs.get(doc_id)
        if doc is None:
            continue
        merged: list[tuple[int, int, list[tuple[int, int]]]] = []
        for lo, hi in sorted(spans):
            wlo, whi = max(0, lo - WINDOW), min(len(doc.text), hi + WINDOW)
            if merged and wlo <= merged[-1][1]:
                plo, phi, held = merged.pop()
                merged.append((plo, max(phi, whi), held + [(lo, hi)]))
            else:
                merged.append((wlo, whi, [(lo, hi)]))
        for wlo, whi, held in merged:
            text = doc.text[wlo:whi]
            for lo, hi in sorted(held, reverse=True):    # from the end: offsets stay valid
                text = text[: hi - wlo] + GOLD_CLOSE + text[hi - wlo :]
                text = text[: lo - wlo] + GOLD_OPEN + text[lo - wlo :]
            blocks.append((doc_id, wlo, whi, text))
    return blocks


def render(unit: dict, n: int, total: int, docs: dict) -> str:
    L: list[str] = []
    add = L.append
    add(f"## Unit {n} of {total} — `{unit['annotation_id']}`\n")
    add("**QASPER question**\n")
    add(f"> {unit['question']}\n")
    add("**Reference answer(s)** — what the dataset says the answer is\n")
    for a in unit.get("reference_answers") or ["(none recorded)"]:
        add(f"- {a}")
    add("")
    add(f"**Corpus can answer:** {unit.get('corpus_can_answer')}\n")

    chunks = unit.get("retrieved_context") or []
    add("### 1. Retrieved text — judge from this\n")
    add(f"{len(chunks)} chunk(s), complete, exactly as the retriever returned them and the "
        "reader saw them. No excerpting.\n")
    for c in chunks:
        lo, hi = c["char_range"]
        add(f"**Chunk rank {c['rank']}** · `{c['chunk_id']}` · document `{c['doc_id']}` · "
            f"source `{c['source']}` · chars [{lo}, {hi}) · {c['n_chars']} chars · "
            f"complete={c['text_complete']}\n")
        add(fence(c.get("text", "")) + "\n")

    add("### 2. Annotated gold span(s) — context only, NOT retrieved for this unit\n")
    add("The system never saw this text. If the answer is only here, the unit is a **NO**.\n")
    for i, g in enumerate(unit.get("gold_evidence") or [], 1):
        lo, hi = g["char_range"]
        add(f"**Gold span {i}** · document `{g['doc_id']}` · chars [{lo}, {hi}) · "
            f"{g['n_chars']} chars · complete={g['text_complete']}\n")
        add(fence(g.get("text", "")) + "\n")

    add("### 3. Where the gold span sits relative to what was retrieved\n")
    for line in positions(unit):
        add(line)
    add("")

    add("### 4. Surrounding document text — reference only, NOT the basis for the label\n")
    add(f"Up to {WINDOW} characters either side of each gold span, from the same paper. "
        "Retrieval did **not** return this text. It is here so you can see what the paper "
        "says around the annotated evidence; a unit is a **YES** only when section 1 "
        "supports the answer.\n")
    for doc_id, wlo, whi, text in neighbourhoods(unit, docs):
        add(f"**Document `{doc_id}`, chars [{wlo}, {whi})** — gold span marked "
            f"`{GOLD_OPEN}` … `{GOLD_CLOSE}`\n")
        add(fence(text) + "\n")

    add("### Your decision\n")
    add("> Is the reference answer derivable from the **retrieved text** (section 1) alone, "
        "without relying on the annotated gold span?\n")
    add("```")
    add(f"UNIT: {unit['annotation_id']}")
    add(f"{BLANK}          Allowed values: YES / NO / CANNOT_TELL")
    add("RATIONALE: __________")
    add("```\n")
    add("---\n")
    return "\n".join(L)


def header(pkg: Path, manifest: dict, units: list[dict]) -> str:
    total = len(units)
    rows = "\n".join(
        f"| {i} | `{u['annotation_id']}` | {len(u.get('retrieved_context') or [])} | "
        f"{len(u.get('gold_evidence') or [])} | {u['question'][:70]}"
        f"{'…' if len(u['question']) > 70 else ''} |"
        for i, u in enumerate(units, 1))
    return f"""# Gold-span adjudication — human review sheet ({total} units)

Generated {datetime.now(UTC).isoformat(timespec="seconds")} by
`scripts/render_goldspan_review.py` from `{pkg.as_posix()}`.

**No label in this file is filled in, suggested or inferred.** Every label field is blank
and is yours to complete.

## The question — unchanged from the first pass

> **{manifest["question_put_to_the_annotator"]}**

- **YES** — someone reading only the retrieved text (section 1 of a unit) could produce
  the reference answer.
- **NO** — the retrieved text does not contain what is needed.
- **CANNOT_TELL** — you genuinely cannot decide. Use it freely; it is handled properly and
  never silently dropped.

Judge from the retrieved text only. "Derivable" means a careful reader could get there —
stated outright, or a short inference away — not guessable from the topic or from prior
knowledge. If the answer appears only in the gold span, or only in the surrounding
document text of section 4, that is a **NO**: neither of those was retrieved.

## What each unit shows

| Section | Contents | Counts towards the label? |
|---|---|---|
| Header | unit id, question, reference answer(s) | — |
| 1 | every retrieved chunk in full, with chunk id, document id, source and exact character range | **yes — this is the evidence** |
| 2 | the annotated gold span(s) in full, with document id and exact character range | no — context only |
| 3 | whether any retrieved chunk came from the gold document, and how far it sits from the span | no — orientation |
| 4 | up to {WINDOW} characters of the paper either side of each gold span | no — reference only |

Section 4 answers "does this information also appear outside the annotated gold span?" for
the *document*. The label answers the narrower question about the *retrieved* text, which
is what the under-coverage estimate is about.

## What is deliberately not shown

The proxy bucket each unit came from, the proxy scores, the taxonomy labels, the reader's
generated answer, and the first pass's answers. Units are ordered by unit id, so group
membership cannot be inferred from position. This is the same blinding as the first pass.

## How to record your labels

Write your decision into each label field below, then copy the {total} labels into
`answers.csv` in this directory — the ids are already listed there in this same order.
`answers.csv` has not been touched.

## Provenance

This is an extension of the author's own adjudication of the first 60 units, by the same
reviewer. It is **not** an independent second annotation, it yields **no** inter-annotator
agreement, and it does not remove the single-annotator limitation.

## Index

| # | Unit | Chunks | Gold spans | Question |
|---:|---|---:|---:|---|
{rows}

---

"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="an adjudication package directory")
    ap.add_argument("--corpus", required=True, help="the raw QASPER split the units come from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg, out = Path(args.package), Path(args.out)
    units = load_units(pkg)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))

    docs = {d.doc_id: d for d in QasperLoader().load(Path(args.corpus), split="dev").documents}

    # Every stored piece of text must sit at its recorded offsets in the loaded document,
    # otherwise the surrounding text shown below would be the wrong part of the paper.
    checked = 0
    for u in units:
        for piece in (u.get("retrieved_context") or []) + (u.get("gold_evidence") or []):
            doc = docs.get(piece["doc_id"])
            if doc is None:
                print(f"{u['annotation_id']}: document {piece['doc_id']} is not in the corpus")
                return 1
            lo, hi = piece["char_range"]
            if doc.text[lo:hi] != piece["text"]:
                print(f"{u['annotation_id']}: text at [{lo}, {hi}) in {piece['doc_id']} "
                      "does not match the package - refusing to render")
                return 1
            checked += 1

    body = [render(u, i, len(units), docs) for i, u in enumerate(units, 1)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header(pkg, manifest, units) + "\n".join(body),
                   encoding="utf-8", newline="\n")

    print(f"wrote {out} - {len(units)} units, {checked} text pieces verified against the corpus")
    print(f"  blank label fields: {out.read_text(encoding='utf-8').count(BLANK)}")
    print(f"  size: {out.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
