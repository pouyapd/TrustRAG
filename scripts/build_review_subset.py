#!/usr/bin/env python
"""Build a focused review package from units flagged by the annotation audit.

Copies the flagged units -- unchanged -- into a standalone package that the
existing annotation tool can serve, and writes a dossier holding everything a
reviewer needs that the blinded interface deliberately does not show: the
automated reference label, what the taxonomy said, and why the audit flagged the
unit.

The source package is opened read-only. Nothing is written back to it, and the
labels carried into the review package are exact copies.

    python scripts/build_review_subset.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --annotator human \
        --audit reports/annotation/qasper_dev_300_full_context/audit/human_annotation_audit.json \
        --verdict likely_inconsistent \
        --out reports/annotation/review_43_flagged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

HUMAN_FIELDS = ("human_label", "human_confidence", "human_notes")
REVIEW_SLOT = "review"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_integrity(units: list[dict]) -> dict:
    """Confirm every retrieved chunk and gold span is stored complete."""
    tally = {"retrieved_chunks": 0, "retrieved_complete": 0,
             "gold_spans": 0, "gold_complete": 0, "max_chunk_chars": 0}
    for unit in units:
        for chunk in unit.get("retrieved_context") or []:
            tally["retrieved_chunks"] += 1
            tally["max_chunk_chars"] = max(tally["max_chunk_chars"], len(chunk.get("text", "")))
            if chunk.get("text_complete") is True:
                tally["retrieved_complete"] += 1
        for span in unit.get("gold_evidence") or []:
            tally["gold_spans"] += 1
            if span.get("text_complete") is True:
                tally["gold_complete"] += 1
    return tally


def render_dossier(entries: list[dict], source: Path, verdict: str) -> str:
    L: list[str] = []
    add = L.append
    source = source.as_posix()
    add(f"# Review dossier — {len(entries)} flagged unit(s)\n")
    add("Everything a reviewer needs in one place, including the material the blinded "
        "annotation interface does not show. **The labels below are copies. Reviewing a "
        "unit here changes nothing in the original 200-unit dataset.**\n")
    add(f"Source package: `{source}` · audit verdict filtered on: `{verdict}`\n")
    add("> The automated reference label is shown for context only. It is a language-model "
        "pass, not ground truth. If you would rather review blind, work in the annotation "
        "UI and consult this file afterwards.\n")
    add("## Index\n")
    add("| Unit | Human label | Reference label | Audit pattern |")
    add("|---|---|---|---|")
    for e in entries:
        add(f"| `{e['annotation_id']}` | `{e['human_label']}` | "
            f"`{e['reference_label'] or '—'}` | {e['audit_pattern']} |")
    add("")
    for e in entries:
        f = e["facts"]
        add(f"---\n\n## `{e['annotation_id']}`\n")
        add(f"**Question.** {e['question']}\n")
        add(f"- **Human label:** `{e['human_label']}` "
            f"(confidence: `{e['human_confidence'] or 'none'}`)")
        add(f"- **Automated reference label:** `{e['reference_label'] or '—'}`")
        add(f"- **Taxonomy:** document-gated `{e['taxonomy_document_gated']}`, "
            f"evidence-gated `{e['taxonomy_evidence_gated']}`")
        if e["human_notes"].strip():
            add(f"- **Your note:** {e['human_notes'].strip()}")
        add(f"- **Answerable:** `{f['corpus_can_answer']}` · "
            f"**gold spans covered by retrieval:** {f['n_gold_spans_covered']}/{f['n_gold_spans']} · "
            f"`evidence_status`: `{f['evidence_status_in_run']}`")
        add(f"- **System abstained:** `{f['abstained_in_run']}` · "
            f"key-fact recall `{f['key_fact_recall']}` · answer F1 `{f['answer_f1_normalized']}`")
        add(f"- **Why it was flagged:** {' '.join(e['reasons'])}\n")
        add("**Reference answers.**\n")
        for ans in e["reference_answers"] or ["(none recorded)"]:
            add(f"- {ans}")
        add("")
        add("**System answer.**\n")
        add("```")
        add(textwrap.fill(str(e["system_answer"] or "(empty)"), 96))
        add("```\n")
        add("**Gold evidence.**\n")
        for g in e["gold_evidence"] or []:
            add(f"*{g['doc_id']} — chars [{g['char_range'][0]}, {g['char_range'][1]}), "
                f"{len(g.get('text', ''))} chars, complete={g.get('text_complete')}*\n")
            add("```")
            add(textwrap.fill(g.get("text", ""), 96))
            add("```\n")
        if not (e["gold_evidence"] or []):
            add("*(none recorded — this question is not answerable from the corpus)*\n")
        add("**Retrieved context (full text, in rank order).**\n")
        for c in e["retrieved_context"] or []:
            add(f"*Rank {c.get('rank')} — {c['doc_id']} — chars "
                f"[{c['char_range'][0]}, {c['char_range'][1]}), {len(c.get('text', ''))} chars, "
                f"complete={c.get('text_complete')}*\n")
            add("```")
            add(textwrap.fill(c.get("text", ""), 96))
            add("```\n")
    return "\n".join(L)


def render_readme(n: int, source: Path, verdict: str, patterns: dict[str, int]) -> str:
    parent = Path(*source.parts[:-1]).as_posix()
    source = source.as_posix()
    lines = [
        f"# Focused manual-review set — {n} flagged units\n",
        "**This is a focused manual-review set containing the "
        f"{n} units flagged by the human-annotation audit. It is NOT a replacement for "
        "the original 200-unit human annotation dataset.**\n",
        f"The complete 200-unit pass remains where it was, untouched, in "
        f"`{source}/annotator_human/completed.jsonl`. The units here are copies, carrying "
        "their original labels, confidences and notes exactly as they were written.\n",
        "## Why these units\n",
        f"The audit (`{source}/audit/`) checked every human label against the decision "
        f"procedure in `docs/ANNOTATION_GUIDELINES.md` and the evidence stored in the "
        f"package. These {n} received the verdict `{verdict}`: the label conflicts with an "
        "explicit rule in the written procedure. That is a statement about the guidelines, "
        "not a judgement that the reading of the case was wrong — several of these are "
        "defensible readings of hard units, and two are places where the *automated* "
        "reference is the one at odds with the evidence.\n",
        "Patterns represented here:\n",
        "| Pattern | n |",
        "|---|---:|",
    ]
    for pattern, count in sorted(patterns.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {pattern} | {count} |")
    lines += [
        "",
        "## How to review\n",
        "```bash",
        f"python scripts/annotate.py --annotator {REVIEW_SLOT} \\",
        f"    --package {parent}/" + "REVIEW_PACKAGE_NAME",
        "```",
        "",
        f"The interface opens on the first unit and shows all {n} in the progress grid. "
        "Each one arrives pre-filled with the label you gave it, so the grid reads "
        f"{n}/{n} from the start — this is a review pass, not a fresh annotation. Change a "
        "label only if you decide it should change; leaving it alone is a valid outcome and "
        "records agreement with your original judgement.\n",
        "The interface is blinded by design: it shows the question, evidence, retrieved "
        "context and system answer, but not the automated reference label or the audit "
        "reasoning. Both are in `review_dossier.md` alongside the full retrieved text, for "
        "reading before or after — whichever you prefer methodologically.\n",
        "## Files\n",
        "| File | Contents |",
        "|---|---|",
        "| `annotation_sheet.jsonl` | the master sheet for this package |",
        f"| `annotator_{REVIEW_SLOT}/annotation_sheet.jsonl` | the sheet the interface serves |",
        f"| `annotator_{REVIEW_SLOT}/completed.jsonl` | working copy, pre-filled with the original labels |",
        "| `review_dossier.md` | full review context per unit, including the reference label and audit reasoning |",
        "| `review_dossier.json` | the same, machine-readable |",
        "| `manifest.json` | provenance, source checksums, integrity counts |",
        "",
        "## What this package does not do\n",
        "- It does not modify, replace or supersede the original 200-unit pass.\n"
        "- It does not feed any evaluation or reported result. Nothing in `docs/` or the "
        "README reads from it.\n"
        "- Decisions made here are yours to apply, or not, to the original dataset in a "
        "separate, deliberate step.\n",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="source annotation package (read-only)")
    ap.add_argument("--annotator", default="human", help="annotator slot holding the labels")
    ap.add_argument("--audit", required=True, help="human_annotation_audit.json")
    ap.add_argument("--verdict", default="likely_inconsistent",
                    help="audit verdict to select on")
    ap.add_argument("--out", required=True, help="review package to create")
    args = ap.parse_args()

    source = Path(args.package)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} already exists and is not empty - refusing to overwrite it")
        return 1

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    flagged = [u for u in audit["units"] if u["verdict"] == args.verdict]
    if not flagged:
        print(f"no units carry the verdict {args.verdict!r}")
        return 1
    wanted = [u["annotation_id"] for u in flagged]

    master = {u["annotation_id"]: u for u in read_jsonl(source / "annotation_sheet.jsonl")}
    human = {r["annotation_id"]: r
             for r in read_jsonl(source / f"annotator_{args.annotator}" / "completed.jsonl")}

    missing = [i for i in wanted if i not in master or i not in human]
    if missing:
        print(f"not present in the source package: {missing}")
        return 1

    def pattern_of(unit: dict) -> str:
        r = unit["reasons"][0] if unit["reasons"] else ""
        if "no_retrieval as an empty" in r:
            return "`no_retrieval` used although chunks were retrieved"
        if "step 2 directs a retrieval label" in r:
            return "answer-quality label although no gold span reached the system"
        if "so the guidelines direct" in r:
            return "wrong choice between the two unanswerable categories"
        if "admits only ok_abstained" in r:
            return "unanswerable question given an answerable-only label"
        if "reserved for corpus_can_answer=false" in r:
            return "unanswerable-only label on an answerable question"
        if "no abstention" in r:
            return "`refusal_when_answerable` without an abstention in the run"
        return r[:70]

    units = [dict(master[i]) for i in wanted]
    slot = out / f"annotator_{REVIEW_SLOT}"
    slot.mkdir(parents=True, exist_ok=True)

    write_jsonl(out / "annotation_sheet.jsonl", units)
    write_jsonl(slot / "annotation_sheet.jsonl", units)

    completed = []
    for unit_id in wanted:
        row = dict(master[unit_id])
        for field in HUMAN_FIELDS:
            row[field] = human[unit_id].get(field, "")
        completed.append(row)
    write_jsonl(slot / "completed.jsonl", completed)

    entries = []
    for u in flagged:
        unit = master[u["annotation_id"]]
        entries.append({
            "annotation_id": u["annotation_id"],
            "question": unit["question"],
            "corpus_can_answer": unit["corpus_can_answer"],
            "reference_answers": unit.get("reference_answers"),
            "gold_evidence": unit.get("gold_evidence"),
            "retrieved_context": unit.get("retrieved_context"),
            "system_answer": unit.get("system_answer"),
            "human_label": u["human_label"],
            "human_confidence": u["human_confidence"],
            "human_notes": u["human_notes"],
            "reference_label": u["reference_label"],
            "taxonomy_document_gated": u["taxonomy_document_gated"],
            "taxonomy_evidence_gated": u["taxonomy_evidence_gated"],
            "audit_verdict": u["verdict"],
            "audit_pattern": pattern_of(u),
            "reasons": u["reasons"],
            "facts": u["facts"],
        })

    (out / "review_dossier.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "review_dossier.md").write_text(
        render_dossier(entries, source, args.verdict), encoding="utf-8")

    patterns: dict[str, int] = {}
    for e in entries:
        patterns[e["audit_pattern"]] = patterns.get(e["audit_pattern"], 0) + 1
    readme = render_readme(len(entries), source, args.verdict, patterns)
    readme = readme.replace("REVIEW_PACKAGE_NAME", out.name)
    (out / "README.md").write_text(readme, encoding="utf-8")

    integrity = context_integrity(units)
    manifest = {
        "kind": "focused manual-review subset",
        "not_a_replacement_for": str(source / f"annotator_{args.annotator}" / "completed.jsonl"),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_package": str(source),
        "source_annotator": args.annotator,
        "audit_file": args.audit,
        "selected_on_verdict": args.verdict,
        "n_units": len(units),
        "annotation_ids": wanted,
        "annotator_slot": REVIEW_SLOT,
        "labels_are_copies": True,
        "source_checksums": {
            "annotation_sheet.jsonl": sha256(source / "annotation_sheet.jsonl"),
            f"annotator_{args.annotator}/completed.jsonl":
                sha256(source / f"annotator_{args.annotator}" / "completed.jsonl"),
        },
        "context_integrity": integrity,
        "audit_patterns": patterns,
        "note": "Labels here are copies of the original human pass. Editing them does not "
                "change the source package; applying any decision back to the 200-unit "
                "dataset is a separate, deliberate step.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"package  : {out.as_posix()}")
    print(f"units    : {len(units)}")
    print(f"integrity: {integrity['retrieved_complete']}/{integrity['retrieved_chunks']} "
          f"retrieved chunks complete, "
          f"{integrity['gold_complete']}/{integrity['gold_spans']} gold spans complete")
    print(f"launch   : python scripts/annotate.py --annotator {REVIEW_SLOT} "
          f"--package {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
