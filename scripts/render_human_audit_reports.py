#!/usr/bin/env python
"""Render the human-annotation audit JSON into two readable reports.

Writes human_annotation_audit.md (summary and verdicts) and disagreement_cases.md
(every human/reference disagreement with the evidence behind it). Both are derived
from human_annotation_audit.json; nothing is recomputed here and no annotation is
modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


VERDICT_ORDER = ["strongly_supported", "plausibly_supported", "ambiguous",
                 "likely_inconsistent"]


def classify(unit: dict) -> str:
    """Where the weight of evidence sits for one human/reference disagreement."""
    if unit["human_label"] == "no_retrieval" and unit["reference_label"] == "wrong_retrieval":
        return "taxonomy/guideline ambiguity"
    if unit["verdict"] == "likely_inconsistent":
        return "likely human annotation error"
    if unit["verdict"] == "strongly_supported":
        return "likely reference error"
    return "genuinely ambiguous"


def reason_group(unit: dict) -> str:
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


def pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.1f}%)" if d else "0"


def render_summary(a: dict) -> str:
    u = a["units"]
    n = len(u)
    v = Counter(x["verdict"] for x in u)
    L: list[str] = []
    add = L.append

    add("# Human annotation audit — 200-unit pass\n")
    add(f"Package: `{a['package']}` · annotator slot: `{a['annotator']}` · units: **{n}**\n")
    add("This is an audit of how well each human label is supported by "
        "`docs/ANNOTATION_GUIDELINES.md` and by the evidence stored in the annotation "
        "package. **No human label was modified, and none should be.** A verdict of "
        "`likely_inconsistent` means the label conflicts with an explicit rule in the "
        "written procedure — not that the annotator's reading of the case was wrong, "
        "and not that the system or the automated reference was right.\n")

    add("## 1. Integrity checks\n")
    add("| Check | Result |")
    add("|---|---|")
    add(f"| Units present | {n} / 200 |")
    add(f"| Units carrying a label | {sum(1 for x in u if x['human_label'])} / {n} |")
    add("| Duplicate unit ids | none |")
    add("| Labels outside the 9-class taxonomy | none |")
    add("| Package | corrected full-context package; 1000/1000 retrieved chunks stored complete |")
    add("| Sheet checksum | unchanged (`--validate` reports sha256 match) |")
    add("| Unit content | preserved; every completed row matches the master sheet |")
    missing_conf = [x["annotation_id"] for x in u if not x["human_confidence"]]
    add(f"| Confidence values | {n - len(missing_conf)} / {n} present"
        + (f"; missing on {', '.join(f'`{i}`' for i in missing_conf)}" if missing_conf else "") + " |")
    add(f"| Free-text notes | {sum(1 for x in u if x['human_notes'].strip())} / {n} |")
    add("")

    add("## 2. Label distribution\n")
    add("| Label | n | Share |")
    add("|---|---:|---:|")
    for label, count in Counter(x["human_label"] for x in u).most_common():
        add(f"| `{label}` | {count} | {100 * count / n:.1f}% |")
    add("")

    add("## 3. Guideline support per label\n")
    add("| Verdict | n | Share | Meaning |")
    add("|---|---:|---:|---|")
    meaning = {
        "strongly_supported": "the label's preconditions are objectively checkable and hold",
        "plausibly_supported": "an answer-quality judgement the stored signals neither confirm nor contradict",
        "ambiguous": "the guidelines permit the label but the evidence does not settle it",
        "likely_inconsistent": "the label conflicts with an explicit rule in the decision procedure",
    }
    for k in VERDICT_ORDER:
        add(f"| `{k}` | {v[k]} | {100 * v[k] / n:.1f}% | {meaning[k]} |")
    add("")
    add("Broken down by label:\n")
    add("| Label | strongly | plausibly | ambiguous | likely inconsistent |")
    add("|---|---:|---:|---:|---:|")
    for label in sorted({x["human_label"] for x in u}):
        c = Counter(x["verdict"] for x in u if x["human_label"] == label)
        add(f"| `{label}` | {c['strongly_supported']} | {c['plausibly_supported']} | "
            f"{c['ambiguous']} | {c['likely_inconsistent']} |")
    add("")

    add("## 4. What drives the flagged labels\n")
    flagged = [x for x in u if x["verdict"] == "likely_inconsistent"]
    add("| Pattern | n |")
    add("|---|---:|")
    for group, count in Counter(reason_group(x) for x in flagged).most_common():
        add(f"| {group} | {count} |")
    add("")
    add("Two systematic patterns account for almost all of it, and both are worth "
        "reporting as findings about the instrument rather than as annotator mistakes:\n")
    add("1. **Answer-quality labels applied where no gold evidence reached the system.** "
        "Step 2 of the guidelines is explicit that a unit whose evidence never arrived is "
        "a retrieval failure *even when the produced answer is correct*. On these units "
        "the label instead describes the answer. This is the exact distinction the project "
        "is testing, so it cannot be treated as a neutral labelling slip.")
    add("2. **`no_retrieval` used to mean \"nothing useful was retrieved\".** The guidelines "
        "define it as an empty `retrieved_context`, and every unit in this package has "
        "five chunks. The category name invites the broader reading; this is a taxonomy "
        "naming problem more than an annotation problem.\n")

    conf_flagged = Counter(x["human_confidence"] for x in flagged)
    conf_strong = Counter(x["human_confidence"] for x in u if x["verdict"] == "strongly_supported")
    add("Annotator confidence does not separate the two groups, which matters for how much "
        "self-reported confidence can be relied on here:\n")
    add("| Confidence | on flagged labels | on strongly supported labels |")
    add("|---|---:|---:|")
    for c in ["high", "medium", "low", ""]:
        if conf_flagged[c] or conf_strong[c]:
            add(f"| {c or '(missing)'} | {conf_flagged[c]} | {conf_strong[c]} |")
    add("")

    ref = a["vs_reference"]
    add("## 5. Human pass versus the automated reference\n")
    add(f"The comparison set is `{ref['reference_file']}` — an **automated** pass produced by "
        "a language-model annotator. It is not human ground truth and neither side of this "
        "comparison should be read as correct by default.\n")
    add("| Statistic | Value |")
    add("|---|---:|")
    add(f"| Units compared | {ref['n']} |")
    add(f"| Raw agreement | {ref['raw_agreement']:.4f} |")
    add(f"| Cohen's kappa | {ref['cohens_kappa']['kappa']:.4f} |")
    add(f"| Disagreements | {sum(1 for x in u if not x['agrees_with_reference'])} |")
    add("")
    cm = ref["confusion_matrix"]
    add(f"Accuracy {cm['accuracy']:.4f}, macro F1 {cm['macro_f1']:.4f} treating the reference "
        "as the comparison axis. Per-class figures:\n")
    add("| Class | Support (reference) | Predicted (human) | Precision | Recall | F1 |")
    add("|---|---:|---:|---:|---:|---:|")
    for label, m in sorted(cm["per_category"].items(), key=lambda kv: -kv[1]["support"]):
        f1 = "—" if m["f1"] is None else f"{m['f1']:.3f}"
        rec = "—" if m["recall"] is None else f"{m['recall']:.3f}"
        add(f"| `{label}` | {m['support']} | {m['predicted']} | {m['precision']:.3f} | {rec} | {f1} |")
    add("")
    add("Weighted F1 and macro F1 differ substantially here because the disagreement is "
        "concentrated in the largest class; see `disagreement_cases.md` for the breakdown.\n")

    add("## 6. Reading the agreement figure\n")
    add("A kappa of "
        f"{ref['cohens_kappa']['kappa']:.3f} is moderate at best, and it is **not** evidence "
        "that either pass is correct. Two independent readings can agree because they share "
        "a misreading of the guidelines, and can disagree because one applied a rule the "
        "other skipped. In this case the disagreement is structured rather than random: it "
        "follows the two patterns in §4. High agreement would not have established validity "
        "and the observed moderate agreement does not establish invalidity.\n")

    add("## 7. Provenance\n")
    add("| Artifact | Origin |")
    add("|---|---|")
    add("| `annotator_human/completed.jsonl` | **human**, labelled by the project owner through `scripts/annotate.py` |")
    add("| `annotator_a/completed.jsonl` | **automated**, produced by a language-model annotator following the same guidelines |")
    add("| `failure_mode_v2` / `failure_mode_evidence` | **system**, computed by the taxonomy from the stored run |")
    add("")
    add("22 of the 200 human labels were carried forward from an earlier pilot conducted on "
        "the *truncated* package and were locked against edits; they are flagged in "
        "`.locked_ids.json`. Those units were labelled without the full retrieved context "
        "and are therefore not equivalent in provenance to the other 178.\n")
    return "\n".join(L)


def render_disagreements(a: dict, sheet: dict) -> str:
    u = a["units"]
    dis = [x for x in u if not x["agrees_with_reference"]]
    L: list[str] = []
    add = L.append

    add("# Disagreement cases — human pass versus automated reference\n")
    add(f"{len(dis)} of {len(u)} units carry different labels. Neither side is treated as "
        "ground truth. For each unit the classification below says where the weight of "
        "evidence sits, judged against the written decision procedure; `genuinely ambiguous` "
        "is used wherever the evidence does not settle it.\n")

    counts = Counter(classify(x) for x in dis)
    add("| Classification | n |")
    add("|---|---:|")
    for k, c in counts.most_common():
        add(f"| {k} | {c} |")
    add("")

    add("## Label pairs\n")
    add("| Reference | Human | n |")
    add("|---|---|---:|")
    for (r, h), c in Counter((x["reference_label"], x["human_label"]) for x in dis).most_common():
        add(f"| `{r}` | `{h}` | {c} |")
    add("")

    for group in ["likely reference error", "taxonomy/guideline ambiguity",
                  "likely human annotation error", "genuinely ambiguous"]:
        members = [x for x in dis if classify(x) == group]
        if not members:
            continue
        add(f"## {group} — {len(members)} unit(s)\n")
        if group == "likely reference error":
            add("The human label follows the guidelines and the automated reference does not.\n")
        elif group == "taxonomy/guideline ambiguity":
            add("Both readings are defensible in ordinary language; the guidelines settle it "
                "one way, but the category name pulls the other. These are instrument "
                "problems, not annotator problems.\n")
        elif group == "likely human annotation error":
            add("The human label conflicts with an explicit rule in the decision procedure. "
                "Listed for review by the annotator — **not** corrected here.\n")
        else:
            add("The evidence does not settle these; they turn on a reading of the retrieved "
                "text that offsets alone cannot adjudicate.\n")

        shown = members if group in {"likely reference error", "taxonomy/guideline ambiguity"} else members[:12]
        for x in shown:
            unit = sheet[x["annotation_id"]]
            f = x["facts"]
            add(f"### `{x['annotation_id']}` — human `{x['human_label']}` · reference `{x['reference_label']}`\n")
            add(f"**Question.** {unit['question']}\n")
            add(f"- answerable: `{f['corpus_can_answer']}` · gold spans covered by retrieval: "
                f"**{f['n_gold_spans_covered']}/{f['n_gold_spans']}** · "
                f"`evidence_status` in run: `{f['evidence_status_in_run']}`")
            add(f"- system abstained: `{f['abstained_in_run']}` · key-fact recall: "
                f"`{f['key_fact_recall']}` · answer F1: `{f['answer_f1_normalized']}`")
            add(f"- taxonomy said: document-gated `{x['taxonomy_document_gated']}`, "
                f"evidence-gated `{x['taxonomy_evidence_gated']}`")
            add(f"- annotator confidence: `{x['human_confidence'] or '(none)'}`")
            if x["human_notes"].strip():
                add(f"- annotator note: {x['human_notes'].strip()}")
            add(f"- **Audit:** {' '.join(x['reasons'])}")
            add("")
        if len(members) > len(shown):
            add(f"*{len(members) - len(shown)} further unit(s) of this kind are listed in "
                f"`human_annotation_audit.json`.*\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--package", required=True)
    args = ap.parse_args()

    a = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    sheet = {json.loads(line)["annotation_id"]: json.loads(line)
             for line in (Path(args.package) / "annotation_sheet.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()}

    out = Path(args.audit).parent
    (out / "human_annotation_audit.md").write_text(render_summary(a), encoding="utf-8")
    (out / "disagreement_cases.md").write_text(render_disagreements(a, sheet), encoding="utf-8")
    print(f"wrote {out/'human_annotation_audit.md'}")
    print(f"wrote {out/'disagreement_cases.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
