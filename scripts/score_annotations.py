"""Score completed human annotations against each other and against the taxonomy.

This is the second half of the taxonomy-validation protocol. The first half
(`build_annotation_package.py`) produces blinded sheets; this one consumes them
once humans have filled them in.

It computes three different things, and the distinction matters:

1. **Annotator vs annotator.** Cohen's kappa and a confusion matrix. This
   measures whether the *task* is well defined. A low kappa here means the
   category boundaries are ambiguous to humans, and no amount of agreement with
   the taxonomy afterwards would be meaningful.

2. **Adjudicated labels.** Where the two annotators agree, that label stands.
   Where they disagree, the item is *unresolved* unless a third-pass
   adjudication file is supplied. Disagreements are never broken by taking the
   system's label, which would quietly make the taxonomy its own referee.

3. **Taxonomy vs adjudicated labels.** Only here is the system evaluated, and
   only on items humans actually resolved. Per-category precision, recall and
   F1, because a single accuracy figure on a skewed label set says nothing
   about the rare categories the taxonomy exists to separate.

The script fabricates nothing. If no human labels are present it says so and
exits non-zero rather than emitting a plausible-looking table.

    python scripts/score_annotations.py \
        --package reports/annotation/qasper_dev_300 \
        --annotator a=reports/annotation/qasper_dev_300/annotator_a/completed.jsonl \
        --annotator b=reports/annotation/qasper_dev_300/annotator_b/completed.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance
from src.evaluation.statistics import cohens_kappa, confusion_matrix
from src.evaluation.taxonomy import FailureModeV2


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def labelled_units(path: Path) -> dict[str, str]:
    """Annotation id -> non-empty human label, for one annotator."""
    labels = {}
    for unit in read_jsonl(path):
        label = str(unit.get("human_label", "")).strip()
        if label:
            labels[unit["annotation_id"]] = label
    return labels


def validate_labels(labels: dict[str, str], who: str) -> list[str]:
    """Labels outside the declared set are a protocol error, not a data point."""
    allowed = {str(mode) for mode in FailureModeV2}
    return sorted({v for v in labels.values() if v not in allowed})


def adjudicate(
    a: dict[str, str], b: dict[str, str], third: dict[str, str] | None
) -> tuple[dict[str, str], list[str]]:
    """Agreed labels stand; disagreements need a human third pass or stay open."""
    resolved, unresolved = {}, []
    for unit_id in sorted(set(a) & set(b)):
        if a[unit_id] == b[unit_id]:
            resolved[unit_id] = a[unit_id]
        elif third and unit_id in third:
            resolved[unit_id] = third[unit_id]
        else:
            unresolved.append(unit_id)
    return resolved, unresolved


def disagreement_pairs(a: dict[str, str], b: dict[str, str]) -> list[dict]:
    """Which category confusions actually occur, most frequent first."""
    pairs = Counter()
    for unit_id in set(a) & set(b):
        if a[unit_id] != b[unit_id]:
            pairs[tuple(sorted((a[unit_id], b[unit_id])))] += 1
    return [
        {"categories": list(pair), "count": count}
        for pair, count in pairs.most_common()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score human annotations of the taxonomy")
    parser.add_argument("--package", required=True,
                        help="annotation package directory (holds proposed_labels_key.jsonl)")
    parser.add_argument("--annotator", action="append", default=[], metavar="ID=PATH",
                        help="completed sheet, e.g. a=path/completed.jsonl (repeatable)")
    parser.add_argument("--adjudicated", default="",
                        help="optional third-pass file resolving disagreements")
    parser.add_argument("--out", default="", help="where to write the report JSON")
    args = parser.parse_args()

    package = Path(args.package)
    key_path = package / "proposed_labels_key.jsonl"
    if not key_path.exists():
        print(f"no annotation package at {package}", file=sys.stderr)
        return 1
    proposed = {k["annotation_id"]: k["proposed_label"] for k in read_jsonl(key_path)}

    if len(args.annotator) < 2:
        print(
            "Need at least two annotators to measure agreement.\n"
            f"The package at {package} holds {len(proposed)} units awaiting labels.\n"
            "Pass completed sheets as --annotator a=PATH --annotator b=PATH.",
            file=sys.stderr,
        )
        return 2

    annotations: dict[str, dict[str, str]] = {}
    for spec in args.annotator:
        if "=" not in spec:
            print(f"expected ID=PATH, got {spec!r}", file=sys.stderr)
            return 1
        who, path = spec.split("=", 1)
        sheet = Path(path)
        if not sheet.exists():
            print(f"annotator {who}: no file at {sheet}", file=sys.stderr)
            return 1
        annotations[who] = labelled_units(sheet)

    empty = [who for who, labels in annotations.items() if not labels]
    if empty:
        print(
            f"No human labels found for: {', '.join(empty)}.\n"
            "Every `human_label` is blank, so there is nothing to score. "
            "This script will not invent labels.",
            file=sys.stderr,
        )
        return 3

    for who, labels in annotations.items():
        invalid = validate_labels(labels, who)
        if invalid:
            print(f"annotator {who} used labels outside the taxonomy: {invalid}",
                  file=sys.stderr)
            return 1

    ids = sorted(set.intersection(*(set(v) for v in annotations.values())))
    if not ids:
        print("annotators share no labelled units", file=sys.stderr)
        return 3

    who_a, who_b = sorted(annotations)[:2]
    a, b = annotations[who_a], annotations[who_b]
    agreement = cohens_kappa([a[i] for i in ids], [b[i] for i in ids])
    inter = confusion_matrix([a[i] for i in ids], [b[i] for i in ids])

    third = labelled_units(Path(args.adjudicated)) if args.adjudicated else None
    resolved, unresolved = adjudicate(a, b, third)

    report = {
        "package": str(package),
        "n_units_in_package": len(proposed),
        "annotators": {who: len(labels) for who, labels in annotations.items()},
        "n_jointly_labelled": len(ids),
        "inter_annotator": {
            "pair": [who_a, who_b],
            "agreement": agreement.as_dict(),
            "confusion_matrix": inter,
            "disagreements": disagreement_pairs(a, b),
        },
        "adjudication": {
            "n_resolved": len(resolved),
            "n_unresolved": len(unresolved),
            "unresolved_ids": unresolved,
            "third_pass_supplied": bool(third),
            "rule": (
                "Agreed labels stand. Disagreements are resolved only by a human "
                "third pass; they are never broken using the system's own label."
            ),
        },
        "provenance": collect_provenance(source_package=str(package)),
    }

    if resolved:
        truth = [resolved[i] for i in sorted(resolved)]
        system = [proposed.get(i, "") for i in sorted(resolved)]
        report["taxonomy_vs_human"] = confusion_matrix(truth, system)
    else:
        report["taxonomy_vs_human"] = {
            "note": "no adjudicated labels; taxonomy not evaluated"
        }

    out = Path(args.out) if args.out else package / "agreement_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== inter-annotator agreement ({who_a} vs {who_b}) ===")
    print(f"jointly labelled : {len(ids)}")
    print(f"observed agreement: {agreement.observed_agreement}")
    print(f"Cohen's kappa     : {agreement.kappa}   ({agreement.note or 'ok'})")
    print(f"\nadjudicated: {len(resolved)} resolved, {len(unresolved)} unresolved")
    tvh = report["taxonomy_vs_human"]
    if "accuracy" in tvh:
        print("\n=== taxonomy vs adjudicated human labels ===")
        print(f"accuracy : {tvh['accuracy']}   macro F1: {tvh['macro_f1']}")
        print(f"{'category':<32}{'support':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
        for category, stats in sorted(tvh["per_category"].items()):
            print(f"{category:<32}{stats['support']:>8}"
                  f"{_fmt(stats['precision']):>8}{_fmt(stats['recall']):>8}"
                  f"{_fmt(stats['f1']):>8}")
    print(f"\nwrote {out}")
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    sys.exit(main())
