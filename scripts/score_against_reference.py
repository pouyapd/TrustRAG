"""Score the taxonomy against a single human reference set.

`score_annotations.py` answers a different question: it measures two
annotators against each other and evaluates the system only on the units they
agreed on. Once a single reference set exists — an adjudicated file, or one
annotator's completed pass designated as the reference — the system should be
scored against all of it, not just the agreed subset.

This script does that, and nothing else. It fabricates no labels: the
reference comes from a completed annotation file, and the system's labels come
from the package's proposed-labels key. It also reports agreement between the
reference and any other completed passes supplied for comparison, because a
reference set built by one annotator says nothing about task difficulty on its
own.

    python scripts/score_against_reference.py \
        --package reports/annotation/<pkg> \
        --reference <pkg>/annotator_a/completed.jsonl \
        --compare a=<other pkg>/annotator_a/completed.jsonl \
        --out <pkg>/final_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance
from src.evaluation.statistics import cohens_kappa, confusion_matrix, mcnemar_exact
from src.evaluation.taxonomy import FailureModeV2


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def labels_of(path: Path) -> dict[str, str]:
    out = {}
    for row in read_jsonl(path):
        label = str(row.get("human_label", "")).strip()
        if label:
            out[row["annotation_id"]] = label
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the taxonomy against one reference set")
    parser.add_argument("--package", required=True)
    parser.add_argument("--reference", required=True,
                        help="completed annotation file to treat as the reference set")
    parser.add_argument("--compare", action="append", default=[], metavar="ID=PATH",
                        help="another completed pass to measure agreement against (repeatable)")
    parser.add_argument("--rows", default="",
                        help="rows.jsonl from the source experiment; adds the evidence-gated "
                             "taxonomy variant as a second scored labeller")
    parser.add_argument("--records", default="",
                        help="inference.jsonl matching --rows, used to join question_id -> row")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    package = Path(args.package)
    key_path = package / "proposed_labels_key.jsonl"
    if not key_path.exists():
        print(f"no proposed-labels key at {key_path}", file=sys.stderr)
        return 1
    key_rows = read_jsonl(key_path)
    proposed = {k["annotation_id"]: k["proposed_label"] for k in key_rows}
    question_ids = {k["annotation_id"]: k.get("question_id", "") for k in key_rows}

    reference = labels_of(Path(args.reference))
    if not reference:
        print(f"no human labels in {args.reference}", file=sys.stderr)
        return 3
    allowed = {str(mode) for mode in FailureModeV2}
    invalid = sorted({v for v in reference.values() if v not in allowed})
    if invalid:
        print(f"reference uses labels outside the taxonomy: {invalid}", file=sys.stderr)
        return 1

    ids = sorted(set(reference) & set(proposed))
    truth = [reference[i] for i in ids]
    system = [proposed[i] for i in ids]

    report = {
        "package": str(package),
        "reference_file": str(args.reference),
        "n_reference_labels": len(reference),
        "n_scored": len(ids),
        "reference_distribution": dict(Counter(truth).most_common()),
        "system_distribution": dict(Counter(system).most_common()),
        "taxonomy_vs_reference": confusion_matrix(truth, system),
        "taxonomy_vs_reference_kappa": cohens_kappa(truth, system).as_dict(),
        "agreement_with_other_passes": {},
        "provenance": collect_provenance(source_package=str(package)),
    }

    # Optional: the evidence-gated variant of the same taxonomy, read from the
    # stored rows of the source experiment. Same units, same reference, one
    # difference — whether the retrieval gate is document-level or span-level.
    if args.rows and args.records:
        records = read_jsonl(Path(args.records))
        rows = read_jsonl(Path(args.rows))
        if len(records) != len(rows):
            print("--records and --rows must be the same length", file=sys.stderr)
            return 1
        index_of = {(r.get("metadata") or {}).get("question_id"): i for i, r in enumerate(records)}
        variant, missing = {}, 0
        for unit_id in ids:
            i = index_of.get(question_ids.get(unit_id))
            if i is None:
                missing += 1
                continue
            label = rows[i].get("failure_mode_evidence")
            if label:
                variant[unit_id] = label
        shared = [i for i in ids if i in variant]
        if shared:
            v_truth = [reference[i] for i in shared]
            v_pred = [variant[i] for i in shared]
            only_main = sum(1 for i in shared
                            if proposed[i] == reference[i] and variant[i] != reference[i])
            only_variant = sum(1 for i in shared
                               if variant[i] == reference[i] and proposed[i] != reference[i])
            report["evidence_gated_variant_vs_reference"] = {
                "rows": args.rows,
                "n_scored": len(shared),
                "n_unmatched": missing,
                "variant_distribution": dict(Counter(v_pred).most_common()),
                "confusion_matrix": confusion_matrix(v_truth, v_pred),
                "agreement": cohens_kappa(v_truth, v_pred).as_dict(),
                "paired_comparison": {
                    "only_document_gated_correct": only_main,
                    "only_evidence_gated_correct": only_variant,
                    "test": mcnemar_exact(only_main, only_variant).as_dict(),
                },
                "note": (
                    "failure_mode_v2 gates the retrieval rule on document-level retrieval; "
                    "failure_mode_evidence gates it on whether the gold span reached the "
                    "generator. Both are scored against the same reference labels."
                ),
            }

    for spec in args.compare:
        if "=" not in spec:
            print(f"expected ID=PATH, got {spec!r}", file=sys.stderr)
            return 1
        who, path = spec.split("=", 1)
        other = labels_of(Path(path))
        shared = sorted(set(reference) & set(other))
        if not shared:
            report["agreement_with_other_passes"][who] = {"n": 0, "note": "no shared units"}
            continue
        agreement = cohens_kappa([reference[i] for i in shared], [other[i] for i in shared])
        report["agreement_with_other_passes"][who] = {
            "path": path,
            "n": len(shared),
            "agreement": agreement.as_dict(),
            "confusion_matrix": confusion_matrix(
                [reference[i] for i in shared], [other[i] for i in shared]
            ),
        }

    out = Path(args.out) if args.out else package / "final_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tvr = report["taxonomy_vs_reference"]
    print(f"=== taxonomy vs reference ({len(ids)} units) ===")
    print(f"accuracy : {tvr['accuracy']}   macro F1: {tvr['macro_f1']}")
    print(f"{'category':<32}{'support':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
    for category, stats in sorted(tvr["per_category"].items()):
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{category:<32}{stats['support']:>8}"
              f"{fmt(stats['precision']):>8}{fmt(stats['recall']):>8}{fmt(stats['f1']):>8}")
    variant = report.get("evidence_gated_variant_vs_reference")
    if variant:
        vcm = variant["confusion_matrix"]
        print("")
        print(f"=== evidence-gated variant vs the same reference "
              f"({variant['n_scored']} units) ===")
        print(f"accuracy : {vcm['accuracy']}   macro F1: {vcm['macro_f1']}   "
              f"kappa: {variant['agreement']['kappa']}")
        pc = variant["paired_comparison"]
        print(f"paired   : {pc['only_evidence_gated_correct']} units only the evidence-gated "
              f"variant gets right vs {pc['only_document_gated_correct']} the other way, "
              f"exact McNemar p={pc['test']['p_value']:.4f}")

    for who, block in report["agreement_with_other_passes"].items():
        if block.get("n"):
            print(f"\nagreement with {who}: kappa {block['agreement']['kappa']} "
                  f"(observed {block['agreement']['observed_agreement']}, n={block['n']})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
