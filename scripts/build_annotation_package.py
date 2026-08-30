"""Build a human-annotation package for validating the failure taxonomy (W4).

The taxonomy's thresholds were chosen by inspecting a 20-question fixture and
have never been checked against human judgement. Until they are, the taxonomy
is a proposal, not a validated instrument. This script produces everything a
human annotator needs and nothing they should not see.

What it does **not** do is produce labels. No label in the output is filled in
by this script or by any model; `human_label` is left empty for a person to
complete. Anything else would be fabricating the very evidence the study is
supposed to collect.

Design decisions that protect the study:

**Stratified sampling.** Rare categories matter most — an abstention failure
that occurs in 3% of rows would barely appear in a uniform sample of 150. Rows
are sampled per proposed failure mode so every category the taxonomy can emit
is represented, and the sampling weights are recorded so results can be
reweighted back to the population.

**The proposed label is withheld by default.** Showing an annotator the
system's own answer invites anchoring, which would inflate agreement. The
proposed label is written to a separate key file, not to the annotation sheet.

**Blind ordering.** Rows are shuffled with a recorded seed so annotators cannot
infer anything from position.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.provenance import collect_provenance
from src.evaluation.records import read_records
from src.evaluation.runner import score_records
from src.evaluation.taxonomy import FailureModeV2, TaxonomyConfig

#: Fixed so the same records always yield the same annotation sample.
DEFAULT_SEED = 20260826


def build_unit(record, row, index: int) -> dict:
    """One annotation unit: everything needed to judge, nothing that anchors.

    The annotator sees the question, the reference answers, whether the corpus
    is supposed to be able to answer it, the retrieved context in rank order,
    and the system's answer. They do not see the proposed failure mode, the
    metric values, or the decision features.
    """
    evidence_preview = []
    for chunk in record.retrieved:
        evidence_preview.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "char_range": [chunk.start_char, chunk.end_char],
                "text": chunk.text[:600],
            }
        )

    gold_spans = (record.metadata or {}).get("supporting_spans") or []
    return {
        "annotation_id": f"unit_{index:04d}",
        "question": record.question,
        "reference_answers": row.reference_answers or [record.reference_answer],
        "corpus_can_answer": bool(record.relevant_doc_ids),
        "gold_evidence": [
            {
                "doc_id": s["doc_id"],
                "char_range": [s["start_char"], s["end_char"]],
                "text": str(s.get("text", ""))[:600],
            }
            for s in gold_spans
        ],
        "retrieved_context": evidence_preview,
        "system_answer": record.predicted_answer,
        # To be completed by a human. Left empty on purpose.
        "human_label": "",
        "human_notes": "",
        "human_confidence": "",
    }


def stratified_sample(rows, records, per_mode: int, seed: int):
    """Sample per proposed failure mode so rare categories are represented."""
    rng = random.Random(seed)
    by_mode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_mode[row.failure_mode_v2].append(index)

    chosen: list[int] = []
    weights: dict[str, dict] = {}
    for mode, indices in sorted(by_mode.items()):
        take = min(per_mode, len(indices))
        picked = rng.sample(indices, take)
        chosen.extend(picked)
        weights[mode] = {
            "population": len(indices),
            "sampled": take,
            # Reweighting factor to recover population proportions.
            "weight": round(len(indices) / take, 4) if take else None,
        }
    rng.shuffle(chosen)
    return chosen, weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human annotation package")
    parser.add_argument("--records", required=True, help="inference.jsonl from an experiment")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--per-mode", type=int, default=25, help="units per proposed failure mode")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"no records at {records_path}", file=sys.stderr)
        return 1

    records = read_records(records_path)
    rows = score_records(records, TaxonomyConfig())
    chosen, weights = stratified_sample(rows, records, args.per_mode, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    units, key = [], []
    for position, index in enumerate(chosen):
        unit = build_unit(records[index], rows[index], position)
        units.append(unit)
        # The proposed label lives only in the key file, so the annotation
        # sheet cannot anchor the annotator.
        key.append(
            {
                "annotation_id": unit["annotation_id"],
                "question_id": (records[index].metadata or {}).get("question_id", ""),
                "proposed_label": rows[index].failure_mode_v2,
                "proposed_rule": rows[index].failure_rule_v2,
                "attribution_stage": rows[index].attribution_stage,
                "evidence_status": rows[index].evidence_status,
            }
        )

    (out_dir / "annotation_sheet.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n", encoding="utf-8"
    )
    (out_dir / "proposed_labels_key.jsonl").write_text(
        "\n".join(json.dumps(k, ensure_ascii=False) for k in key) + "\n", encoding="utf-8"
    )

    manifest = {
        "n_units": len(units),
        "per_mode_target": args.per_mode,
        "seed": args.seed,
        "sampling_weights": weights,
        "population_distribution": dict(Counter(r.failure_mode_v2 for r in rows)),
        "label_set": [str(m) for m in FailureModeV2],
        "status": "AWAITING HUMAN ANNOTATION - no labels have been collected",
        "agreement_statistic": (
            "Cohen's kappa for two annotators on the nominal label set. Report the "
            "confusion matrix alongside it: kappa summarises agreement but hides "
            "which distinctions the rules actually miss, and those are the "
            "informative part."
        ),
        "provenance": collect_provenance(source_records=str(records_path)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(units)} annotation units to {out_dir}")
    print("  annotation_sheet.jsonl     - for the annotator (human_label empty)")
    print("  proposed_labels_key.jsonl  - withheld until annotation is complete")
    print("  manifest.json              - sampling weights and protocol")
    print("\nNo labels were generated. Human annotation is still required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
