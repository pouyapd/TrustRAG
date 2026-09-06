#!/usr/bin/env python
"""Ask whether the taxonomy's thresholds, not its rules, explain its disagreements.

The v2 thresholds were set by inspection on a 20-question development fixture and
have never been fitted to any evaluation set. That protects the headline comparison
from circularity, but it leaves an obvious question open: how much of the
generation-side disagreement with human labels is a threshold artefact?

The protocol is leakage-safe. Units are split in half under a fixed seed; a grid
search picks the configuration that maximises macro F1 on the tuning half only; the
winner is then scored once on the held-out half, which the search never saw. The
held-out number is the one that means anything. Both gates are evaluated, so tuning
cannot silently favour one.

    python scripts/threshold_ablation.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --labels reports/annotation/qasper_dev_300_full_context/final_human_reviewed/completed.jsonl \
        --rows reports/experiments/qasper_dev_300/rows.jsonl \
        --records reports/experiments/qasper_dev_300/inference.jsonl \
        --out reports/annotation/qasper_dev_300_full_context/audit/threshold_ablation.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.statistics import cohens_kappa, confusion_matrix  # noqa: E402
from src.evaluation.taxonomy import (  # noqa: E402
    DecisionFeatures,
    TaxonomyConfig,
    classify_features,
)

GRID = {
    "answer_f1_ok": [0.30, 0.40, 0.50, 0.60],
    "key_fact_recall_ok": [0.50, 0.60, 0.75, 1.0],
    "key_fact_recall_incorrect": [0.0, 0.10, 0.20],
    "fallback_f1_incorrect": [0.05, 0.10, 0.20],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def features_from_row(row: dict, gate: str) -> DecisionFeatures:
    """Rebuild the classifier's inputs from a stored row.

    `gate` selects which retrieval signal fills `retrieval_hit`: the document-level
    match, or whether the gold span actually arrived.
    """
    doc_hit = (row.get("doc_recall_at_k") or 0) > 0
    evidence_hit = row.get("evidence_status") == "complete"
    return DecisionFeatures(
        is_answerable=bool(row.get("is_answerable")),
        num_retrieved=len(row.get("retrieved_chunk_ids") or []),
        num_relevant_retrieved=int(bool(doc_hit)),
        retrieval_hit=doc_hit if gate == "document" else evidence_hit,
        abstained=bool(row.get("abstained")),
        faithfulness=row.get("faithfulness"),
        answer_f1=row.get("answer_f1_normalized") or 0.0,
        answer_precision=row.get("answer_precision_normalized") or 0.0,
        answer_recall=row.get("answer_recall_normalized") or 0.0,
        answer_exact_match=row.get("answer_exact_match") or 0.0,
        key_fact_recall=row.get("key_fact_recall"),
        num_key_facts=row.get("num_key_facts") or 0,
    )


def score(truth: list[str], pred: list[str]) -> dict:
    cm = confusion_matrix(truth, pred)
    return {"n": len(truth), "accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"],
            "kappa": cohens_kappa(truth, pred).kappa, "per_category": cm["per_category"],
            "matrix": cm["matrix"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--labels", required=True, help="the label set to tune and test against")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg = Path(args.package)
    labels = {r["annotation_id"]: r["human_label"] for r in read_jsonl(Path(args.labels))}
    key = {k["annotation_id"]: k.get("question_id")
           for k in read_jsonl(pkg / "proposed_labels_key.jsonl")}
    records = read_jsonl(Path(args.records))
    rows = read_jsonl(Path(args.rows))
    index = {(r.get("metadata") or {}).get("question_id"): i for i, r in enumerate(records)}
    by_unit = {u: rows[index[key[u]]] for u in labels if key.get(u) in index}

    ids = sorted(by_unit)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    tune_ids, test_ids = sorted(ids[:half]), sorted(ids[half:])

    baseline = TaxonomyConfig()
    result = {
        "protocol": "50/50 split under a fixed seed; grid search maximises macro F1 on the "
                    "tuning half only; the selected configuration is scored once on the "
                    "held-out half, which the search never saw.",
        "labels_file": Path(args.labels).as_posix(),
        "seed": args.seed,
        "n_tune": len(tune_ids),
        "n_test": len(test_ids),
        "grid": GRID,
        "grid_size": len(list(product(*GRID.values()))),
        "baseline_thresholds": baseline.as_dict(),
        "gates": {},
    }

    for gate in ("document", "evidence"):
        feats = {u: features_from_row(by_unit[u], gate) for u in ids}

        def evaluate(cfg: TaxonomyConfig, subset: list[str], feats=feats) -> dict:
            truth = [labels[u] for u in subset]
            pred = [str(classify_features(feats[u], cfg).mode) for u in subset]
            return score(truth, pred)

        best, best_cfg = None, None
        for combo in product(*GRID.values()):
            cfg = TaxonomyConfig(**dict(zip(GRID, combo, strict=True)),
                                 faithfulness_threshold=baseline.faithfulness_threshold)
            s = evaluate(cfg, tune_ids)
            if best is None or s["macro_f1"] > best["macro_f1"]:
                best, best_cfg = s, cfg

        result["gates"][gate] = {
            "baseline": {
                "tune": evaluate(baseline, tune_ids),
                "held_out": evaluate(baseline, test_ids),
            },
            "tuned": {
                "thresholds": best_cfg.as_dict(),
                "fingerprint": best_cfg.fingerprint(),
                "tune": best,
                "held_out": evaluate(best_cfg, test_ids),
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"grid: {result['grid_size']} configurations · "
          f"tune n={len(tune_ids)} · held-out n={len(test_ids)}\n")
    for gate, g in result["gates"].items():
        b, t = g["baseline"]["held_out"], g["tuned"]["held_out"]
        print(f"{gate}-gated, held-out half:")
        print(f"  baseline  acc={b['accuracy']:.4f}  macroF1={b['macro_f1']:.4f}  kappa={b['kappa']:.4f}")
        print(f"  tuned     acc={t['accuracy']:.4f}  macroF1={t['macro_f1']:.4f}  kappa={t['kappa']:.4f}")
        print(f"  selected  {g['tuned']['thresholds']}")
    print(f"\nwrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
