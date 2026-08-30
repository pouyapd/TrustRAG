"""Controlled comparison of evaluation methodologies over one fixed run.

The comparison this project exists to make is between *ways of measuring*, not
between systems. So the retrieval and generation are run once, and every
methodology is then applied to the same stored inference records. Nothing
varies between conditions except the measurement, which is what makes the
differences attributable to the methodology rather than to run-to-run noise.

Conditions
----------
A. legacy            document-level retrieval, single reference answer, v1 taxonomy
B. taxonomy_v2       adds the corrected failure taxonomy
C. evidence_aware    adds exact character-offset evidence alignment
D. evidence_answerability  adds explicit answerability and abstention handling

Because all four score identical records, paired statistics apply: McNemar for
the paired binary "was this row judged a retrieval success" and a paired
bootstrap for the difference in rates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evidence import (
    EvidenceStatus,
    align_evidence,
    attribute_stage,
    retrieved_from_chunks,
    spans_from_records,
)
from src.evaluation.metrics import hit_rate_at_k
from src.evaluation.provenance import collect_provenance
from src.evaluation.records import read_records
from src.evaluation.runner import score_records
from src.evaluation.statistics import (
    mcnemar_exact,
    paired_bootstrap_difference,
    wilson_proportion_ci,
)
from src.evaluation.taxonomy import TaxonomyConfig


def retrieval_success_document_level(record) -> bool | None:
    """Condition A/B: did any retrieved chunk come from a relevant document?

    This is the conventional definition. On a corpus of long documents it is
    close to free: a chunk from anywhere in the right document counts.
    """
    if not record.relevant_doc_ids:
        return None
    hit = hit_rate_at_k(record.retrieved_doc_ids, record.relevant_doc_ids, record.top_k or 5)
    return bool(hit)


def retrieval_success_evidence_level(record) -> bool | None:
    """Condition C/D: did the gold evidence span actually reach the generator?"""
    metadata = record.metadata or {}
    gold = spans_from_records(metadata.get("supporting_spans") or [])
    if not gold:
        return None
    alignment = align_evidence(
        gold,
        retrieved_from_chunks(record.retrieved),
        evidence_mode=metadata.get("evidence_mode") or "any_sufficient",
    )
    return alignment.status is EvidenceStatus.COMPLETE


def build_conditions(records, rows) -> dict:
    """Per-row judgements under each condition, aligned by index."""
    doc_level = [retrieval_success_document_level(r) for r in records]
    ev_level = [retrieval_success_evidence_level(r) for r in records]

    # Attribution under document-level evidence: with no span information the
    # only thing a document-level pipeline can say is "the document was there,
    # so any remaining failure is the generator's".
    doc_attribution, ev_attribution = [], []
    for record, row, doc_ok in zip(records, rows, doc_level, strict=True):
        answerable = bool(record.relevant_doc_ids)
        correct = bool(row.answer_exact_match) or (
            row.key_fact_recall is not None and row.key_fact_recall >= 1.0
        )
        if not answerable:
            doc_attribution.append("none" if row.abstained else "generation")
        elif not doc_ok:
            doc_attribution.append("retrieval")
        else:
            doc_attribution.append("none" if correct else "generation")

        metadata = record.metadata or {}
        gold = spans_from_records(metadata.get("supporting_spans") or [])
        alignment = align_evidence(
            gold,
            retrieved_from_chunks(record.retrieved),
            evidence_mode=metadata.get("evidence_mode") or "any_sufficient",
        )
        stage, _ = attribute_stage(
            alignment=alignment,
            answer_is_correct=correct,
            is_answerable=answerable,
            abstained=row.abstained,
            n_retrieved=len(record.retrieved),
        )
        ev_attribution.append(str(stage))

    return {
        "document_level_success": doc_level,
        "evidence_level_success": ev_level,
        "document_level_attribution": doc_attribution,
        "evidence_level_attribution": ev_attribution,
    }


def compare(records, rows) -> dict:
    """Paired comparison of document-level and evidence-level judgements."""
    conditions = build_conditions(records, rows)
    doc = conditions["document_level_success"]
    ev = conditions["evidence_level_success"]

    paired = [(d, e) for d, e in zip(doc, ev, strict=True) if d is not None and e is not None]
    n = len(paired)
    doc_hits = sum(1 for d, _ in paired if d)
    ev_hits = sum(1 for _, e in paired if e)

    # Discordant pairs. only_doc is the interesting cell: rows the conventional
    # metric calls a retrieval success and evidence alignment calls a miss.
    only_doc = sum(1 for d, e in paired if d and not e)
    only_ev = sum(1 for d, e in paired if e and not d)

    from collections import Counter

    return {
        "n_paired": n,
        "document_level_success_rate": round(doc_hits / n, 4) if n else None,
        "evidence_level_success_rate": round(ev_hits / n, 4) if n else None,
        "document_level_ci": wilson_proportion_ci(doc_hits, n).as_dict() if n else None,
        "evidence_level_ci": wilson_proportion_ci(ev_hits, n).as_dict() if n else None,
        "gap_percentage_points": round(100 * (doc_hits - ev_hits) / n, 2) if n else None,
        "discordant_document_only": only_doc,
        "discordant_evidence_only": only_ev,
        "mcnemar": mcnemar_exact(only_doc, only_ev).as_dict(),
        "paired_bootstrap_difference": paired_bootstrap_difference(
            [1.0 if d else 0.0 for d, _ in paired],
            [1.0 if e else 0.0 for _, e in paired],
        ).as_dict(),
        "attribution_document_level": dict(Counter(conditions["document_level_attribution"])),
        "attribution_evidence_level": dict(Counter(conditions["evidence_level_attribution"])),
        "interpretation": (
            "discordant_document_only counts questions where the conventional "
            "document-level metric reports a retrieval success but the labelled "
            "supporting span was never placed in the generator's context. Those "
            "rows are the ones a document-level evaluation misattributes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare evaluation methodologies on one run")
    parser.add_argument("--records", required=True, help="inference.jsonl from an experiment")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--tag", default="", help="label for this comparison")
    args = parser.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"no records at {records_path}", file=sys.stderr)
        return 1

    records = read_records(records_path)
    rows = score_records(records, TaxonomyConfig())
    result = {
        "tag": args.tag or records_path.parent.name,
        "source_records": str(records_path),
        "n_records": len(records),
        "comparison": compare(records, rows),
        "provenance": collect_provenance(
            note="Methodology comparison. Scored from stored records; no model was called."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    c = result["comparison"]
    print(f"\n=== {result['tag']} (n={c['n_paired']}) ===")
    print(f"  document-level retrieval success : {c['document_level_success_rate']}")
    print(f"  evidence-level retrieval success : {c['evidence_level_success_rate']}")
    print(f"  gap                              : {c['gap_percentage_points']} pp")
    print(f"  document-says-yes/evidence-says-no: {c['discordant_document_only']}")
    print(f"  evidence-says-yes/document-says-no: {c['discordant_evidence_only']}")
    print(f"  McNemar p                        : {c['mcnemar']['p_value']}")
    print(f"  attribution (document-level)     : {c['attribution_document_level']}")
    print(f"  attribution (evidence-level)     : {c['attribution_evidence_level']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
