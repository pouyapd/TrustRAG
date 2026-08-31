"""Decompose the disagreement between retrieval-success definitions.

Retrieval and generation run once; every definition is then applied to the same
stored records, so differences are attributable to the measurement and the
comparison is paired by construction.

**Why three conditions and not two.** An earlier version of this script
compared only the conventional document-level definition against the
span-level one. On multi-hop data that comparison changes *two* things at
once — the granularity (document to span) and the quantifier (any relevant
document to every required document) — so the resulting gap cannot be
attributed to either. The conditions below separate them:

    A  document-level, ANY relevant document        (the conventional metric)
    B  document-level, honouring the evidence mode  (A -> B isolates quantifier)
    C  span-level,     honouring the evidence mode  (B -> C isolates granularity)

For a single-hop dataset the evidence mode is `any_sufficient`, so A and B are
identical by definition and the whole A->C gap is granularity. For a multi-hop
dataset they differ, and the decomposition shows which effect is doing the
work. Reporting only A->C would attribute a multi-hop quantifier effect to
span granularity, which is a different claim about a different mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evidence import (
    EvidenceStatus,
    align_evidence,
    attribute_stage,
    retrieved_from_chunks,
    spans_from_records,
)
from src.evaluation.provenance import collect_provenance
from src.evaluation.records import read_records
from src.evaluation.runner import score_records
from src.evaluation.statistics import (
    mcnemar_exact,
    paired_bootstrap_difference,
    wilson_proportion_ci,
)
from src.evaluation.taxonomy import TaxonomyConfig


def evidence_mode_of(record) -> str:
    return (record.metadata or {}).get("evidence_mode") or "any_sufficient"


def gold_spans_of(record):
    return spans_from_records((record.metadata or {}).get("supporting_spans") or [])


def condition_a_document_any(record) -> bool | None:
    """Conventional: did any retrieved chunk come from any relevant document?"""
    gold = gold_spans_of(record)
    if not gold:
        return None
    gold_docs = {s.doc_id for s in gold}
    retrieved = set(record.retrieved_doc_ids[: record.top_k or 5])
    return bool(gold_docs & retrieved)


def condition_b_document_quantified(record) -> bool | None:
    """Document-level, but honouring the question's evidence mode.

    Under `all_required` every gold document must appear among the retrieved
    documents. Under `any_sufficient` this is identical to condition A, which
    is why single-hop datasets show a zero A->B step rather than a spurious one.
    """
    gold = gold_spans_of(record)
    if not gold:
        return None
    gold_docs = {s.doc_id for s in gold}
    retrieved = set(record.retrieved_doc_ids[: record.top_k or 5])
    if evidence_mode_of(record) == "all_required":
        return gold_docs.issubset(retrieved)
    return bool(gold_docs & retrieved)


def condition_c_span(record) -> bool | None:
    """Span-level: did a retrieved chunk actually contain the gold evidence?"""
    gold = gold_spans_of(record)
    if not gold:
        return None
    alignment = align_evidence(
        gold, retrieved_from_chunks(record.retrieved), evidence_mode_of(record)
    )
    return alignment.status is EvidenceStatus.COMPLETE


def paired_step(before: list[bool], after: list[bool], label: str) -> dict:
    """One methodological step, with the paired statistics it supports."""
    n = len(before)
    n_before, n_after = sum(before), sum(after)
    only_before = sum(1 for b, a in zip(before, after, strict=True) if b and not a)
    only_after = sum(1 for b, a in zip(before, after, strict=True) if a and not b)
    test = mcnemar_exact(only_before, only_after)
    boot = paired_bootstrap_difference(
        [1.0 if b else 0.0 for b in before], [1.0 if a else 0.0 for a in after]
    )
    return {
        "step": label,
        "n": n,
        "rate_before": round(n_before / n, 4) if n else None,
        "rate_after": round(n_after / n, 4) if n else None,
        "absolute_gap_pp": round(100 * (n_before - n_after) / n, 2) if n else None,
        # Relative gap: what fraction of the apparent successes disappear.
        "relative_gap": (
            round((n_before - n_after) / n_before, 4) if n_before else None
        ),
        "discordant_lost": only_before,
        "discordant_gained": only_after,
        "mcnemar": test.as_dict(),
        "paired_bootstrap_difference": boot.as_dict(),
    }


def compare(records, rows) -> dict:
    """Three conditions, two isolated steps, and the covariate that explains them."""
    triples = []
    for record in records:
        a = condition_a_document_any(record)
        b = condition_b_document_quantified(record)
        c = condition_c_span(record)
        if a is None or b is None or c is None:
            continue
        triples.append((a, b, c))

    if not triples:
        return {"n_paired": 0, "note": "no rows carry gold evidence spans"}

    A = [t[0] for t in triples]
    B = [t[1] for t in triples]
    C = [t[2] for t in triples]
    n = len(triples)

    # How many chunks a gold document spans is what determines whether
    # document-level and span-level can differ at all: a document that is one
    # chunk long cannot show a granularity effect.
    chunks_per_doc = [r.n_relevant_chunks for r in records if r.n_relevant_chunks]
    median_chunks = (
        sorted(chunks_per_doc)[len(chunks_per_doc) // 2] if chunks_per_doc else None
    )

    modes = Counter(evidence_mode_of(r) for r in records if gold_spans_of(r))

    return {
        "n_paired": n,
        "evidence_modes": dict(modes),
        "median_chunks_per_relevant_document": median_chunks,
        "conditions": {
            "A_document_any": round(sum(A) / n, 4),
            "B_document_quantified": round(sum(B) / n, 4),
            "C_span_quantified": round(sum(C) / n, 4),
        },
        "confidence_intervals": {
            "A_document_any": wilson_proportion_ci(sum(A), n).as_dict(),
            "B_document_quantified": wilson_proportion_ci(sum(B), n).as_dict(),
            "C_span_quantified": wilson_proportion_ci(sum(C), n).as_dict(),
        },
        "steps": {
            "quantifier_A_to_B": paired_step(A, B, "any -> all_required (document level)"),
            "granularity_B_to_C": paired_step(B, C, "document -> span (same quantifier)"),
            "total_A_to_C": paired_step(A, C, "conventional -> span-level"),
        },
        "attribution": attribution_comparison(records, rows),
        "interpretation": (
            "A->B isolates the quantifier: whether a metric requires every document a "
            "multi-hop question needs, or accepts any one of them. B->C isolates "
            "granularity: whether retrieving the document is treated as retrieving the "
            "evidence. On a single-hop dataset A and B coincide by definition, so the "
            "whole gap is granularity. On a dataset whose documents are only one or two "
            "chunks long the granularity step is necessarily small, because retrieving "
            "the document nearly guarantees retrieving the span."
        ),
    }


def attribution_comparison(records, rows) -> dict:
    """Where failures are charged, under the conventional and evidence views."""
    doc_stage, evidence_stage = [], []
    for record, row in zip(records, rows, strict=True):
        answerable = bool(record.relevant_doc_ids)
        correct = bool(row.answer_exact_match) or (
            row.key_fact_recall is not None and row.key_fact_recall >= 1.0
        )
        doc_ok = condition_a_document_any(record)
        if not answerable:
            doc_stage.append("none" if row.abstained else "generation")
        elif doc_ok is False:
            doc_stage.append("retrieval")
        else:
            doc_stage.append("none" if correct else "generation")

        alignment = align_evidence(
            gold_spans_of(record),
            retrieved_from_chunks(record.retrieved),
            evidence_mode_of(record),
        )
        stage, _ = attribute_stage(
            alignment=alignment,
            answer_is_correct=correct,
            is_answerable=answerable,
            abstained=row.abstained,
            n_retrieved=len(record.retrieved),
        )
        evidence_stage.append(str(stage))
    return {
        "document_level": dict(Counter(doc_stage)),
        "evidence_level": dict(Counter(evidence_stage)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose retrieval-definition disagreement")
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
            note="Methodology decomposition. Scored from stored records; no model was called."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    c = result["comparison"]
    cond, steps = c["conditions"], c["steps"]
    print(f"\n=== {result['tag']} (n={c['n_paired']}) ===")
    print(f"  evidence modes: {c['evidence_modes']}")
    print(f"  median chunks per relevant document: {c['median_chunks_per_relevant_document']}")
    print(f"  A  document-level, ANY        : {cond['A_document_any']}")
    print(f"  B  document-level, quantified : {cond['B_document_quantified']}")
    print(f"  C  span-level,     quantified : {cond['C_span_quantified']}")
    for key, label in (("quantifier_A_to_B", "quantifier  A->B"),
                       ("granularity_B_to_C", "granularity B->C"),
                       ("total_A_to_C", "total       A->C")):
        s = steps[key]
        p = s["mcnemar"]["p_value"]
        p_text = "n/a (no discordant pairs)" if p is None else f"{p:.3g}"
        print(f"  {label}: {s['absolute_gap_pp']:>6} pp   discordant {s['discordant_lost']}/"
              f"{s['discordant_gained']}   McNemar p={p_text}")
    print(f"\n  attribution document-level: {c['attribution']['document_level']}")
    print(f"  attribution evidence-level: {c['attribution']['evidence_level']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
