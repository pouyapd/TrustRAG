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
infer anything from position, and each annotator receives an independently
shuffled order so two sheets cannot be compared by row number.

**Boundary cases are sampled on purpose.** A uniform sample is dominated by
rows the rules get trivially right, which inflates agreement and teaches
nothing. A share of the budget is therefore reserved for rows whose deciding
feature sits within a small margin of the threshold that classified them —
exactly the rows where a tuned constant is doing the work, and where human
judgement is most informative about whether the constant is defensible.

**Two annotators, same units.** Cohen's kappa needs both annotators to label
the same items independently; the sheets differ only in order.
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


def span_length(start: int | None, end: int | None) -> int | None:
    """Characters the recorded range covers, or None when offsets are absent."""
    if start is None or end is None:
        return None
    return end - start


def is_complete(text: str, start: int | None, end: int | None) -> bool | None:
    """Whether `text` is the whole span its offsets claim.

    None means unverifiable: without offsets there is nothing to compare the
    text against, and silently calling that "complete" is how the previous
    truncation went unnoticed for a whole annotation round.
    """
    length = span_length(start, end)
    return None if length is None else len(text) == length


def describe_chunk(chunk) -> dict:
    """One retrieved chunk, complete.

    The text stored here is the entire chunk the retriever returned and the
    generator was shown — never an excerpt. An annotator answering step 2 of
    the guidelines ("did the evidence reach the system?") is judging what the
    system received, so showing them a prefix of it makes the answer to that
    question unknowable: evidence past the cut looks identical to evidence that
    was never retrieved. `n_chars` and `text_complete` are recorded so the
    claim is checkable rather than asserted.
    """
    return {
        "rank": chunk.rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "char_range": [chunk.start_char, chunk.end_char],
        "n_chars": len(chunk.text),
        "text_complete": is_complete(chunk.text, chunk.start_char, chunk.end_char),
        "text": chunk.text,
    }


def describe_span(span: dict) -> dict:
    """One gold supporting span, complete, on the same terms as a chunk."""
    text = str(span.get("text", ""))
    start, end = span["start_char"], span["end_char"]
    return {
        "doc_id": span["doc_id"],
        "char_range": [start, end],
        "n_chars": len(text),
        "text_complete": is_complete(text, start, end),
        "text": text,
    }


def truncation_problems(units: list[dict]) -> list[str]:
    """Every place a unit shows less text than its offsets promise.

    Called before the sheet is written. A package that cannot show the whole
    retrieved chunk is not a weaker package, it is an invalid instrument for
    step 2, so this aborts the build instead of warning.
    """
    problems = []
    for unit in units:
        for chunk in unit["retrieved_context"]:
            length = span_length(*chunk["char_range"])
            if length is not None and chunk["n_chars"] < length:
                problems.append(
                    f"{unit['annotation_id']} rank {chunk['rank']}: "
                    f"{chunk['n_chars']} chars stored for a {length}-char range"
                )
        for span in unit["gold_evidence"]:
            length = span_length(*span["char_range"])
            if length is not None and span["n_chars"] < length:
                problems.append(
                    f"{unit['annotation_id']} gold {span['doc_id']}: "
                    f"{span['n_chars']} chars stored for a {length}-char range"
                )
    return problems


def context_integrity(units: list[dict]) -> dict:
    """Counts that let a reader verify the no-truncation claim from the sheet.

    `unverifiable` is reported separately from `complete` on purpose: a chunk
    without offsets cannot be shown to be whole, and rolling the two together
    would turn missing evidence into a clean bill of health.
    """
    chunks = [c for u in units for c in u["retrieved_context"]]
    spans = [g for u in units for g in u["gold_evidence"]]
    def tally(items):
        return {
            "n": len(items),
            "complete": sum(1 for i in items if i["text_complete"] is True),
            "truncated": sum(1 for i in items if i["text_complete"] is False),
            "unverifiable_no_offsets": sum(
                1 for i in items if i["text_complete"] is None
            ),
            "max_chars": max((i["n_chars"] for i in items), default=0),
        }
    return {"retrieved_chunks": tally(chunks), "gold_spans": tally(spans)}


def build_unit(record, row, index: int) -> dict:
    """One annotation unit: everything needed to judge, nothing that anchors.

    The annotator sees the question, the reference answers, whether the corpus
    is supposed to be able to answer it, the retrieved context in rank order —
    each chunk complete, exactly as retrieved — and the system's answer. They
    do not see the proposed failure mode, the metric values, or the decision
    features.
    """
    evidence = [describe_chunk(chunk) for chunk in record.retrieved]

    gold_spans = (record.metadata or {}).get("supporting_spans") or []
    return {
        "annotation_id": f"unit_{index:04d}",
        "question": record.question,
        "reference_answers": row.reference_answers or [record.reference_answer],
        "corpus_can_answer": bool(record.relevant_doc_ids),
        "gold_evidence": [describe_span(s) for s in gold_spans],
        "retrieved_context": evidence,
        "system_answer": record.predicted_answer,
        # To be completed by a human. Left empty on purpose.
        "human_label": "",
        "human_notes": "",
        "human_confidence": "",
    }


#: How close a deciding feature has to be to its threshold to count as a
#: boundary case. 0.1 on a 0-1 scale: wide enough to find rows, narrow enough
#: that they really are near the line.
BOUNDARY_MARGIN = 0.10


def boundary_distance(row, config: TaxonomyConfig) -> float | None:
    """How near this row sits to the threshold that decided it.

    Returns the smallest distance to any threshold the classifier consulted, or
    None when nothing numeric was in play. Rows with a small distance are the
    ones where a tuned constant, rather than an obvious fact, chose the label.
    """
    distances = []
    if row.key_fact_recall is not None:
        distances.append(abs(row.key_fact_recall - config.key_fact_recall_incorrect))
        distances.append(abs(row.key_fact_recall - config.key_fact_recall_ok))
    if row.answer_f1_normalized is not None:
        distances.append(abs(row.answer_f1_normalized - config.answer_f1_ok))
    faithfulness = (row.decision_features or {}).get("faithfulness")
    if faithfulness is not None:
        distances.append(abs(faithfulness - config.faithfulness_threshold))
    return min(distances) if distances else None


def stratified_sample(rows, n_units: int, seed: int, config: TaxonomyConfig,
                      min_per_mode: int = 8, boundary_share: float = 0.25):
    """Choose the annotation units.

    Three competing needs, in priority order:

    1. every proposed failure mode must appear, or the confusion matrix has
       empty rows and per-category recall is undefined;
    2. boundary cases must be over-represented, because that is where the
       thresholds are actually load-bearing;
    3. what remains should look like the population, so the reweighted
       estimates mean something.

    The sampling weight per mode is recorded so population proportions can be
    recovered; the sample itself is deliberately *not* representative.
    """
    rng = random.Random(seed)
    by_mode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_mode[row.failure_mode_v2].append(index)

    chosen: set[int] = set()
    strata: dict[str, dict] = {}

    # 1. floor for every mode present
    for mode, indices in sorted(by_mode.items()):
        take = min(min_per_mode, len(indices))
        chosen.update(rng.sample(indices, take))
        strata[mode] = {"population": len(indices), "sampled_floor": take}

    # 2. boundary cases
    ranked = sorted(
        (
            (d, i) for i, row in enumerate(rows)
            if (d := boundary_distance(row, config)) is not None and d <= BOUNDARY_MARGIN
        ),
        key=lambda pair: pair[0],
    )
    boundary_budget = int(n_units * boundary_share)
    boundary_taken = 0
    for _, index in ranked:
        if len(chosen) >= n_units or boundary_taken >= boundary_budget:
            break
        if index not in chosen:
            chosen.add(index)
            boundary_taken += 1

    # 3. fill the rest proportionally to the population
    remaining = [i for i in range(len(rows)) if i not in chosen]
    rng.shuffle(remaining)
    for index in remaining:
        if len(chosen) >= n_units:
            break
        chosen.add(index)

    order = sorted(chosen)
    rng.shuffle(order)

    for mode, indices in by_mode.items():
        sampled = sum(1 for i in indices if i in chosen)
        strata[mode]["sampled"] = sampled
        strata[mode]["weight"] = (
            round(len(indices) / sampled, 4) if sampled else None
        )
    return order, strata, {"boundary_units": boundary_taken,
                           "boundary_budget": boundary_budget,
                           "boundary_margin": BOUNDARY_MARGIN}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human annotation package")
    parser.add_argument("--records", required=True, help="inference.jsonl from an experiment")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-units", type=int, default=200,
                        help="target number of annotation units; the per-mode floor may push "
                             "the final count above it, and the manifest records the actual")
    parser.add_argument("--min-per-mode", type=int, default=8,
                        help="floor per proposed failure mode, so no category is empty")
    parser.add_argument("--boundary-share", type=float, default=0.25,
                        help="fraction of the budget reserved for near-threshold rows")
    parser.add_argument("--annotators", default="a,b",
                        help="comma-separated annotator ids; each gets its own shuffled sheet")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"no records at {records_path}", file=sys.stderr)
        return 1

    config = TaxonomyConfig()
    records = read_records(records_path)
    rows = score_records(records, config)
    chosen, weights, boundary_info = stratified_sample(
        rows, args.n_units, args.seed, config,
        min_per_mode=args.min_per_mode, boundary_share=args.boundary_share,
    )

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

    problems = truncation_problems(units)
    if problems:
        print(
            f"refusing to write: {len(problems)} retrieved chunk(s) or gold span(s) "
            "hold less text than their char_range covers",
            file=sys.stderr,
        )
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    (out_dir / "annotation_sheet.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n", encoding="utf-8"
    )
    # One sheet per annotator: the same units in an independently shuffled
    # order, so two returned sheets cannot be aligned by position — only by
    # annotation_id — and neither annotator sees the other's sequence.
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    for offset, annotator in enumerate(annotators):
        shuffled = list(units)
        random.Random(args.seed + 1 + offset).shuffle(shuffled)
        annotator_dir = out_dir / f"annotator_{annotator}"
        annotator_dir.mkdir(parents=True, exist_ok=True)
        (annotator_dir / "annotation_sheet.jsonl").write_text(
            "\n".join(json.dumps(u, ensure_ascii=False) for u in shuffled) + "\n",
            encoding="utf-8",
        )

    (out_dir / "proposed_labels_key.jsonl").write_text(
        "\n".join(json.dumps(k, ensure_ascii=False) for k in key) + "\n", encoding="utf-8"
    )

    manifest = {
        "n_units": len(units),
        "n_units_requested": args.n_units,
        "min_per_mode": args.min_per_mode,
        "boundary_sampling": boundary_info,
        "annotators": annotators,
        "seed": args.seed,
        "sampling_weights": weights,
        "guidelines": "docs/ANNOTATION_GUIDELINES.md",
        "retrieved_context_policy": (
            "Every retrieved chunk is stored complete, exactly as the retriever "
            "returned it and the generator saw it. No excerpting. Each chunk "
            "carries n_chars and text_complete so the claim can be checked."
        ),
        "context_integrity": context_integrity(units),
        "scoring_command": (
            "python scripts/score_annotations.py --package <this dir> "
            "--annotator a=<path> --annotator b=<path>"
        ),
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
    print(f"  boundary units: {boundary_info['boundary_units']} "
          f"(within {BOUNDARY_MARGIN} of a deciding threshold)")
    print("  annotation_sheet.jsonl        - master copy (human_label empty)")
    for annotator in annotators:
        print(f"  annotator_{annotator}/annotation_sheet.jsonl - independently shuffled")
    print("  proposed_labels_key.jsonl     - withheld until annotation is complete")
    print("  manifest.json                 - sampling weights and protocol")
    print("\nGuidelines: docs/ANNOTATION_GUIDELINES.md")
    print("No labels were generated. Human annotation is still required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
