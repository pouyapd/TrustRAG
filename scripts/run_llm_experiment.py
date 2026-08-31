"""Does evidence status predict how a real language model fails?

The retrieval and evidence results in this repository are produced with a
deterministic extractive control. That control is the right baseline but it
cannot hallucinate, refuse, or waffle, so it cannot answer the question this
script exists for: when the passage that supports the answer never reaches the
generator, what does a *language model* do?

Design. Retrieval is not re-run. The script reads the stored records of a
finished experiment, rebuilds the exact context each question was given, and
swaps in a different generator. Retrieval, chunking, embedder, corpus and
questions are therefore identical by construction, and any difference is the
generator's. Every question is then assigned to a stratum by the evidence
status already computed for it, and outcomes are reported per stratum:

    COMPLETE          every required gold span reached the generator
    PARTIAL           some but not all of it did (multi-hop, or a clipped span)
    NONE_DOC_HIT      a chunk of the right document arrived, but not the span
    NONE              nothing from any gold document arrived

`NONE_DOC_HIT` is separated out deliberately: it is exactly the case a
document-level retrieval metric scores as a success, so it is where a
conventional evaluation would blame the model for a retrieval problem.

Terminology, kept deliberately narrow. An answer that does not match the
reference is *incorrect*, not a hallucination. The measured behaviour is
`answered_without_evidence`: the model produced a substantive answer when the
supporting evidence was provably absent from its context. That is a property of
the transcript, not an inference about the model's internal state, and it is
the quantity the attribution hierarchy actually needs.

    python scripts/run_llm_experiment.py \
        --records reports/experiments/qasper_dev_300/inference.jsonl \
        --generator qwen0.5b --limit 150 \
        --out reports/experiments/llm_qasper_qwen

Generators: `mock` (control, no download), `qwen0.5b` / `smollm360m` (local
open weights, no credentials), or `openai:MODEL` / `anthropic:MODEL` (hosted,
requires the matching key in the environment). Nothing here runs in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evidence import (
    EvidenceStatus,
    align_evidence,
    retrieved_from_chunks,
    spans_from_records,
)
from src.evaluation.provenance import collect_provenance
from src.evaluation.records import read_records, write_records
from src.evaluation.runner import aggregate, score_records, write_outputs
from src.evaluation.statistics import (
    permutation_test_distributions,
    wilson_proportion_ci,
)
from src.evaluation.taxonomy import TaxonomyConfig
from src.logging_setup import get_logger, setup_logging
from src.rag.local_llm import build_generator
from src.rag.pipeline import SYSTEM_PROMPT

#: Substantive refusal markers. Kept small and literal: a longer list starts
#: silently reclassifying real answers that happen to contain a hedge.
REFUSAL_MARKERS = (
    "cannot answer",
    "can't answer",
    "not contain",
    "no information",
    "insufficient context",
    "unable to answer",
    "does not provide",
    "not provided in the context",
)


def format_context(record) -> str:
    """Rebuild the exact context string the original run put in the prompt."""
    if not record.retrieved:
        return "(no context available)"
    return "\n\n".join(f"[source: {c.source}]\n{c.text}" for c in record.retrieved)


def abstained(answer: str) -> bool:
    """Whether the model declined rather than answered."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evidence_stratum(record) -> str:
    """Which evidence condition this question was actually in.

    Computed from the stored retrieval, so it is a property of the run rather
    than of the generator, and is identical across every generator compared.
    """
    gold = spans_from_records((record.metadata or {}).get("supporting_spans") or [])
    if not gold:
        return "no_gold_evidence"
    mode = (record.metadata or {}).get("evidence_mode") or "any_sufficient"
    alignment = align_evidence(gold, retrieved_from_chunks(record.retrieved), mode)
    if alignment.status is EvidenceStatus.COMPLETE:
        return "COMPLETE"
    if alignment.status is EvidenceStatus.PARTIAL:
        return "PARTIAL"
    # No span coverage at all. Did a chunk of a gold document still arrive?
    gold_docs = {s.doc_id for s in gold}
    if gold_docs & {c.doc_id for c in record.retrieved}:
        return "NONE_DOC_HIT"
    return "NONE"


def regenerate(records, generator, log, sleep_s: float = 0.0):
    """Answer every question again with a different generator.

    Retrieval is reused verbatim. Failures are recorded per record rather than
    aborting the run, so one API error does not discard an hour of work.
    """
    out, latencies, errors = [], [], 0
    for i, record in enumerate(records, start=1):
        context = format_context(record)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {record.question}\n\nANSWER:"
        started = time.perf_counter()
        try:
            answer = generator.generate(SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
            errors += 1
            log.warning("generation_failed", index=i, error=str(exc)[:160])
            answer = ""
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        # Faithfulness is left unset: the only judge available here would be the
        # same small model that wrote the answer, and self-judged faithfulness
        # is not evidence. Grounding is measured structurally instead, by
        # whether the evidence was in the context at all.
        out.append(_replaced(record, answer, latency_ms))
        if i % 25 == 0:
            log.info("generated", done=i, total=len(records),
                     mean_ms=round(sum(latencies) / len(latencies)))
        if sleep_s:
            time.sleep(sleep_s)
    return out, latencies, errors


def _replaced(record, answer: str, latency_ms: float):
    """A copy of an inference record carrying a new answer."""
    import copy

    clone = copy.deepcopy(record)
    clone.predicted_answer = answer
    clone.faithfulness = None
    clone.latency_ms = round(latency_ms, 2)
    return clone


def stratified_outcomes(records, rows) -> dict:
    """Outcomes conditioned on what evidence actually reached the generator."""
    strata: dict[str, list[dict]] = {}
    for record, row in zip(records, rows, strict=True):
        correct = bool(row.answer_exact_match) or (
            row.key_fact_recall is not None and row.key_fact_recall >= 1.0
        )
        strata.setdefault(evidence_stratum(record), []).append(
            {
                "correct": correct,
                "abstained": abstained(record.predicted_answer),
                "empty": not record.predicted_answer.strip(),
                "key_fact_recall": row.key_fact_recall,
                "answer_f1": row.answer_f1_normalized,
            }
        )

    summary = {}
    for name, items in sorted(strata.items()):
        n = len(items)
        n_correct = sum(1 for x in items if x["correct"])
        n_abstained = sum(1 for x in items if x["abstained"])
        # The behaviour of interest: answered anyway, with no evidence present.
        answered = [x for x in items if not x["abstained"] and not x["empty"]]
        recalls = [x["key_fact_recall"] for x in items if x["key_fact_recall"] is not None]
        summary[name] = {
            "n": n,
            "correct_rate": wilson_proportion_ci(n_correct, n).as_dict(),
            "abstention_rate": wilson_proportion_ci(n_abstained, n).as_dict(),
            "answered_rate": wilson_proportion_ci(len(answered), n).as_dict(),
            "mean_key_fact_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        }
    return summary


def evidence_predicts_correctness(records, rows) -> dict:
    """The headline test: does having the evidence change the outcome?

    COMPLETE against everything else. These are independent groups rather than
    paired observations — the same question cannot be in both — so this is a
    permutation test over the two label distributions, not McNemar.
    """
    have, lack = [], []
    for record, row in zip(records, rows, strict=True):
        stratum = evidence_stratum(record)
        if stratum == "no_gold_evidence":
            continue
        correct = bool(row.answer_exact_match) or (
            row.key_fact_recall is not None and row.key_fact_recall >= 1.0
        )
        label = "correct" if correct else "incorrect"
        (have if stratum == "COMPLETE" else lack).append(label)

    if not have or not lack:
        return {"note": "one side of the comparison is empty", "n_complete": len(have),
                "n_incomplete": len(lack)}

    test = permutation_test_distributions(have, lack, n_permutations=10000)
    p_have = sum(1 for x in have if x == "correct") / len(have)
    p_lack = sum(1 for x in lack if x == "correct") / len(lack)
    return {
        "n_evidence_complete": len(have),
        "n_evidence_incomplete": len(lack),
        "correct_rate_evidence_complete": round(p_have, 4),
        "correct_rate_evidence_incomplete": round(p_lack, 4),
        "absolute_difference_pp": round(100 * (p_have - p_lack), 2),
        "wilson_complete": wilson_proportion_ci(
            sum(1 for x in have if x == "correct"), len(have)
        ).as_dict(),
        "wilson_incomplete": wilson_proportion_ci(
            sum(1 for x in lack if x == "correct"), len(lack)
        ).as_dict(),
        "permutation_test": test.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-generate a finished run with a real LLM and stratify by evidence"
    )
    parser.add_argument("--records", required=True, help="inference.jsonl from an experiment")
    parser.add_argument("--generator", required=True,
                        help="mock | qwen0.5b | smollm360m | openai:MODEL | anthropic:MODEL")
    parser.add_argument("--limit", type=int, default=0, help="first N records (0 = all)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds between calls, for hosted rate limits")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    setup_logging()
    log = get_logger("llm_experiment")

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"no records at {records_path}", file=sys.stderr)
        return 1

    records = read_records(records_path)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("no records to run", file=sys.stderr)
        return 1

    try:
        generator, generator_identity = build_generator(args.generator)
    except (ValueError, RuntimeError) as exc:
        print(f"generator unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"re-generating {len(records)} answers with {generator_identity['name']}")
    print("retrieval is reused verbatim; only the generator changes\n")

    started = time.time()
    new_records, latencies, errors = regenerate(records, generator, log, args.sleep)
    elapsed = time.time() - started

    taxonomy_config = TaxonomyConfig()
    rows = score_records(new_records, taxonomy_config)
    report = aggregate(rows, taxonomy_config=taxonomy_config)

    report["generation_experiment"] = {
        "tag": args.tag or f"{records_path.parent.name}_{args.generator}",
        "source_records": str(records_path),
        "n": len(new_records),
        "generator": generator_identity,
        "generation_errors": errors,
        "runtime_seconds": round(elapsed, 1),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "latency_ms_max": round(max(latencies), 1) if latencies else None,
        "retrieval": "reused verbatim from the source run; not re-executed",
        "evidence_strata": stratified_outcomes(new_records, rows),
        "evidence_predicts_correctness": evidence_predicts_correctness(new_records, rows),
        "stratum_counts": dict(Counter(evidence_stratum(r) for r in new_records)),
        "terminology": (
            "'answered_rate' counts substantive answers, not hallucinations. An "
            "answer that does not match the reference is incorrect; no claim is "
            "made about why. Faithfulness is unset because the only available "
            "judge would be the model under test."
        ),
    }
    report["provenance"] = collect_provenance(
        pipeline={"generator": generator_identity, "retrieval": "reused"},
        taxonomy={"version": taxonomy_config.version,
                  "fingerprint": taxonomy_config.fingerprint()},
    )

    out_dir = Path(args.out)
    write_outputs(rows, report, out_dir, records=new_records)
    write_records(new_records, out_dir / "inference.jsonl")
    (out_dir / "generation_summary.json").write_text(
        json.dumps(report["generation_experiment"], indent=2), encoding="utf-8"
    )

    ge = report["generation_experiment"]
    print(f"\n=== {ge['tag']} ===")
    print(f"generator      : {generator_identity['name']}")
    print(f"errors         : {errors}")
    print(f"mean latency   : {ge['latency_ms_mean']} ms   total {ge['runtime_seconds']}s")
    print(f"\n{'stratum':<16}{'n':>5}{'correct':>10}{'abstained':>11}{'answered':>10}")
    for name, s in ge["evidence_strata"].items():
        print(f"{name:<16}{s['n']:>5}{_pct(s['correct_rate']):>10}"
              f"{_pct(s['abstention_rate']):>11}{_pct(s['answered_rate']):>10}")
    head = ge["evidence_predicts_correctness"]
    if "absolute_difference_pp" in head:
        p = head["permutation_test"].get("p_value")
        print(f"\ncorrect | evidence complete  : {head['correct_rate_evidence_complete']}")
        print(f"correct | evidence incomplete: {head['correct_rate_evidence_incomplete']}")
        print(f"difference: {head['absolute_difference_pp']} pp   permutation p={p}")
    print(f"\nwrote {out_dir}")
    return 0


def _pct(estimate: dict) -> str:
    point = estimate.get("point")
    return "n/a" if point is None else f"{100 * point:.1f}%"


if __name__ == "__main__":
    sys.exit(main())
