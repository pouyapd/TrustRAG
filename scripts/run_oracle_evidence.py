#!/usr/bin/env python
"""A paired oracle-evidence control: does supplying the missing evidence fix the answer?

The repository's own limitations section says stage attribution is a declared
mapping rather than a causal claim, and that establishing causality needs a
controlled oracle-context ablation. This is that ablation.

Every question is answered twice by the same generator, with the same prompt, at
the same decoding settings. The only thing that changes is the context:

    retrieved   the chunks the retriever actually returned
    oracle      the gold supporting spans, verbatim

Because the pairing is within-question, the comparison is not confounded by
question difficulty, document, or generator. The informative subset is the
questions whose evidence never arrived under retrieval: if supplying it flips them
to correct, missing evidence explains the failure; if it does not, the failure was
the generator's.

    python scripts/run_oracle_evidence.py \
        --records reports/experiments/qasper_dev_300/inference.jsonl \
        --generator qwen0.5b --limit 150 \
        --out reports/experiments/oracle_qasper_qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.correctness import key_fact_recall  # noqa: E402
from src.evaluation.evidence import (  # noqa: E402
    EvidenceStatus,
    align_evidence,
    retrieved_from_chunks,
    spans_from_records,
)
from src.evaluation.records import InferenceRecord  # noqa: E402
from src.evaluation.statistics import mcnemar_exact, wilson_proportion_ci  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.rag.local_llm import build_generator  # noqa: E402

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context does not contain the answer, say that it does not."
)


def is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in ("does not contain", "not contain the answer",
                                  "cannot answer", "no information", "not provided",
                                  "unable to answer", "i don't know", "i do not know"))


def stratum(record: InferenceRecord) -> str:
    gold = spans_from_records((record.metadata or {}).get("supporting_spans") or [])
    if not gold:
        return "no_gold_evidence"
    mode = (record.metadata or {}).get("evidence_mode") or "any_sufficient"
    alignment = align_evidence(gold, retrieved_from_chunks(record.retrieved), mode)
    if alignment.status is EvidenceStatus.COMPLETE:
        return "COMPLETE"
    if alignment.status is EvidenceStatus.PARTIAL:
        return "PARTIAL"
    gold_docs = {s.doc_id for s in gold}
    if gold_docs & {c.doc_id for c in record.retrieved}:
        return "NONE_DOC_HIT"
    return "NONE"


def retrieved_context(record: InferenceRecord) -> str:
    return "\n\n".join(f"[{c.doc_id}]\n{c.text}" for c in record.retrieved)


def oracle_context(record: InferenceRecord) -> str:
    """The gold supporting spans, verbatim, in the order the dataset records them."""
    spans = (record.metadata or {}).get("supporting_spans") or []
    return "\n\n".join(f"[{s['doc_id']}]\n{s.get('text', '')}" for s in spans)


def answer_once(generator, question: str, context: str) -> tuple[str, float]:
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    started = time.perf_counter()
    try:
        answer = generator.generate(SYSTEM_PROMPT, user)
    except Exception:  # noqa: BLE001 - one bad call must not end the run
        answer = ""
    return answer, (time.perf_counter() - started) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--generator", default="qwen0.5b")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log = get_logger("oracle")
    records = [InferenceRecord.from_dict(json.loads(line)) for line
               in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if (r.metadata or {}).get("supporting_spans")]
    if args.limit:
        records = records[:args.limit]

    try:
        generator, identity = build_generator(args.generator)
    except Exception as exc:  # noqa: BLE001
        print(f"generator unavailable: {exc}", file=sys.stderr)
        return 1

    rows = []
    for i, record in enumerate(records, start=1):
        reference = record.reference_answer or ""
        r_ans, r_ms = answer_once(generator, record.question, retrieved_context(record))
        o_ans, o_ms = answer_once(generator, record.question, oracle_context(record))
        rows.append({
            "question_id": (record.metadata or {}).get("question_id"),
            "stratum": stratum(record),
            "retrieved": {
                "answer": r_ans,
                "key_fact_recall": key_fact_recall(r_ans, reference),
                "abstained": is_refusal(r_ans),
                "latency_ms": round(r_ms, 1),
            },
            "oracle": {
                "answer": o_ans,
                "key_fact_recall": key_fact_recall(o_ans, reference),
                "abstained": is_refusal(o_ans),
                "latency_ms": round(o_ms, 1),
            },
        })
        if i % 10 == 0:
            log.info("oracle_progress", done=i, total=len(records))

    def correct(side: dict) -> bool:
        return (side.get("key_fact_recall") or 0.0) >= 1.0

    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        if not n:
            return {"n": 0}
        rc = sum(correct(r["retrieved"]) for r in subset)
        oc = sum(correct(r["oracle"]) for r in subset)
        only_o = sum(1 for r in subset if correct(r["oracle"]) and not correct(r["retrieved"]))
        only_r = sum(1 for r in subset if correct(r["retrieved"]) and not correct(r["oracle"]))
        return {
            "n": n,
            "retrieved_correct": rc,
            "oracle_correct": oc,
            "retrieved_rate": round(rc / n, 4),
            "oracle_rate": round(oc / n, 4),
            "difference_pp": round(100 * (oc - rc) / n, 1),
            "retrieved_ci": wilson_proportion_ci(rc, n).as_dict(),
            "oracle_ci": wilson_proportion_ci(oc, n).as_dict(),
            "only_oracle_correct": only_o,
            "only_retrieved_correct": only_r,
            "paired_test": mcnemar_exact(only_r, only_o).as_dict(),
            "retrieved_abstained": sum(r["retrieved"]["abstained"] for r in subset),
            "oracle_abstained": sum(r["oracle"]["abstained"] for r in subset),
        }

    strata = sorted({r["stratum"] for r in rows})
    report = {
        "design": "within-question paired comparison; identical generator, prompt and "
                  "decoding, only the context differs (retrieved chunks vs gold spans)",
        "records": Path(args.records).as_posix(),
        "generator": identity,
        "n": len(rows),
        "overall": summarise(rows),
        "by_stratum": {s: summarise([r for r in rows if r["stratum"] == s]) for s in strata},
        "caveat": "The oracle context is shorter and cleaner than retrieved context, so "
                  "part of any gain may come from reduced distraction rather than from the "
                  "evidence itself. This bounds the causal reading rather than establishing it.",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    o = report["overall"]
    print(f"\nn={o['n']}  retrieved {o['retrieved_rate']:.3f} -> oracle {o['oracle_rate']:.3f} "
          f"({o['difference_pp']:+.1f} pp, p={o['paired_test']['p_value']:.4g})")
    for s, v in report["by_stratum"].items():
        if v["n"]:
            print(f"  {s:14} n={v['n']:3}  {v['retrieved_rate']:.3f} -> {v['oracle_rate']:.3f} "
                  f"({v['difference_pp']:+.1f} pp, p={v['paired_test']['p_value']:.4g})")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
