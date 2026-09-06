#!/usr/bin/env python
"""Ask whether gold-span annotations under-cover the evidence that answers a question.

Span-gated attribution charges a failure to retrieval when no retrieved chunk
overlaps a gold evidence span. That rule is only as good as the gold spans. If a
corpus marks one supporting sentence but the answer is also derivable from other
retrieved text, the rule reports a retrieval failure that did not happen.

This measures how often that occurs, using a deliberately crude lexical proxy: for
every answerable unit with zero gold-span coverage, what fraction of the reference
answer's content words appear anywhere in the retrieved context? High coverage does
not prove the answer was derivable -- token overlap is not entailment -- so the
result is an upper bound on how often the span rule is wrong, not a correction.

    python scripts/audit_gold_span_coverage.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --out reports/annotation/qasper_dev_300_full_context/audit/gold_span_coverage.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MIN_TOKEN_LEN = 4
HIGH = 0.8
PARTIAL = 0.5


def content_tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            if len(t) >= MIN_TOKEN_LEN]


def span_covered(unit: dict) -> int:
    covered = 0
    for g in unit.get("gold_evidence") or []:
        gs, ge = g["char_range"]
        for c in unit.get("retrieved_context") or []:
            if c["doc_id"] != g["doc_id"]:
                continue
            cs, ce = c["char_range"]
            if min(ge, ce) - max(gs, cs) > 0:
                covered += 1
                break
    return covered


def answer_presence(unit: dict) -> float:
    """Best fraction of any reference answer's content words present in the context."""
    context = " ".join(c.get("text", "") for c in unit.get("retrieved_context") or []).lower()
    best = 0.0
    for answer in unit.get("reference_answers") or []:
        tokens = content_tokens(answer)
        if not tokens:
            continue
        best = max(best, sum(1 for t in tokens if t in context) / len(tokens))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sheet = [json.loads(line) for line
             in (Path(args.package) / "annotation_sheet.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]

    uncovered, detail = [], []
    for unit in sheet:
        if not unit["corpus_can_answer"]:
            continue
        if span_covered(unit) > 0:
            continue
        presence = answer_presence(unit)
        uncovered.append(presence)
        detail.append({
            "annotation_id": unit["annotation_id"],
            "n_gold_spans": len(unit.get("gold_evidence") or []),
            "answer_token_presence": round(presence, 4),
            "band": "high" if presence >= HIGH else ("partial" if presence >= PARTIAL else "low"),
        })

    n = len(uncovered)
    high = sum(1 for p in uncovered if p >= HIGH)
    partial = sum(1 for p in uncovered if PARTIAL <= p < HIGH)
    report = {
        "package": Path(args.package).as_posix(),
        "method": "lexical upper bound; content words of length >= 4, substring match "
                  "against the concatenated retrieved context",
        "caveat": "Token presence is not entailment. These counts bound how often the "
                  "span rule can be wrong; they do not establish that the answer was "
                  "derivable from the retrieved text, and they are not a correction.",
        "n_answerable_units_with_zero_gold_span_coverage": n,
        "reference_answer_present": {
            "high_ge_0.8": high,
            "partial_0.5_to_0.8": partial,
            "low_lt_0.5": n - high - partial,
        },
        "share_high": round(high / n, 4) if n else None,
        "share_high_or_partial": round((high + partial) / n, 4) if n else None,
        "units": sorted(detail, key=lambda d: -d["answer_token_presence"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"answerable units with zero gold-span coverage : {n}")
    print(f"  reference answer present (>= {HIGH})            : {high} ({100 * high / n:.1f}%)")
    print(f"  partially present ({PARTIAL}-{HIGH})                  : {partial} ({100 * partial / n:.1f}%)")
    print(f"  not present (< {PARTIAL})                          : {n - high - partial}")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
