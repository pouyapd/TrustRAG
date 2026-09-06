#!/usr/bin/env python
"""Second opinion on whether zero-gold-span units really lack usable evidence.

`audit_gold_span_coverage.py` answers this lexically, which over-fires on shared
vocabulary and under-fires on paraphrase. This adds two independent signals over the
same units and reports where they agree:

  lexical    fraction of the reference answer's content words present in the context
  semantic   max cosine similarity between the reference answer and any retrieved
             sentence, using the same MiniLM encoder the retriever uses

Neither is entailment, and agreement between two weak proxies is not truth. The
point is to bracket the quantity rather than to settle it: units both signals call
supported are the strongest candidates for gold-span under-coverage, and units both
call unsupported are the strongest candidates for genuine retrieval failure. The
band in between is exactly what needs human adjudication, and is reported as such.

    python scripts/audit_gold_span_semantic.py \
        --package reports/annotation/qasper_dev_300_full_context \
        --out reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_TOKEN_LEN = 4
LEXICAL_HIGH = 0.8
SEMANTIC_HIGH = 0.60
SEMANTIC_LOW = 0.35
SENTENCE = re.compile(r"(?<=[.!?])\s+")


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


def classify(lexical: float, semantic: float) -> tuple[str, str]:
    """Bucket a unit, and say what a human adjudicator would need to settle it."""
    if lexical >= LEXICAL_HIGH and semantic >= SEMANTIC_HIGH:
        return ("B_supported_outside_gold_span",
                "both signals agree the answer content is in the retrieved text")
    if lexical < 0.5 and semantic < SEMANTIC_LOW:
        return ("A_genuinely_unsupported",
                "both signals agree the answer content is absent")
    if semantic >= SEMANTIC_HIGH > lexical:
        return ("C_possibly_inferable",
                "semantically close but lexically different; may be paraphrase or may be "
                "topical similarity without the fact")
    if lexical >= LEXICAL_HIGH > semantic:
        return ("D_ambiguous_lexical_only",
                "shared vocabulary without semantic support; likely incidental overlap")
    return ("D_ambiguous", "signals disagree or both are mid-range")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, util

    sheet = [json.loads(line) for line
             in (Path(args.package) / "annotation_sheet.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    targets = [u for u in sheet if u["corpus_can_answer"] and span_covered(u) == 0]
    print(f"units to examine: {len(targets)}")

    model = SentenceTransformer(args.model)
    detail = []
    for n, unit in enumerate(targets, 1):
        context = " ".join(c.get("text", "") for c in unit.get("retrieved_context") or [])
        sentences = [s.strip() for s in SENTENCE.split(context) if len(s.strip()) > 20]
        answers = [a for a in (unit.get("reference_answers") or []) if a.strip()]

        lexical = 0.0
        low = context.lower()
        for answer in answers:
            tokens = content_tokens(answer)
            if tokens:
                lexical = max(lexical, sum(1 for t in tokens if t in low) / len(tokens))

        semantic = 0.0
        if answers and sentences:
            a_emb = model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
            s_emb = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
            semantic = float(util.cos_sim(a_emb, s_emb).max())

        bucket, why = classify(lexical, semantic)
        detail.append({
            "annotation_id": unit["annotation_id"],
            "lexical_presence": round(lexical, 4),
            "semantic_max_cosine": round(semantic, 4),
            "bucket": bucket,
            "rationale": why,
        })
        if n % 25 == 0:
            print(f"  {n}/{len(targets)}")

    counts: dict[str, int] = {}
    for d in detail:
        counts[d["bucket"]] = counts.get(d["bucket"], 0) + 1

    n = len(detail)
    report = {
        "package": Path(args.package).as_posix(),
        "encoder": args.model,
        "thresholds": {"lexical_high": LEXICAL_HIGH, "semantic_high": SEMANTIC_HIGH,
                       "semantic_low": SEMANTIC_LOW},
        "n_units": n,
        "buckets": counts,
        "shares": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "interpretation": {
            "A_genuinely_unsupported": "retrieval failure that the span rule reports correctly",
            "B_supported_outside_gold_span": "the span rule reports a retrieval failure that "
                                             "the retrieved text does not support; gold-span "
                                             "under-coverage",
            "C_possibly_inferable": "needs human adjudication",
            "D_ambiguous": "needs human adjudication",
            "D_ambiguous_lexical_only": "needs human adjudication",
        },
        "caveat": "Both signals are proxies. Neither establishes entailment, and the "
                  "semantic threshold is a convention, not a calibrated decision boundary. "
                  "Bucket B is a lower bound on under-coverage under agreement of two weak "
                  "signals; the C/D bands are explicitly unresolved and require human "
                  "adjudication that has not been performed.",
        "units": sorted(detail, key=lambda d: -d["semantic_max_cosine"]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    for bucket, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:34} {count:3}  ({100 * count / n:.1f}%)")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
