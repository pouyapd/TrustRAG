#!/usr/bin/env python
"""Estimate gold-span under-coverage from the completed adjudication sample.

The 133 zero-coverage units split into a census part and a sampled part:

  46 units were settled by agreement of two proxies (36 answer absent, 10 answer
     present outside the gold span) and are counted directly, without error;
  87 units were unresolved, of which 60 were sampled in three strata and are
     adjudicated by a human here.

The estimator is therefore a stratified estimate over the 87, added to the known
count from the 46. Variance comes only from the sampled part, with a finite
population correction -- sampling 60 of 87 removes most of the sampling error, and
ignoring the fpc would overstate the interval substantially.

CANNOT_TELL is not discarded silently. Three figures are reported: the rate among
units the annotator could decide, and the two bounds obtained by counting every
CANNOT_TELL as YES and then as NO. The bounds are the honest headline whenever the
undecidable share is large.

    python scripts/score_goldspan_adjudication.py \
        --package reports/annotation/goldspan_adjudication \
        --audit reports/annotation/qasper_dev_300_full_context/audit/gold_span_semantic.json \
        --out reports/annotation/goldspan_adjudication/estimate.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959963985


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = successes / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def stratified(counts: dict[str, tuple[int, int, int]]) -> tuple[float, float]:
    """Estimated YES count over the unresolved population, and its standard error.

    counts maps stratum -> (N_h population, n_h sampled, yes_h). Uses the standard
    stratified total estimator with a finite population correction per stratum.
    """
    total, variance = 0.0, 0.0
    for _, (big_n, n, yes) in counts.items():
        if n == 0:
            continue
        p = yes / n
        total += big_n * p
        if n > 1:
            fpc = 1 - n / big_n
            variance += (big_n ** 2) * fpc * p * (1 - p) / (n - 1)
    return total, math.sqrt(variance)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pkg = Path(args.package)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))

    answers: dict[str, str] = {}
    for line in (pkg / "answers.csv").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.lower().startswith("annotation_id"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            answers[parts[0]] = parts[1].upper()

    expected = {i for ids in manifest["selected_ids_by_stratum"].values() for i in ids}
    missing = sorted(expected - set(answers))
    unknown = sorted(set(answers) - expected)
    bad = {i: a for i, a in answers.items() if a not in ("YES", "NO", "CANNOT_TELL")}
    if missing or unknown or bad:
        print("cannot score yet:")
        if missing:
            print(f"  {len(missing)} unit(s) still unlabelled, first few: {missing[:5]}")
        if unknown:
            print(f"  ids not in the sample: {unknown[:5]}")
        if bad:
            print(f"  invalid answers: {list(bad.items())[:5]}")
        return 1

    strata = manifest["selected_ids_by_stratum"]
    sizes = manifest["population"]["stratum_sizes"]
    per_stratum = {}
    for bucket, ids in strata.items():
        a = [answers[i] for i in ids]
        per_stratum[bucket] = {
            "population": sizes[bucket],
            "sampled": len(ids),
            "yes": a.count("YES"),
            "no": a.count("NO"),
            "cannot_tell": a.count("CANNOT_TELL"),
        }

    known_yes = audit["buckets"].get("B_supported_outside_gold_span", 0)
    total_units = audit["n_units"]

    scenarios = {}
    for name, treat in (("decidable_only", None), ("cannot_tell_as_yes", "YES"),
                        ("cannot_tell_as_no", "NO")):
        counts = {}
        for bucket, v in per_stratum.items():
            if treat is None:
                n = v["yes"] + v["no"]
                yes = v["yes"]
            else:
                n = v["sampled"]
                yes = v["yes"] + (v["cannot_tell"] if treat == "YES" else 0)
            counts[bucket] = (v["population"], n, yes)
        est, se = stratified(counts)
        rate = (known_yes + est) / total_units
        half = Z * se / total_units
        scenarios[name] = {
            "estimated_yes_in_unresolved": round(est, 2),
            "under_coverage_count": round(known_yes + est, 2),
            "under_coverage_rate": round(rate, 4),
            "ci95": [round(max(0.0, rate - half), 4), round(min(1.0, rate + half), 4)],
            "standard_error": round(se / total_units, 4),
        }

    flat = [answers[i] for ids in strata.values() for i in ids]
    p, lo, hi = wilson(flat.count("YES"), len(flat) - flat.count("CANNOT_TELL")
                       if flat.count("CANNOT_TELL") < len(flat) else len(flat))

    report = {
        "note": "Under-coverage = the span rule reports a retrieval failure on a unit "
                "where the answer was in fact derivable from the retrieved text.",
        "population_total": total_units,
        "counted_directly": {"under_coverage": known_yes,
                             "no_under_coverage": audit["buckets"].get("A_genuinely_unsupported", 0)},
        "sampled": {"population": manifest["population"]["unresolved_eligible_for_sampling"],
                    "n": len(flat),
                    "answers": {"YES": flat.count("YES"), "NO": flat.count("NO"),
                                "CANNOT_TELL": flat.count("CANNOT_TELL")}},
        "per_stratum": per_stratum,
        "estimates": scenarios,
        "unstratified_check": {"sample_yes_rate_among_decidable": round(p, 4),
                               "wilson_ci95": [round(lo, 4), round(hi, 4)]},
        "reporting_guidance":
            "Quote `cannot_tell_as_yes` and `cannot_tell_as_no` as bounds when the "
            "undecidable share exceeds ~10%; otherwise `decidable_only` with the "
            "undecidable count stated alongside. The interval covers sampling error "
            "only -- it does not cover annotator error, and with one annotator there "
            "is no way to estimate that from these data.",
    }

    print(f"answers: YES {flat.count('YES')} · NO {flat.count('NO')} · "
          f"CANNOT_TELL {flat.count('CANNOT_TELL')}  (n={len(flat)})\n")
    for name, s in scenarios.items():
        print(f"  {name:20} rate={s['under_coverage_rate']:.3f} "
              f"95% CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {Path(args.out).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
