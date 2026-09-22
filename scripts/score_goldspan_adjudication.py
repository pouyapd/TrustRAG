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

``--remainder`` adds the second package (the 27 unresolved units the sample did not
draw, plus the 10 bucket-B units the proxies resolved on their own). It is the same
estimator, not a new one: once a stratum is fully adjudicated the finite population
correction is zero, so the stratified estimate degenerates to a direct count and the
sampling error vanishes. Bucket B stops being an assumption and becomes observed
labels. Bucket A -- the 36 units both proxies called 'answer absent' -- is still
unchecked, so it stays an assumption and is reported as a sensitivity band rather
than folded into a confidence interval.

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


def read_answers(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.lower().startswith("annotation_id"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            out[parts[0]] = parts[1].upper()
    return out


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


def combine(remainder: Path, audit: dict, per_stratum: dict, first_answers: dict[str, str],
            report: dict) -> dict:
    """Fold the remaining units in: the unresolved part becomes a census.

    Returns the combined result, or a refusal describing what is still missing. The
    estimator is unchanged -- with n_h = N_h the finite population correction is zero,
    so `stratified` returns the exact count with zero variance.
    """
    manifest = json.loads((remainder / "manifest.json").read_text(encoding="utf-8"))
    groups = manifest["ids_by_group"]
    expected = {i for ids in groups.values() for i in ids}
    answers = read_answers(remainder / "answers.csv")

    overlap = sorted(expected & set(first_answers))
    missing = sorted(expected - set(answers))
    unknown = sorted(set(answers) - expected)
    bad = {i: a for i, a in answers.items() if a not in ("YES", "NO", "CANNOT_TELL")}
    if overlap or missing or unknown or bad:
        return {
            "status": "not scored",
            "units_expected": len(expected),
            "units_labelled": len(expected) - len(missing),
            "still_unlabelled": len(missing),
            "first_few_unlabelled": missing[:5],
            "overlap_with_first_pass": overlap,
            "ids_not_in_the_package": unknown[:5],
            "invalid_answers": {k: v for k, v in list(bad.items())[:5]},
            "note": "Fill in answers.csv in the remainder package, then re-run with "
                    "--remainder to obtain the census estimate.",
        }

    total_units = audit["n_units"]
    bucket_a = audit["buckets"].get("A_genuinely_unsupported", 0)
    proxy_present_ids = groups.get("B_supported_outside_gold_span", [])
    unresolved_ids = {b: ids for b, ids in groups.items()
                      if b != "B_supported_outside_gold_span"}

    # unresolved strata: first-pass answers + the remaining ones = the whole stratum
    per_group, census = {}, {}
    for bucket, v in per_stratum.items():
        extra = [answers[i] for i in unresolved_ids.get(bucket, [])]
        yes = v["yes"] + extra.count("YES")
        no = v["no"] + extra.count("NO")
        ct = v["cannot_tell"] + extra.count("CANNOT_TELL")
        per_group[bucket] = {"population": v["population"], "adjudicated": yes + no + ct,
                             "from_first_pass": v["sampled"], "from_remainder": len(extra),
                             "yes": yes, "no": no, "cannot_tell": ct}
        census[bucket] = (v["population"], yes + no + ct, yes)

    b_answers = [answers[i] for i in proxy_present_ids]
    per_group["B_supported_outside_gold_span"] = {
        "population": len(proxy_present_ids), "adjudicated": len(b_answers),
        "from_first_pass": 0, "from_remainder": len(b_answers),
        "yes": b_answers.count("YES"), "no": b_answers.count("NO"),
        "cannot_tell": b_answers.count("CANNOT_TELL"),
        "previously": "counted as under-coverage on the proxies' word, unchecked",
    }

    scenarios = {}
    for name, treat in (("decidable_only", None), ("cannot_tell_as_yes", "YES"),
                        ("cannot_tell_as_no", "NO")):
        counts = {}
        for bucket, v in per_group.items():
            if bucket == "B_supported_outside_gold_span":
                continue
            if treat is None:
                n, yes = v["yes"] + v["no"], v["yes"]
            else:
                n = v["adjudicated"]
                yes = v["yes"] + (v["cannot_tell"] if treat == "YES" else 0)
            counts[bucket] = (v["population"], n, yes)
        est, se = stratified(counts)
        b = per_group["B_supported_outside_gold_span"]
        b_yes = b["yes"] + (b["cannot_tell"] if treat == "YES" else 0)
        rate = (est + b_yes) / total_units
        scenarios[name] = {
            "yes_in_unresolved_87": round(est, 2),
            "yes_in_bucket_B": b_yes,
            "under_coverage_count": round(est + b_yes, 2),
            "under_coverage_rate": round(rate, 4),
            "sampling_standard_error": round(se / total_units, 4),
        }

    decided = [a for g in per_group.values() for a in
               ["YES"] * g["yes"] + ["NO"] * g["no"]]
    observed_rate = decided.count("YES") / len(decided) if decided else 0.0
    point = scenarios["decidable_only"]["under_coverage_count"]
    sensitivity = {}
    for label, assumed in (("bucket_a_all_correct_0pct", 0.0),
                           ("bucket_a_behaves_like_adjudicated_units", observed_rate),
                           ("bucket_a_10pct", 0.10), ("bucket_a_25pct", 0.25)):
        rate = (point + bucket_a * assumed) / total_units
        sensitivity[label] = {"assumed_yes_rate_in_bucket_A": round(assumed, 4),
                              "under_coverage_rate": round(rate, 4)}

    prev = report["estimates"]["decidable_only"]
    return {
        "status": "scored",
        "design": "census of the 87 unresolved units (60 sampled + "
                  f"{sum(len(v) for v in unresolved_ids.values())} remaining) plus direct "
                  "adjudication of the 10 bucket-B units; the 36 bucket-A units remain "
                  "unchecked and are assumed to contain no under-coverage",
        "sampling_error": "zero - every unresolved unit is adjudicated, so the finite "
                          "population correction is 1 - n/N = 0 in every stratum",
        "provenance": "same reviewer (the author) as the first pass; not an independent "
                      "second annotation and no inter-annotator agreement",
        "per_group": per_group,
        "adjudicated_units": sum(g["adjudicated"] for g in per_group.values()),
        "units_still_unadjudicated": bucket_a,
        "estimates": scenarios,
        "sensitivity_to_bucket_A": {
            "what_varies": "the 36 units both proxies called 'answer absent' were never "
                           "checked by a person; this varies their true rate",
            "scenarios": sensitivity,
        },
        "comparison_with_sampled_estimate": {
            "previous_point_estimate": prev["under_coverage_rate"],
            "previous_ci95_sampling_only": prev["ci95"],
            "previous_bucket_b_assumption": "all 10 counted as under-coverage",
            "new_point_estimate": scenarios["decidable_only"]["under_coverage_rate"],
            "change": round(scenarios["decidable_only"]["under_coverage_rate"]
                            - prev["under_coverage_rate"], 4),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--remainder", default="",
                    help="package of the units the first pass left open "
                         "(build_goldspan_remainder.py); makes the unresolved part a census")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pkg = Path(args.package)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))

    answers = read_answers(pkg / "answers.csv")

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

    # --- sensitivity to the census part ------------------------------------
    # The 46 "resolved" units were settled by two uncalibrated proxies, not by a
    # human. The 10 counted as under-coverage carry the point estimate, and the
    # adjudication just gave us the first evidence about how those proxies behave:
    # on units they could not resolve, the human agreed with "answer present" far
    # less often than the proxies claimed for bucket B. That gap is a larger source
    # of uncertainty than the sampling error, so it is reported rather than buried.
    flat_all = [answers[i] for ids in strata.values() for i in ids]
    decided = [a for a in flat_all if a != "CANNOT_TELL"]
    human_yes_rate = (flat_all.count("YES") / len(decided)) if decided else 0.0
    base_counts = {b: (v["population"], v["yes"] + v["no"], v["yes"])
                   for b, v in per_stratum.items()}
    est_unres, se_unres = stratified(base_counts)
    half_unres = Z * se_unres / total_units
    sensitivity = {}
    for label, assumed in (("proxies_correct_100pct", 1.0), ("75pct", 0.75),
                           ("50pct", 0.50), ("25pct", 0.25), ("proxies_wrong_0pct", 0.0),
                           ("bucket_b_behaves_like_adjudicated_units", human_yes_rate)):
        b = known_yes * assumed
        rate = (b + est_unres) / total_units
        sensitivity[label] = {
            "assumed_yes_rate_in_bucket_B": round(assumed, 4),
            "under_coverage_rate": round(rate, 4),
            "ci95_sampling_error_only": [round(max(0.0, rate - half_unres), 4),
                                         round(min(1.0, rate + half_unres), 4)],
        }

    flat = flat_all
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
        "sensitivity_to_unverified_census_units": {
            "what_varies": "the 10 units both proxies called under-coverage were never "
                           "checked by a human; this varies their true rate",
            "why_it_matters": "this uncertainty is larger than the sampling error and is "
                              "not covered by the confidence intervals above",
            "scenarios": sensitivity,
        },
        "reporting_guidance":
            "Quote `cannot_tell_as_yes` and `cannot_tell_as_no` as bounds when the "
            "undecidable share exceeds ~10%; otherwise `decidable_only` with the "
            "undecidable count stated alongside. The interval covers sampling error "
            "only -- it does not cover annotator error, and with one annotator there "
            "is no way to estimate that from these data.",
    }

    if args.remainder:
        report["combined_with_remaining_units"] = combine(
            Path(args.remainder), audit, per_stratum, answers, report)

    print(f"answers: YES {flat.count('YES')} · NO {flat.count('NO')} · "
          f"CANNOT_TELL {flat.count('CANNOT_TELL')}  (n={len(flat)})\n")
    for name, s in scenarios.items():
        print(f"  {name:20} rate={s['under_coverage_rate']:.3f} "
              f"95% CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    print("")
    print("  sensitivity to the 10 unverified proxy-resolved units:")
    for label, sc in sensitivity.items():
        print(f"    {label:42} rate={sc['under_coverage_rate']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {Path(args.out).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
