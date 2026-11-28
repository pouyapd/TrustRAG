#!/usr/bin/env python
"""Agreement between two adjudicators of QASPER gold-span completeness.

Reads CSVs with columns annotation_id, answer (YES / NO / CANNOT_TELL). Annotator A may be
given as several files (the author's two rounds). Prints raw agreement, Cohen's kappa,
PABAK and Gwet's AC1 on the units both judged, and under-coverage over the 133-unit
population under several combination rules. Refuses to run on an empty B file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

POPULATION = 133


def read(paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = next(k for k in row if k.startswith("annotation_id"))
                ans_key = next(k for k in row if k.startswith("answer"))
                ans = (row[ans_key] or "").strip().upper()
                if ans:
                    out[row[key].strip()] = ans
    return out


def stats(a: dict[str, str], b: dict[str, str]) -> dict:
    common = sorted(set(a) & set(b))
    binary = [(a[u] == "YES", b[u] == "YES") for u in common if "CANNOT_TELL" not in (a[u], b[u])]
    n = len(binary)
    yy = sum(1 for x, y in binary if x and y)
    nn = sum(1 for x, y in binary if not x and not y)
    yn = sum(1 for x, y in binary if x and not y)
    ny = sum(1 for x, y in binary if not x and y)
    po = (yy + nn) / n
    pa, pb = (yy + yn) / n, (yy + ny) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    pi = (pa + pb) / 2
    pe_ac1 = 2 * pi * (1 - pi)
    ac1 = (po - pe_ac1) / (1 - pe_ac1)
    return {"units_both_judged": len(common), "units_binary": n, "table": {"YES/YES": yy, "YES/NO": yn,
            "NO/YES": ny, "NO/NO": nn}, "raw_agreement": po, "cohen_kappa": kappa, "pabak": 2 * po - 1,
            "gwet_ac1": ac1, "disagreements": [u for u in common if a[u] != b[u]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    a, b = read(args.a), read([args.b])
    if not b:
        raise SystemExit("annotator B file has no answers; nothing to score (no labels are invented)")
    out = {"agreement": stats(a, b)}
    yes_a = {u for u, v in a.items() if v == "YES"}
    yes_b = {u for u, v in b.items() if v == "YES"}
    out["under_coverage_over_133"] = {
        "A": len(yes_a) / POPULATION, "B": len(yes_b) / POPULATION,
        "either": len(yes_a | yes_b) / POPULATION, "both": len(yes_a & yes_b) / POPULATION}
    unread = [u for u in b if u not in a]
    out["B_on_units_A_never_read"] = {"n": len(unread), "yes": sum(b[u] == "YES" for u in unread)}
    print(json.dumps(out, indent=1))
    Path("agreement_result.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
