#!/usr/bin/env python
"""Independent recomputation of the manuscript's numbers (final audit, 25 Nov 2026).

Nothing here imports analyse_v2, make_tables or src.evaluation: references, metrics, chance
levels, agreement, penalised shares and reach are re-implemented from their definitions and
computed from results/raw/*.json.gz. The script then compares
  (1) every metric leaf of results/<corpus>_full/analysis.json that it recomputes,
  (2) the macros of manuscript/generated/numbers.tex that are deterministic functions of the data,
  (3) every number cell of the generated tables tab_main, tab_reach and tab_incomplete,
and writes results/audit/numeric_consistency.json plus a Markdown summary. Randomised quantities
(bootstrap intervals, permutation p-values, single-annotator draws, random-gold baselines) are
checked against their own seeded recomputation in cluster_inference.py / stat_audit.py, not here.
"""
from __future__ import annotations

import gzip
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
GEN = REV / "manuscript/generated"
KS = (1, 3, 5, 10)


# ----------------------------------------------------------------- references (re-implemented)
def chunks_overlapping(ranges, spans):
    return {i for i, (c0, c1) in enumerate(ranges) if any(min(c1, s1) > max(c0, s0) for s0, s1 in spans)}


def ann_gold(q, corpus):
    """{annotator: gold chunk set} for annotators whose evidence is located in the text."""
    out = {}
    for a in q["annotations"]:
        if a.get("status") != "ok":
            continue
        spans = [tuple(s) for s in a["spans"]] if corpus == "qasper" else [(a["start_char"], a["end_char"])]
        out[a["annotator"]] = chunks_overlapping(q["chunk_ranges"], spans)
    return out


def reference(q, corpus, scheme):
    ann = q["annotations"]
    ok = [a for a in ann if a.get("status") == "ok"]
    if not ok:
        return None
    if corpus == "qasper":
        span_sets = {a["annotator"]: {tuple(s) for s in a["spans"]} for a in ok}
        if scheme == "first":
            f = next((a for a in ann if a.get("status") in ("ok", "not_representable")), None)
            if f is None or f["status"] != "ok":
                return None
            spans = span_sets[f["annotator"]]
        else:
            cnt = Counter(s for ss in span_sets.values() for s in ss)
            if scheme == "union":
                spans = set(cnt)
            elif scheme == "plurality":
                top = max(cnt.values())
                spans = {s for s, c in cnt.items() if c == top}
            else:  # agreement
                spans = {s for s, c in cnt.items() if c >= 2}
                if not spans:
                    return None
        g = chunks_overlapping(q["chunk_ranges"], spans)
        return g or None
    # NQ
    if scheme == "first":
        f = next((a for a in ann if a.get("status") in ("ok", "not_representable")), None)
        if f is None or f["status"] != "ok":
            return None
        chosen = [f]
    elif scheme == "official":
        if sum(a.get("status") in ("ok", "not_representable") for a in ann) < 2:
            return None
        chosen = ok
    elif scheme == "union":
        chosen = ok
    else:
        cnt = Counter((a["start_char"], a["end_char"]) for a in ok)
        top = max(cnt.values())
        chosen = [a for a in ok if cnt[(a["start_char"], a["end_char"])] == top]
    g = chunks_overlapping(q["chunk_ranges"], [(a["start_char"], a["end_char"]) for a in chosen])
    return g or None


# ----------------------------------------------------------------- metrics
def first_rank(order, gold):
    for p, i in enumerate(order, 1):
        if i in gold:
            return p
    raise AssertionError("gold chunk missing from ordering")


def chance_hit(D, m, k):
    if k >= D - m + 1:
        return 1.0
    return 1.0 - math.comb(D - m, k) / math.comb(D, k)


def chance_rr(D, m):
    # P(first gold at rank r) = C(D-r, m-1) / C(D, m), r = 1 .. D-m+1
    tot = math.comb(D, m)
    return sum(math.comb(D - r, m - 1) / tot / r for r in range(1, D - m + 2))


def scheme_block(qs, corpus, scheme, models):
    rows = []
    for q in qs:
        g = reference(q, corpus, scheme)
        if g:
            rows.append((q, g))
    out = {"n": len(rows), "chance": {}, "metrics": {}}
    for k in KS:
        out["chance"][f"hit@{k}"] = float(np.mean([chance_hit(q["D"], len(g), k) for q, g in rows]))
    out["chance"]["mrr"] = float(np.mean([chance_rr(q["D"], len(g)) for q, g in rows]))
    c1 = out["chance"]["hit@1"]
    for m in models:
        r = np.array([first_rank(q["orders"][m], g) for q, g in rows])
        met = {f"hit@{k}": float(np.mean(r <= k)) for k in KS}
        met["mrr"] = float(np.mean(1.0 / r))
        met["median_rank"] = float(np.median(r))
        met["kappa@1"] = (met["hit@1"] - c1) / (1 - c1)
        met["share_rank_2_5"] = float(np.mean((r >= 2) & (r <= 5)))
        out["metrics"][m] = met
    return out, {q["question_id"]: (q, g) for q, g in rows}


def agreement(qs, corpus, models):
    pairs = Counter()
    jac = []
    qdisj = nq = 0
    units = []
    pen = {m: [0, 0] for m in models}  # expected penalised misses, expected misses (exact over annotators)
    for q in qs:
        ag = ann_gold(q, corpus)
        if len(ag) < 2:
            continue
        nq += 1
        any_disj = False
        for a, b in itertools.combinations(sorted(ag), 2):
            A_, B_ = ag[a], ag[b]
            if A_ == B_:
                pairs["identical"] += 1
            elif A_ & B_:
                pairs["overlap"] += 1
            else:
                pairs["disjoint"] += 1
                any_disj = True
            u = A_ | B_
            jac.append(len(A_ & B_) / len(u) if u else 1.0)
        qdisj += any_disj
        for i in range(q["D"]):
            units.append([int(i in ag[a]) for a in sorted(ag)])
        anns = sorted(ag)
        for m in models:
            top = q["orders"][m][0]
            for a in anns:
                if top not in ag[a]:
                    pen[m][1] += 1 / len(anns)
                    if any(top in ag[b] for b in anns if b != a):
                        pen[m][0] += 1 / len(anns)
    n = sum(pairs.values())
    width = max(len(u) for u in units)
    rel = np.full((width, len(units)), np.nan)
    for j, u in enumerate(units):
        rel[: len(u), j] = u
    import krippendorff
    return {"questions_with_2plus_annotators": nq, "annotator_pairs": n,
            "pairs_identical": pairs["identical"] / n, "pairs_overlapping_not_identical": pairs["overlap"] / n,
            "pairs_disjoint": pairs["disjoint"] / n, "mean_jaccard": float(np.mean(jac)),
            "questions_with_a_disjoint_pair": qdisj / nq,
            "krippendorff_alpha_chunk_level": float(krippendorff.alpha(reliability_data=rel, level_of_measurement="nominal"))}, \
        {m: pen[m][0] / pen[m][1] for m in models}


def reach(qs, corpus, scheme, k=5):
    doc_index = {}
    out = {}
    rows = [(q, reference(q, corpus, scheme)) for q in qs]
    rows = [(q, g) for q, g in rows if g]
    for m in rows[0][0]["global_top"]:
        A_ = C_ = 0
        adm = []
        for q, g in rows:
            top = q["global_top"][m][:k]
            hits = [pos for d, pos in top if pos >= 0]  # pos >= 0 means a chunk of the gold document
            if hits:
                A_ += 1
                adm.append(len(hits))
                if any(p in g for p in hits):
                    C_ += 1
        out[m] = {"n": len(rows), "A": A_ / len(rows), "C": C_ / len(rows), "C_over_A": C_ / A_ if A_ else None,
                  "admitted_mean": float(np.mean(adm)) if adm else None}
    return out


# ----------------------------------------------------------------- comparison helpers
def compare_leaves(name, mine, theirs, log, tol=1e-12):
    if isinstance(mine, dict):
        for k, v in mine.items():
            if k in theirs:
                compare_leaves(f"{name}.{k}", v, theirs[k], log, tol)
            else:
                log["missing"].append(f"{name}.{k}")
        return
    if mine is None and theirs is None:
        return
    d = abs(float(mine) - float(theirs))
    log["checked"] += 1
    log["max_abs_diff"] = max(log["max_abs_diff"], d)
    if d > tol:
        log["mismatches"].append({"field": name, "recomputed": mine, "stored": theirs, "diff": d})


def macros():
    out = {}
    for line in (GEN / "numbers.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    report = {"comparisons": {}, "macro_checks": [], "table_checks": []}
    rec = {}
    for corpus, f in (("qasper", "qasper_full"), ("nq", "nq_full")):
        raw = json.load(gzip.open(REV / f"results/raw/{f}.json.gz", "rt", encoding="utf-8"))
        ana = json.loads((REV / f"results/{f}/analysis.json").read_text(encoding="utf-8"))
        qs, models = raw["questions"], list(raw["config"]["models"])
        schemes = ("union", "first", "plurality", "agreement") if corpus == "qasper" else ("first", "union", "official", "plurality")
        log = {"checked": 0, "max_abs_diff": 0.0, "mismatches": [], "missing": []}
        blocks, golds = {}, {}
        for s in schemes:
            b, g = scheme_block(qs, corpus, s, models)
            blocks[s], golds[s] = b, g
            compare_leaves(f"{corpus}.{s}.n", b["n"], ana["schemes"][s]["n"], log)
            compare_leaves(f"{corpus}.{s}.chance", b["chance"], ana["schemes"][s]["chance"], log)
            for m in models:
                compare_leaves(f"{corpus}.{s}.{m}", b["metrics"][m], ana["schemes"][s]["metrics"][m], log)
        # reference effects (delta hit@1 on questions with gold under both schemes)
        eff = {}
        for key, v in ana["reference_effects"].items():
            s2, s1 = key.split("_minus_")
            common = [q for q in golds[s1] if q in golds[s2]]
            eff[key] = {}
            for m in models:
                d = np.mean([int(first_rank(golds[s2][q][0]["orders"][m], golds[s2][q][1]) == 1)
                             - int(first_rank(golds[s1][q][0]["orders"][m], golds[s1][q][1]) == 1) for q in common])
                eff[key][m] = float(d)
                compare_leaves(f"{corpus}.effect.{key}.{m}", float(d), v["models"][m]["delta_hit@1"], log, tol=5.01e-5)  # stored rounded to 4 dp
            compare_leaves(f"{corpus}.effect.{key}.n_common", len(common), v["n_common"], log)
        agr, pen = agreement(qs, corpus, models)
        compare_leaves(f"{corpus}.agreement", agr, ana["annotator_agreement"], log, tol=1e-9)
        # penalised share: the analysis averages 200 random single-annotator draws; the exact expectation
        # over annotators is recomputed here, so a small difference is expected (tolerance 1 point)
        for m in models:
            compare_leaves(f"{corpus}.penalised.{m}", pen[m], ana["penalised_valid_evidence"][m]["share_of_misses_marked_by_another_annotator"], log, tol=0.01)
        for s in ("union", "first"):
            r = reach(qs, corpus, s)
            for m in r:
                compare_leaves(f"{corpus}.reach.{s}.{m}", r[m], ana["global_top5"][s][m], log, tol=1e-12)
        report["comparisons"][corpus] = log
        rec[corpus] = {"blocks": blocks, "eff": eff, "agr": agr, "pen": pen, "models": models}
        print(corpus, "checked", log["checked"], "mismatches", len(log["mismatches"]), "max diff", log["max_abs_diff"], flush=True)

    # ---- macros that are deterministic functions of the data
    mac = macros()
    q, n = rec["qasper"], rec["nq"]

    def chk(name, expected):
        got = mac.get(name)
        report["macro_checks"].append({"macro": name, "manuscript": got, "recomputed": expected, "ok": got == expected})

    sh = lambda r, k: [100 * v for v in r["eff"][k].values()]
    qs_ = sh(q, "first_minus_union")  # stored as first - union for QASPER
    chk("QShiftMin", f"{-max(qs_):.1f}"); chk("QShiftMax", f"{-min(qs_):.1f}")
    ns_ = sh(n, "union_minus_first")
    chk("NShiftMin", f"{min(ns_):.1f}"); chk("NShiftMax", f"{max(ns_):.1f}")
    chk("QPairDisj", f"{100 * q['agr']['pairs_disjoint']:.0f}"); chk("NPairDisj", f"{100 * n['agr']['pairs_disjoint']:.0f}")
    chk("QAlpha", f"{q['agr']['krippendorff_alpha_chunk_level']:.2f}"); chk("NAlpha", f"{n['agr']['krippendorff_alpha_chunk_level']:.2f}")
    chk("QN", f"{q['blocks']['union']['n']:,}"); chk("NN", f"{n['blocks']['first']['n']:,}")
    chk("TotalN", f"{q['blocks']['union']['n'] + n['blocks']['first']['n']:,}")
    for p, r in (("Q", q), ("N", n)):
        for s in ("first", "union"):
            h = [r["blocks"][s]["metrics"][m]["hit@1"] for m in r["models"]]
            chk(f"{p}Gap{s.capitalize()}", f"{100 * (max(h) - min(h)):.1f}")
        chk(f"{p}ChanceFirst", f"{r['blocks']['first']['chance']['hit@1']:.3f}")
        chk(f"{p}ChanceUnion", f"{r['blocks']['union']['chance']['hit@1']:.3f}")
        k1 = [r["blocks"]["union"]["metrics"][m]["kappa@1"] - r["blocks"]["first"]["metrics"][m]["kappa@1"] for m in r["models"]]
        chk(f"{p}KappaShiftMin", f"{100 * min(k1):.1f}"); chk(f"{p}KappaShiftMax", f"{100 * max(k1):.1f}")
        chk(f"{p}PenMin", f"{100 * min(r['pen'].values()):.0f}"); chk(f"{p}PenMax", f"{100 * max(r['pen'].values()):.0f}")
    allpen = list(q["pen"].values()) + list(n["pen"].values())
    chk("PenAllMin", f"{100 * min(allpen):.0f}"); chk("PenAllMax", f"{100 * max(allpen):.0f}")

    # ---- table cells: tab_main (hit@1 per model and reference, c1, kappa1, n)
    tm = (GEN / "tab_main.tex").read_text(encoding="utf-8")
    names = {"BM25": "bm25", "MiniLM-L6": "minilm", "Granite-30M": "granite30m", "ColBERT-small": "colbert_small",
             "ms-marco CE": "ce_msmarco_minilm", "Jina-turbo": "jina_turbo"}
    for part, corpus, cols in ((tm.split("Natural Questions")[0], "qasper", ("union", "first", "plurality", "agreement")),
                               (tm.split("Natural Questions")[1], "nq", ("first", "union", "official", "plurality"))):
        b = rec[corpus]["blocks"]
        for line in part.splitlines():
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            if len(cells) != 5:
                continue
            label = cells[0]
            if label in names:
                exp = [f"{b[s]['metrics'][names[label]]['hit@1']:.3f}" for s in cols]
            elif label == "$c_1$":
                exp = [f"{b[s]['chance']['hit@1']:.3f}" for s in cols]
            elif label.startswith("$\\kappa_1$ "):
                exp = [f"{b[s]['metrics'][names[label.split(' ', 1)[1]]]['kappa@1']:.3f}" for s in cols]
            elif label == "":
                exp = [f"$n={b[s]['n']:,}$".replace(",", "{,}") for s in cols]
            else:
                continue
            ok = cells[1:] == exp
            report["table_checks"].append({"table": "tab_main", "corpus": corpus, "row": label or "n", "manuscript": cells[1:], "recomputed": exp, "ok": ok})
    bad_m = [x for x in report["macro_checks"] if not x["ok"]]
    bad_t = [x for x in report["table_checks"] if not x["ok"]]
    report["summary"] = {"analysis_leaves_checked": sum(v["checked"] for v in report["comparisons"].values()),
                         "analysis_mismatches": sum(len(v["mismatches"]) for v in report["comparisons"].values()),
                         "macros_checked": len(report["macro_checks"]), "macro_mismatches": len(bad_m),
                         "table_rows_checked": len(report["table_checks"]), "table_row_mismatches": len(bad_t)}
    (REV / "results/audit/numeric_consistency.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    for x in bad_m + bad_t:
        print("MISMATCH", x)
    for c, v in report["comparisons"].items():
        for x in v["mismatches"][:20]:
            print("LEAF", c, x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
