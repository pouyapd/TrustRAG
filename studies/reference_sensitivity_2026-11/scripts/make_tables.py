#!/usr/bin/env python
"""Generate every number, table and figure of the revised manuscript from result files.

Inputs (all produced by scripts in this revision or by the repository):
  results/qasper_full/analysis.json, results/nq_full/analysis.json   (analyse_v2.py)
  results/raw/qasper_full.json.gz, results/raw/nq_full.json.gz       (score_v2.py; counts)
  <repo>/results/exploratory/*_reference_sensitivity/summary.json    (seven-model subsets)
  <repo>/results/exploratory/reference_sensitivity_claims.json       (claim verdicts)
  <repo>/results/goldspan_adjudication/remaining37/estimate_combined.json
Outputs: manuscript/generated/numbers.tex, manuscript/generated/tab_*.tex, figures/*.pdf
No number in the manuscript's results is typed by hand; the text uses the macros.
"""
from __future__ import annotations

import gzip
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[1]
GEN = REV / "manuscript" / "generated"
FIG = REV / "figures"
NAMES = {"colbert_small": "ColBERT-small", "bm25": "BM25", "minilm": "MiniLM-L6", "bge": "BGE-small", "e5": "E5-small", "mpnet": "MPNet-base",
         "ce_msmarco_minilm": "ms-marco CE", "bge_reranker_base": "bge-reranker-base",
         "granite30m": "Granite-30M", "jina_turbo": "Jina-turbo", "bge_m3": "BGE-M3"}
ORDER = ["bm25", "minilm", "bge", "e5", "mpnet", "granite30m", "bge_m3", "colbert_small", "ce_msmarco_minilm", "jina_turbo",
         "bge_reranker_base"]
SEVEN = ["bm25", "minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base"]
SCHEME_NAMES = {"first": "First", "union": "Union", "official": "Official", "plurality": "Plurality",
                "agreement": "Agreement", "agreement_subset": "Agreement subset"}
macros: dict[str, str] = {}


def mac(name: str, value) -> None:
    if not name.isalpha():
        raise ValueError(name)
    macros[name] = str(value)


def f3(x: float) -> str:
    return f"{x:.3f}"


def pts(x: float) -> str:
    """Percentage points with sign, one decimal."""
    return f"{100 * x:+.1f}".replace("-", "$-$")


def fmt_p(p: float) -> str:
    """p-value for running text, in math mode."""
    if p >= 0.001:
        return f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def num(n: int) -> str:
    return f"{n:,}"


def load(corpus: str) -> tuple[dict, dict]:
    a = json.loads((REV / f"results/{corpus}_full/analysis.json").read_text(encoding="utf-8"))
    raw = json.load(gzip.open(REV / f"results/raw/{corpus}_full.json.gz", "rt", encoding="utf-8"))
    return a, raw


def corpus_macros(p: str, a: dict, raw: dict) -> None:
    qs = raw["questions"]
    mac(f"{p}N", num(len(qs)))
    mac(f"{p}Docs", num(raw["config"]["n_documents"]))
    mac(f"{p}Chunks", num(raw["config"]["n_chunks"]))
    mac(f"{p}meanD", f"{np.mean([q['D'] for q in qs]):.1f}")
    per_doc = Counter(q["doc_id"] for q in qs)
    mac(f"{p}maxPerDoc", max(per_doc.values()))
    sys.path.insert(0, str(REV / "scripts"))
    from analyse_v2 import cluster_key
    corpus = "qasper" if p == "Q" else "nq"
    clusters = Counter(cluster_key(corpus, q["doc_id"]) for q in qs)
    mac(f"{p}NDocIds", num(len(per_doc)))
    mac(f"{p}NClusters", num(len(clusters)))
    multi_rev = Counter(cluster_key(corpus, d) for d in per_doc)
    mac(f"{p}MultiRevision", sum(1 for v in multi_rev.values() if v > 1))
    # where the secondary (question-level McNemar) test decides differently from the primary test
    diff = [(s, x) for s in a["schemes"] for x in a["schemes"][s]["pairwise"] if x["significant"] != x["significant_mcnemar"]]
    mac(f"{p}McDiffer", len(diff))
    mac(f"{p}McDifferList", "; ".join(
        f"{' vs '.join(NAMES[m] for m in sorted((x['a'], x['b']), key=ORDER.index))} under the {SCHEME_NAMES[s].lower()} reference "
        f"(primary {'significant' if x['significant'] else 'not significant'}, Holm-adjusted $p$ = {x['p_holm']:.3f}; McNemar "
        f"{'significant' if x['significant_mcnemar'] else 'not significant'}, $p$ = {x['p_mcnemar_holm']:.3f})" for s, x in diff) or "none")
    mac(f"{p}NComparisons", sum(len(a["schemes"][s]["pairwise"]) for s in a["schemes"]))
    n_ann = Counter(len([v for v in q["gold_by_annotator"].values() if v]) for q in qs)
    multi = sum(v for k, v in n_ann.items() if k >= 2)
    mac(f"{p}Multi", num(multi))
    mac(f"{p}MultiShare", f"{100 * multi / len(qs):.0f}")
    if p == "Q":
        mac("QNdev", num(sum(q["split"] == "dev" for q in qs)))
        mac("QNtest", num(sum(q["split"] == "test" for q in qs)))
    ag = a["annotator_agreement"]
    mac(f"{p}AgrQ", num(ag["questions_with_2plus_annotators"]))
    mac(f"{p}Alpha", f"{ag['krippendorff_alpha_chunk_level']:.2f}")
    mac(f"{p}Jacc", f"{ag['mean_jaccard']:.2f}")
    mac(f"{p}PairIdent", f"{100 * ag['pairs_identical']:.0f}")
    mac(f"{p}PairDisj", f"{100 * ag['pairs_disjoint']:.0f}")
    mac(f"{p}QDisj", f"{100 * ag['questions_with_a_disjoint_pair']:.0f}")
    pen = a["penalised_valid_evidence"]
    shares = [v["share_of_misses_marked_by_another_annotator"] for v in pen.values()]
    mac(f"{p}PenMin", f"{100 * min(shares):.0f}")
    mac(f"{p}PenMax", f"{100 * max(shares):.0f}")
    eff_key = "first_minus_union" if p == "Q" else "union_minus_first"
    eff = a["reference_effects"][eff_key]["models"]
    sign = -1 if p == "Q" else 1  # report union minus first in both corpora
    d = [sign * v["delta_hit@1"] for v in eff.values()]
    mac(f"{p}ShiftMin", f"{100 * min(d):.1f}")
    mac(f"{p}ShiftMax", f"{100 * max(d):.1f}")
    mac(f"{p}DiffGold", num(a["reference_effects"][eff_key]["questions_with_different_gold"]))
    mac(f"{p}DiffGoldShare", f"{100 * a['reference_effects'][eff_key]['questions_with_different_gold'] / a['reference_effects'][eff_key]['n_common']:.0f}")
    sch = a["schemes"]
    hits = [sch[s]["metrics"][m]["hit@1"] for s in sch for m in sch[s]["metrics"]]
    mac(f"{p}HitMin", f3(min(hits)))
    mac(f"{p}HitMax", f3(max(hits)))
    sa = a["single_annotator"]["hit@1"]
    mac(f"{p}DrawRangeMax", f"{100 * max(v['draw_range'] for v in sa.values()):.1f}")
    mac(f"{p}DrawRangeMin", f"{100 * min(v['draw_range'] for v in sa.values()):.1f}")
    w = [sch["union"]["metrics"][m]["hit@1_cluster95"] for m in sch["union"]["metrics"]]
    mac(f"{p}SampleCIWidth", f"{100 * np.mean([hi - lo for lo, hi in w]):.1f}")
    models = a["models"]
    for s in sch:
        mac(f"{p}Sig{s.capitalize()}", len(sch[s]["significant_pairs"]))
    mac(f"{p}SigSame", f"{100 * a['single_annotator']['holm_conclusion_same_as_study_share_of_200']:.0f}")
    mac(f"{p}SigDistinct", a["single_annotator"]["distinct_holm_conclusions_in_200"])
    # which model pairs swap order between references, and are any of them significant?
    swaps, swap_gap, swap_sig = set(), 0.0, False
    for s1, s2 in itertools.combinations(list(sch), 2):
        for m1, m2 in itertools.combinations(models, 2):
            d1 = sch[s1]["metrics"][m1]["hit@1"] - sch[s1]["metrics"][m2]["hit@1"]
            d2 = sch[s2]["metrics"][m1]["hit@1"] - sch[s2]["metrics"][m2]["hit@1"]
            if d1 * d2 < 0:
                swaps.add((m1, m2))
                swap_gap = max(swap_gap, abs(d1), abs(d2))
                for s in (s1, s2):
                    pr = next(x for x in sch[s]["pairwise"] if {x["a"], x["b"]} == {m1, m2})
                    swap_sig |= pr["significant"]
    # comparisons whose significance differs between the first-annotator and union references
    sf = {frozenset((x["a"], x["b"])): x["significant"] for x in sch["first"]["pairwise"]}
    su = {frozenset((x["a"], x["b"])): x["significant"] for x in sch["union"]["pairwise"]}
    changed = [k for k in sf if sf[k] != su[k]]
    mac(f"{p}NPairs", len(sf))
    mac(f"{p}SigChanged", len(changed))
    mac(f"{p}SigChangedPairs", "; ".join(" vs ".join(NAMES[m] for m in sorted(k, key=ORDER.index)) for k in changed) or "none")
    mac(f"{p}SigOnlyUnion", sum(1 for k in changed if su[k]))
    mac(f"{p}SwapCount", len(swaps))
    mac(f"{p}SwapMaxGap", f"{100 * swap_gap:.1f}")
    mac(f"{p}SwapSig", "yes" if swap_sig else "no")
    mac(f"{p}SwapPairs", "; ".join(f"{NAMES[a]}/{NAMES[b]}" for a, b in sorted(swaps)) or "none")
    taus = list(a["ordering_kendall_tau"].values())
    mac(f"{p}TauMin", f"{min(taus):.2f}")
    mac(f"{p}NModels", len(models))
    ch = {s: sch[s]["chance"]["hit@1"] for s in sch}
    mac(f"{p}ChanceFirst", f3(ch["first"]))
    mac(f"{p}ChanceUnion", f3(ch["union"]))
    for s in ("first", "union"):
        h = [sch[s]["metrics"][m]["hit@1"] for m in models]
        mac(f"{p}Gap{s.capitalize()}", f"{100 * (max(h) - min(h)):.1f}")
    mac(f"{p}NFirst", num(sch["first"]["n"]))
    mac(f"{p}NUnion", num(sch["union"]["n"]))
    mac(f"{p}NPlurality", num(sch["plurality"]["n"]))
    if "official" in sch:
        mac(f"{p}NOfficial", num(sch["official"]["n"]))
    if "agreement" in sch:
        mac(f"{p}NAgreement", num(sch["agreement"]["n"]))
    from scipy.stats import spearmanr
    rho = spearmanr([sch["union"]["metrics"][m]["hit@1"] for m in models],
                    [pen[m]["share_of_misses_marked_by_another_annotator"] for m in models]).statistic
    mac(f"{p}PenRho", f"{rho:.2f}")
    pen_lo = min(models, key=lambda m: pen[m]["share_of_misses_marked_by_another_annotator"])
    pen_hi = max(models, key=lambda m: pen[m]["share_of_misses_marked_by_another_annotator"])
    mac(f"{p}PenLoModel", NAMES[pen_lo])
    mac(f"{p}PenHiModel", NAMES[pen_hi])
    g = a["global_top5"]
    dc = [g["union"][m]["C"] - g["first"][m]["C"] for m in g["union"]]
    mac(f"{p}CShiftMin", f"{100 * min(dc):.1f}")
    mac(f"{p}CShiftMax", f"{100 * max(dc):.1f}")
    sub = a["subsample_first_vs_union"]
    for n in ("100", "300", "1000"):
        if n in sub:
            mac(f"{p}Sub{ {'100': 'Hundred', '300': 'ThreeHundred', '1000': 'Thousand'}[n] }Differ",
                f"{100 * sub[n]['share_conclusions_differ']:.0f}")
    mac(f"{p}FullDiffer", "differ" if sub["full"]["conclusions_differ"] else "agree")
    for s in ("first", "union"):
        pr = next(x for x in sch[s]["pairwise"] if {x["a"], x["b"]} == {"bm25", "minilm"})
        # pairwise stores a - b; report MiniLM minus BM25
        flip = pr["a"] == "bm25"
        d = -pr["delta_hit@1"] if flip else pr["delta_hit@1"]
        lo, hi = sorted(-x for x in pr["cluster95"]) if flip else pr["cluster95"]
        mac(f"{p}Pair{s.capitalize()}D", f"{100 * d:.1f}")
        mac(f"{p}Pair{s.capitalize()}Lo", f"{100 * lo:.1f}")
        mac(f"{p}Pair{s.capitalize()}Hi", f"{100 * hi:.1f}")
        mac(f"{p}Pair{s.capitalize()}P", fmt_p(pr["p_holm"]))
        mac(f"{p}Pair{s.capitalize()}PMc", fmt_p(pr["p_mcnemar_holm"]))
    k1 = [sch["union"]["metrics"][m]["kappa@1"] - sch["first"]["metrics"][m]["kappa@1"] for m in models]
    mac(f"{p}KappaShiftMin", f"{100 * min(k1):.1f}")
    mac(f"{p}KappaShiftMax", f"{100 * max(k1):.1f}")


def table_main(qa: dict, na: dict) -> str:
    """hit@1 with cluster-bootstrap CI per model and reference, both corpora."""
    out = ["\\begin{table}[t]",
           "\\caption{Within-document hit@1 under each evidence reference, full samples. Intervals are "
           "shown in Figure~\\ref{fig:reference}; the row ``CI $\\pm$'' gives the mean half-width of the "
           "cluster bootstrap 95\\% intervals (QASPER: paper; NQ: Wikipedia page), in points. $c_1$: chance hit@1; $\\kappa_1$: "
           "chance-corrected hit@1, $(\\text{hit@1}-c_1)/(1-c_1)$. $n$: questions with gold chunks "
           "under the reference. The earlier study's references are QASPER union and NQ first.}",
           "\\label{tab:main}", "\\centering\\small\\setlength{\\tabcolsep}{5pt}"]
    for label, a in (("QASPER", qa), ("Natural Questions", na)):
        schemes = list(a["schemes"])
        out.append("\\begin{tabular}{l" + "c" * len(schemes) + "}")
        out.append("\\hline")
        out.append(f"\\multicolumn{{{len(schemes) + 1}}}{{l}}{{\\textit{{{label}}}}}\\\\")
        out.append("Model & " + " & ".join(SCHEME_NAMES[s] for s in schemes) + "\\\\")
        out.append(" & " + " & ".join("$n=" + f"{a['schemes'][s]['n']:,}".replace(",", "{,}") + "$" for s in schemes) + "\\\\")
        out.append("\\hline")
        for m in [x for x in ORDER if x in a["models"]]:
            cells = [f3(a["schemes"][s]["metrics"][m]["hit@1"]) for s in schemes]
            out.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
        half = []
        for s in schemes:
            w = [(a["schemes"][s]["metrics"][m]["hit@1_cluster95"][1] - a["schemes"][s]["metrics"][m]["hit@1_cluster95"][0]) / 2
                 for m in a["models"]]
            half.append(f"{100 * np.mean(w):.1f}")
        out.append("CI $\\pm$ & " + " & ".join(half) + "\\\\")
        out.append("\\hline")
        out.append("$c_1$ & " + " & ".join(f3(a["schemes"][s]["chance"]["hit@1"]) for s in schemes) + "\\\\")
        for m in [x for x in ORDER if x in a["models"]]:
            out.append(f"$\\kappa_1$ {NAMES[m]} & " + " & ".join(
                f3(a["schemes"][s]["metrics"][m]["kappa@1"]) for s in schemes) + "\\\\")
        out.append("\\hline\\end{tabular}")
        out.append("\\par\\medskip")
    out.append("\\end{table}")
    return "\n".join(out)


def table_models(full: list[str]) -> str:
    rows = [("bm25", "BM25 (Okapi)", "lexical", "--", "classic"), ("minilm", "all-MiniLM-L6-v2", "bi-encoder", "22M", "2021"),
            ("bge", "bge-small-en-v1.5", "bi-encoder", "33M", "2023"), ("e5", "e5-small-v2", "bi-encoder", "33M", "2023"),
            ("mpnet", "all-mpnet-base-v2", "bi-encoder", "110M", "2021"),
            ("granite30m", "granite-embedding-30m-english", "bi-encoder", "30M", "2024"),
            ("bge_m3", "bge-m3 (dense)", "bi-encoder", "568M", "2024"),
            ("colbert_small", "answerai-colbert-small-v1", "late interaction", "33M", "2024"),
            ("ce_msmarco_minilm", "ms-marco-MiniLM-L-6-v2", "cross-encoder", "22M", "2021"),
            ("jina_turbo", "jina-reranker-v1-turbo-en", "cross-encoder", "38M", "2024"),
            ("bge_reranker_base", "bge-reranker-base", "cross-encoder", "278M", "2023")]
    rev = {"minilm": "1110a243fd", "bge": "5c38ec7c40", "e5": "ffb93f3bd4", "mpnet": "e8c3b32edf",
           "ce_msmarco_minilm": "233902d25c", "bge_reranker_base": "2cfc18c941"}
    seven = {"bm25", "minilm", "bge", "e5", "mpnet", "ce_msmarco_minilm", "bge_reranker_base"}
    for f in ("qasper_full", "qasper_sub290"):
        p = REV / f"results/raw/{f}.json.gz"
        if p.exists():
            cfg = json.load(gzip.open(p, "rt", encoding="utf-8"))["config"]["models"]
            for k, v in cfg.items():
                if k not in rev and v.get("cached_revision"):
                    rev[k] = v["cached_revision"][:10]
    sub = set()
    sp = REV / "results/qasper_sub290/analysis.json"
    if sp.exists():
        sub = set(json.loads(sp.read_text(encoding="utf-8"))["models"]) - set(full)
    out = ["\\begin{table}[t]",
           "\\caption{Localisers, with the year of the model's release. ``Full'': run on the full samples (\\QN{} QASPER "
           "and \\NN{} NQ questions; ColBERT-small on the full QASPER sample only). ``Subsets'': run on the earlier "
           "study's 290 QASPER and 300 NQ questions (BGE-M3 on the 290 QASPER questions only). Revisions are the Hugging Face commits "
           "the local cache resolved.}", "\\label{tab:models}", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{llrclcc}", "\\hline", "Model & Family & Params & Year & Revision & Full & Subsets\\\\", "\\hline"]
    for key, name, fam, par, yr in rows:
        if key not in full and key not in seven and key not in sub:
            continue
        fullmark = "\\checkmark" if key in full else ("QASPER" if key == "colbert_small" else "")
        submark = "\\checkmark" if key in seven or key in sub or key in ("granite30m", "jina_turbo", "colbert_small") else ""
        out.append(f"{name} & {fam} & {par} & {yr} & {('\\texttt{' + rev[key] + '}') if key in rev else '--'} & "
                   f"{fullmark} & {submark}\\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def table_effects(qa: dict, na: dict) -> str:
    def iv(v, lo_hi, sign, scale=100, fmt="+.1f"):
        lo, hi = sorted(sign * x for x in lo_hi)
        return f"{scale * sign * v:{fmt}} [{scale * lo:{fmt}}, {scale * hi:{fmt}}]".replace("-", "$-$")
    out = ["\\begin{table}[t]",
           "\\caption{How large is the reference effect? Union minus first-annotator scores on the questions "
           "both references define (paired; cluster bootstrap 95\\% intervals, QASPER paper and NQ Wikipedia page): hit@1 and hit@5 in "
           "percentage points, MRR in hundredths. Last two columns: central 95\\% range of hit@1 over 1,000 "
           "single-annotator draws, and the width of the hit@1 cluster interval under the union reference, "
           "both in points.}",
           "\\label{tab:effects}", "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{llccccc}", "\\hline",
           "Corpus & Model & $\\Delta$hit@1 & $\\Delta$hit@5 & $\\Delta$MRR & Draw range & CI width\\\\", "\\hline"]
    # Point estimate of the hit@1 effect: the unrounded value from cluster_inference.json (same paired questions).
    # analysis.json stores it rounded to four decimals, and rounding that value again to one decimal of a point can
    # differ from the exact value (QASPER ms-marco CE: stored 0.0545, exact 0.05448 -> +5.4, not +5.5). 28 Nov 2026.
    ci_p = REV / "results/audit/cluster_inference.json"
    ci = json.loads(ci_p.read_text(encoding="utf-8")) if ci_p.exists() else None
    for label, a, key, sign, corpus in (("QASPER", qa, "first_minus_union", -1, "qasper"), ("NQ", na, "union_minus_first", 1, "nq")):
        eff = a["reference_effects"][key]["models"]
        for m in [x for x in ORDER if x in a["models"]]:
            e = eff[m]
            d1 = e["delta_hit@1"]
            if ci:
                exact = ci[corpus]["reference_effects"]["union_minus_first"][m]["delta"]
                assert abs(sign * exact - d1) < 5.1e-5, (corpus, m, exact, d1)
                d1 = sign * exact
            sa = a["single_annotator"]["hit@1"][m]
            lo2, hi2 = a["schemes"]["union"]["metrics"][m]["hit@1_cluster95"]
            out.append(f"{label} & {NAMES[m]} & {iv(d1, e['cluster95'], sign)} & "
                       f"{iv(e['delta_hit@5'], e['cluster95_hit@5'], sign)} & "
                       f"{iv(e['delta_mrr'], e['cluster95_mrr'], sign)} & "
                       f"{100 * sa['draw_range']:.1f} & {100 * (hi2 - lo2):.1f}\\\\")
            label = ""
        out.append("\\hline")
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def table_conclusions(qa: dict, na: dict) -> str:
    short = {"bm25": "BM25", "minilm": "MiniLM", "ce_msmarco_minilm": "CE", "bge": "BGE",
             "granite30m": "Granite", "jina_turbo": "Jina", "colbert_small": "ColBERT", "bge_m3": "BGE-M3"}
    out = ["\\begin{table}[t]",
           "\\caption{Model comparisons on hit@1 that are significant after Holm correction over all pairs, under each "
           "reference, full samples. Primary test: paired permutation test at cluster level (QASPER paper, NQ Wikipedia "
           "page; 20,000 sign vectors). $A>B$: $A$ ranks a gold chunk first significantly more often. $^\\dagger$: the "
           "question-level exact McNemar test with Holm correction (secondary analysis) does not reach significance for "
           "this pair; it agrees with the primary test on every other pair. CE: ms-marco cross-encoder; Jina: Jina "
           "reranker v1 turbo; ColBERT: answerai-colbert-small-v1. Source: this study, computed from the released "
           "per-question orderings.}", "\\label{tab:conclusions}", "\\centering\\small",
           "\\begin{tabular}{llp{0.62\\textwidth}}", "\\hline",
           "Corpus & Reference & Significant pairs\\\\", "\\hline"]
    for label, a in (("QASPER", qa), ("NQ", na)):
        for s, v in a["schemes"].items():
            cells = []
            for x in v["pairwise"]:
                if not x["significant"]:
                    continue
                loser = x["b"] if x["winner"] == x["a"] else x["a"]
                cells.append(short[x["winner"]] + " $>$ " + short[loser] + ("$^\\dagger$" if not x["significant_mcnemar"] else ""))
            out.append(f"{label} & {SCHEME_NAMES[s]} & {', '.join(cells) or 'none'}\\\\")
            label = ""
        out.append("\\hline")
    out += ["\\end{tabular}", "\\end{table}"]
    # pairs significant under McNemar only (none expected; listed for completeness)
    mc_only = [(lab, s) for lab, a in (("QASPER", qa), ("NQ", na)) for s, v in a["schemes"].items()
               for x in v["pairwise"] if x["significant_mcnemar"] and not x["significant"]]
    mac("McOnlyCount", len(mc_only))
    return "\n".join(out)


def table_incomplete(qa: dict, na: dict, adj: dict) -> str:
    out = ["\\begin{table}[t]",
           "\\caption{Incompleteness of single-annotator references. Top: agreement between the "
           "datasets' own annotators on gold chunks (questions with $\\geq$2 usable annotators). "
           "Bottom: share of a model's rank-1 misses under one randomly drawn annotator's reference "
           "that another annotator marked as gold (mean over 200 draws).}",
           "\\label{tab:incomplete}", "\\centering\\small", "\\begin{tabular}{lcc}", "\\hline",
           " & QASPER & NQ\\\\", "\\hline"]
    qg, ng = qa["annotator_agreement"], na["annotator_agreement"]
    rows = [("Questions with $\\geq$2 annotators", num(qg["questions_with_2plus_annotators"]), num(ng["questions_with_2plus_annotators"])),
            ("Annotator pairs", num(qg["annotator_pairs"]), num(ng["annotator_pairs"])),
            ("Pairs with identical gold chunks", f"{100 * qg['pairs_identical']:.1f}\\%", f"{100 * ng['pairs_identical']:.1f}\\%"),
            ("Pairs overlapping, not identical", f"{100 * qg['pairs_overlapping_not_identical']:.1f}\\%", f"{100 * ng['pairs_overlapping_not_identical']:.1f}\\%"),
            ("Pairs with disjoint gold chunks", f"{100 * qg['pairs_disjoint']:.1f}\\%", f"{100 * ng['pairs_disjoint']:.1f}\\%"),
            ("Questions with a disjoint pair", f"{100 * qg['questions_with_a_disjoint_pair']:.1f}\\%", f"{100 * ng['questions_with_a_disjoint_pair']:.1f}\\%"),
            ("Mean Jaccard of gold chunk sets", f"{qg['mean_jaccard']:.3f}", f"{ng['mean_jaccard']:.3f}"),
            ("Krippendorff's $\\alpha$ (chunk level)", f"{qg['krippendorff_alpha_chunk_level']:.3f}", f"{ng['krippendorff_alpha_chunk_level']:.3f}")]
    out += [" & ".join(r) + "\\\\" for r in rows]
    out.append("\\hline")
    out.append("\\multicolumn{3}{l}{\\textit{Rank-1 misses marked as gold by another annotator}}\\\\")
    def cell(a, m):
        v = a["penalised_valid_evidence"].get(m)
        return "--" if v is None else f"{100 * v['share_of_misses_marked_by_another_annotator']:.1f}\\%"
    for m in [x for x in ORDER if x in qa["models"] or x in na["models"]]:
        out.append(f"{NAMES[m]} & {cell(qa, m)} & {cell(na, m)}\\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def table_reach(qa: dict, na: dict) -> str:
    out = ["\\begin{table}[t]",
           "\\caption{Global top-5 retrieval over each corpus: reach A (a top-5 chunk from the gold document), "
           "span coverage C (a top-5 chunk is gold), C/A and chunks admitted from the gold document when it "
           "is reached, under the first-annotator and union references. A and admitted chunks do not depend "
           "on the reference by construction.}",
           "\\label{tab:reach}", "\\centering\\small", "\\begin{tabular}{llccccc}", "\\hline",
           "Corpus & Model & A & C first & C union & C/A first & C/A union\\\\", "\\hline"]
    for label, a in (("QASPER", qa), ("NQ", na)):
        g = a["global_top5"]
        for m in [x for x in ORDER if x in g["union"]]:
            f, u = g["first"][m], g["union"][m]
            out.append(f"{label} & {NAMES[m]} & {u['A']:.3f} & {f['C']:.3f} & {u['C']:.3f} & "
                       f"{f['C_over_A']:.3f} & {u['C_over_A']:.3f}\\\\")
            label = ""
        out.append("\\hline")
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def seven_model() -> tuple[str, dict]:
    """Seven-model subsets: hit@1 under first and union references, both corpora."""
    ex = REPO / "results/exploratory"
    q = json.loads((ex / "qasper_reference_sensitivity/summary.json").read_text(encoding="utf-8"))
    n = json.loads((ex / "nq_reference_sensitivity/summary.json").read_text(encoding="utf-8"))
    out = ["\\begin{table}[t]",
           "\\caption{Seven localisers on the earlier study's subsets (QASPER 290, NQ 300 questions): hit@1 "
           "under the first-annotator and union references, and the number of pairs (of 21) significant after Holm "
           "correction under the primary cluster permutation test (in parentheses: under the question-level McNemar test). "
           "The earlier study reported the combinations marked $^\\ast$ (QASPER union, NQ first).}",
           "\\label{tab:seven}", "\\centering\\small", "\\begin{tabular}{lcccc}", "\\hline",
           " & \\multicolumn{2}{c}{QASPER} & \\multicolumn{2}{c}{NQ}\\\\",
           "Model & First & Union$^\\ast$ & First$^\\ast$ & Union\\\\", "\\hline"]
    for m in SEVEN:
        out.append(f"{NAMES[m]} & {q['schemes']['first']['models'][m]['hit@1']:.3f} & "
                   f"{q['schemes']['union']['models'][m]['hit@1']:.3f} & "
                   f"{n['schemes']['first']['models'][m]['hit@1']:.3f} & "
                   f"{n['schemes']['union']['models'][m]['hit@1']:.3f}\\\\")
    # significance: primary cluster permutation test (seven_inference.py); McNemar in parentheses
    si = json.loads((REV / "results/audit/seven_inference.json").read_text(encoding="utf-8"))
    sig_p = {("q", s): si["qasper"][s]["n_sig_primary"] for s in ("first", "union")}
    sig_p.update({("n", s): si["nq"][s]["n_sig_primary"] for s in ("first", "union")})
    sig_m = {("q", s): si["qasper"][s]["n_sig_mcnemar"] for s in ("first", "union")}
    sig_m.update({("n", s): si["nq"][s]["n_sig_mcnemar"] for s in ("first", "union")})
    cell = lambda c, s: f"{sig_p[(c, s)]} ({sig_m[(c, s)]})"
    out.append("\\hline")
    out.append(f"Significant pairs & {cell('q', 'first')} & {cell('q', 'union')} & "
               f"{cell('n', 'first')} & {cell('n', 'union')}\\\\")
    out.append(f"$n$ & {q['schemes']['first']['n_questions']} & {q['schemes']['union']['n_questions']} & "
               f"{n['schemes']['first']['n_questions']} & {n['schemes']['union']['n_questions']}\\\\")
    out += ["\\hline", "\\multicolumn{5}{l}{\\footnotesize $^\\ast$ the earlier study's reference for that corpus}\\\\",
            "\\end{tabular}", "\\end{table}"]
    mac("SevenQSigFirst", sig_p[("q", "first")])
    mac("SevenQSigUnion", sig_p[("q", "union")])
    mac("SevenNSigFirst", sig_p[("n", "first")])
    mac("SevenNSigUnion", sig_p[("n", "union")])
    mac("SevenNSigFirstMc", sig_m[("n", "first")])
    best_u = max(SEVEN, key=lambda m: n["schemes"]["union"]["models"][m]["hit@1"])
    mac("SevenNBestUnion", f3(n["schemes"]["union"]["models"][best_u]["hit@1"]))
    return "\n".join(out), {"q": q, "n": n}


def claims_table() -> str:
    c = json.loads((REPO / "results/exploratory/reference_sensitivity_claims.json").read_text(encoding="utf-8"))
    return json.dumps(c)[:0]  # verdicts are summarised in the text from the JSON; see results/claims


def figures(qa: dict, na: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    FIG.mkdir(exist_ok=True)
    markers = {"first": "o", "union": "s", "official": "D", "plurality": "^", "agreement": "v"}
    COLOURS = {"first": "C1", "union": "C0", "official": "C2", "plurality": "C3", "agreement": "C4"}
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 3.0), sharey=False)
    for ax, (label, a) in zip(axes, (("QASPER", qa), ("Natural Questions", na)), strict=True):
        models = [x for x in ORDER if x in a["models"]]
        schemes = list(a["schemes"])
        for j, s in enumerate(schemes):
            off = (j - (len(schemes) - 1) / 2) * 0.14
            for i, m in enumerate(models):
                mm = a["schemes"][s]["metrics"][m]
                lo, hi = mm["hit@1_cluster95"]
                ax.errorbar(i + off, mm["hit@1"], yerr=[[mm["hit@1"] - lo], [hi - mm["hit@1"]]], fmt=markers[s],
                            ms=3.5, lw=0.8, color=COLOURS[s], label=SCHEME_NAMES[s] if i == 0 else None)
        sa = a["single_annotator"]["hit@1"]
        for i, m in enumerate(models):
            lo, hi = sa[m]["draw95"]
            ax.add_patch(plt.Rectangle((i - 0.38, lo), 0.76, hi - lo, color="0.85", zorder=0, lw=0))
        short = {"bm25": "BM25", "minilm": "MiniLM", "granite30m": "Granite", "colbert_small": "ColBERT",
                 "ce_msmarco_minilm": "CE", "jina_turbo": "Jina", "bge": "BGE", "bge_m3": "BGE-M3"}
        ax.set_xticks(range(len(models)), [short.get(m, NAMES[m]) for m in models], rotation=40, ha="right",
                      rotation_mode="anchor", fontsize=7)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_title(label)
        ax.set_ylabel("hit@1 inside the gold document")
    # one shared legend above both panels, so that no marker or band is covered
    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(FIG / "fig_reference_hit1.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_reference_hit1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sensitivity_outputs() -> None:
    """Table and macros for the overlap-rule (full samples) and chunk-size (subsets) sensitivity analyses."""
    ov = json.loads((REV / "results/audit/overlap_sensitivity.json").read_text(encoding="utf-8"))
    chp = REV / "results/audit/chunk_sensitivity.json"
    ch = json.loads(chp.read_text(encoding="utf-8")) if chp.exists() else {}
    short = {"bm25": "BM25", "minilm": "MiniLM", "ce_msmarco_minilm": "CE", "granite30m": "Granite",
             "jina_turbo": "Jina", "colbert_small": "ColBERT"}

    def row(r):
        e = [100 * v["delta"] for v in r["union_minus_first"].values()]
        g = [100 * v["delta"] for v in r["neural_minus_bm25"].values()]
        gs = sum(1 for v in r["neural_minus_bm25"].values() if v["p_perm_holm"] < 0.05)
        es = sum(1 for v in r["union_minus_first"].values() if v["p_perm_holm"] < 0.05)
        lab = lambda x: short[x.split('>')[0]] + " $>$ " + short[x.split('>')[1]]
        chg = ", ".join(lab(x) for x in r["significance_changed"]) or "none"
        return (f"{min(e):.1f} to {max(e):.1f} ({es}/{len(e)})", f"{min(g):+.1f} to {max(g):+.1f} ({gs}/{len(g)})".replace("-", "$-$"), chg)

    out = ["\\begin{table}[t]",
           "\\caption{Sensitivity of the reference effects to the overlap rule (full samples, all models) and to the chunk "
           "size (earlier study's subsets; BM25, MiniLM-L6 and the ms-marco cross-encoder, re-run at each size). "
           "Union $-$ first: range over models of the union minus first-annotator hit@1, in points (in parentheses: models "
           "whose effect is significant, cluster permutation test, Holm over models). Neural $-$ BM25: range of the excess "
           "of a neural model's union gain over BM25's (models with a significant excess). Changed pairs: model pairs whose "
           "significance differs between the two references (primary test, Holm over all pairs). The two alternative "
           "overlap rules and the two alternative chunk sizes were fixed before any result was computed. "
           "Source: this study.}",
           "\\label{tab:sensitivity}", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{llrccp{0.22\\textwidth}}", "\\hline",
           "Corpus & Setting & $n$ & Union $-$ first & Neural $-$ BM25 & Changed pairs\\\\", "\\hline"]
    names = {"any": "overlap $\\geq$ 1 char.$^\\ast$", "min50": "overlap $\\geq$ 50 char.", "prop50": "overlap $\\geq$ 50\\%"}
    for label, c in (("QASPER", "qasper"), ("NQ", "nq")):
        first = True
        for rule in ("any", "min50", "prop50"):
            r = ov[c][rule]
            a, b, chg = row(r)
            out.append(f"{label if first else ''} & {names[rule]} & {num(r['n_common'])} & {a} & {b} & {chg}\\\\")
            first = False
        for cs in ("128", "256", "512"):
            if c in ch and cs in ch[c]:
                r = ch[c][cs]
                a, b, chg = row(r)
                out.append(f" & {cs} tokens{'$^\\ast$' if cs == '256' else ''} (subset) & {num(r['n_common'])} & {a} & {b} & {chg}\\\\")
        out.append("\\hline")
    out += ["\\multicolumn{6}{l}{\\footnotesize $^\\ast$ the setting used in the rest of the paper}\\\\",
            "\\end{tabular}", "\\end{table}"]
    (GEN / "tab_sensitivity.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    alle = [100 * v["delta"] for c in ("qasper", "nq") for rule in ("min50", "prop50") for v in ov[c][rule]["union_minus_first"].values()]
    mac("OverlapShiftMin", f"{min(alle):.1f}"); mac("OverlapShiftMax", f"{max(alle):.1f}")
    allsig = all(v["p_perm_holm"] < 0.05 for c in ("qasper", "nq") for rule in ov[c] if rule != "rules" for v in ov[c][rule]["union_minus_first"].values())
    mac("OverlapAllSig", "yes" if allsig else "no")
    same_chg = all(ov[c][rule]["significance_changed"] == ov[c]["any"]["significance_changed"] for c in ("qasper", "nq") for rule in ("min50", "prop50"))
    mac("OverlapSameChanged", "yes" if same_chg else "no")
    if ch:
        ce = [100 * v["delta"] for c in ch for cs in ch[c] for v in ch[c][cs]["union_minus_first"].values()]
        mac("ChunkShiftMin", f"{min(ce):.1f}"); mac("ChunkShiftMax", f"{max(ce):.1f}")
        mac("ChunkAllPositive", "yes" if min(ce) > 0 else "no")
        ex = [100 * v["delta"] for c in ch for cs in ch[c] for v in ch[c][cs]["neural_minus_bm25"].values()]
        mac("ChunkExcessMin", f"{min(ex):.1f}"); mac("ChunkExcessMax", f"{max(ex):.1f}")
        mac("ChunkExcessNegative", sum(1 for x in ex if x < 0)); mac("ChunkExcessN", len(ex))
        mac("ChunkExcessSig", sum(1 for c in ch for cs in ch[c] for v in ch[c][cs]["neural_minus_bm25"].values() if v["p_perm_holm"] < 0.05))


def extra_figures(qa: dict, na: dict) -> None:
    """Figure 2: forest plot of union - first hit@1; Figure 3: annotator disagreement and penalised evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    ov = json.loads((REV / "results/audit/overlap_sensitivity.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 3.1), sharex=True)
    for ax, (label, a, key, sign, c) in zip(axes, (("QASPER", qa, "first_minus_union", -1, "qasper"),
                                                   ("Natural Questions", na, "union_minus_first", 1, "nq")), strict=True):
        models = [m for m in ORDER if m in a["models"]]
        eff = a["reference_effects"][key]["models"]
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            d = 100 * sign * eff[m]["delta_hit@1"]
            lo, hi = sorted(100 * sign * x for x in eff[m]["cluster95"])
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color="C0", ms=4.5, lw=1.1, capsize=2,
                        label="overlap $\\geq$ 1 character (study), 95% interval" if i == 0 else None)
            for rule, mk, col, nm, dy in (("min50", "s", "C1", "overlap $\\geq$ 50 characters", 0.28),
                                          ("prop50", "^", "C2", "overlap $\\geq$ 50% of span or chunk", -0.28)):
                ax.plot(100 * ov[c][rule]["union_minus_first"][m]["delta"], y + dy, mk, color=col, ms=4,
                        label=nm if i == 0 else None)
        ax.set_yticks(range(len(models)), [NAMES[m] for m in reversed(models)], fontsize=7)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(label)
        ax.set_xlabel("Union $-$ first, hit@1 (points)")
        ax.set_xlim(0, 12)
        ax.grid(axis="x", color="0.9", lw=0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "fig_reference_effects.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_reference_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.45, 2.7), gridspec_kw={"width_ratios": [1, 1.4]})
    labels = ["QASPER", "NQ"]
    parts = [("pairs_identical", "identical", "0.75"), ("pairs_overlapping_not_identical", "overlapping", "C0"),
             ("pairs_disjoint", "disjoint", "C3")]
    left = np.zeros(2)
    for k, nm, col in parts:
        v = np.array([100 * qa["annotator_agreement"][k], 100 * na["annotator_agreement"][k]])
        ax1.barh([1, 0], v, left=left, color=col, label=nm, height=0.55)
        for y, (l0, w) in zip([1, 0], zip(left, v), strict=True):
            ax1.text(l0 + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7, color="white" if col != "0.75" else "black")
        left += v
    ax1.set_yticks([1, 0], labels)
    ax1.set_xlabel("Annotator pairs (per cent)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 1.9)
    ax1.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               handlelength=1.0, columnspacing=0.8)
    ax1.set_title("Gold chunks of two annotators")
    short = {"bm25": "BM25", "minilm": "MiniLM", "granite30m": "Granite", "colbert_small": "ColBERT",
             "ce_msmarco_minilm": "CE", "jina_turbo": "Jina"}
    for a, mk, col, nm in ((qa, "o", "C0", "QASPER"), (na, "s", "C1", "NQ")):
        models = [m for m in ORDER if m in a["models"]]
        x = [a["schemes"]["union"]["metrics"][m]["hit@1"] for m in models]
        y = [100 * a["penalised_valid_evidence"][m]["share_of_misses_marked_by_another_annotator"] for m in models]
        ax2.plot(x, y, mk, color=col, ms=4.5, label=nm, ls="none")
        # label only the weakest and the strongest model of each corpus; the others are close together
        for j in (int(np.argmin(x)), int(np.argmax(x))):
            ax2.annotate(short[models[j]], (x[j], y[j]), fontsize=7, xytext=(4, -3), textcoords="offset points")
    ax2.set_xlabel("hit@1 under the union reference")
    ax2.set_ylabel("Rank-1 misses marked by\nanother annotator (per cent)")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.set_title("Penalised evidence (one marker per model)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_disagreement.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_disagreement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_table_sources() -> None:
    """Cambridge requires a source for every table: add one to captions that lack it."""
    import re
    for f in GEN.glob("tab_*.tex"):
        s = f.read_text(encoding="utf-8")
        if "Source:" in s:
            continue
        src = (" Source: Hugging Face model cards; revisions as resolved by the local model cache." if f.name == "tab_models.tex"
               else " Source: this study, computed from the released per-question orderings.")
        s2 = re.sub(r"(\\caption\{.*?)\}(\n\\label)", lambda m: m.group(1) + src + "}" + m.group(2), s, count=1, flags=re.S)
        f.write_text(s2, encoding="utf-8")


def audit_outputs(qa: dict, na: dict) -> None:
    """Macros and tables from results/audit/*.json and the BGE-M3 subset analysis."""
    aud_p = REV / "results/audit/statistical_audit.json"
    if aud_p.exists():
        aud = json.loads(aud_p.read_text(encoding="utf-8"))
        mac("AuditMetricsChecked", aud["qasper"]["A_metrics"]["values_checked"] + aud["nq"]["A_metrics"]["values_checked"])
        for p, c in (("Q", "qasper"), ("N", "nq")):
            r = aud[c]
            lo, hi = r["C_alpha"]["cluster_bootstrap_95"]
            mac(f"{p}AlphaLo", f"{lo:.2f}"); mac(f"{p}AlphaHi", f"{hi:.2f}")
            mac(f"{p}AlphaPerQMedian", f"{r['C_alpha']['per_question_alpha_median']:.2f}")
            # independent re-implementation of the primary test (stat_audit.py, own RNG) vs the analysis
            agree = all(v["reimplementation_agrees_with_primary"] for v in r["E_cluster_permutation"].values())
            mac(f"{p}ReimplAgree", "agrees" if agree else "does not agree")
            g = r["G_random_gold_control"]
            mac(f"{p}ControlN", num(g["n_questions_union_adds_chunks"]))
            ratios = [v["ratio_real_to_random"] for v in g["models"].values() if v["ratio_real_to_random"]]
            mac(f"{p}ControlRatioMin", f"{min(ratios):.1f}"); mac(f"{p}ControlRatioMax", f"{max(ratios):.1f}")
        allr = [v["ratio_real_to_random"] for c in ("qasper", "nq") for v in aud[c]["G_random_gold_control"]["models"].values() if v["ratio_real_to_random"]]
        mac("ControlAllMin", f"{min(allr):.1f}"); mac("ControlAllMax", f"{max(allr):.1f}")
        # position-matched control (scripts/position_control.py, added 27 Nov 2026)
        pos_p = REV / "results/audit/position_control.json"
        pos = json.loads(pos_p.read_text(encoding="utf-8")) if pos_p.exists() else None
        if pos:
            for p, c in (("Q", "qasper"), ("N", "nq")):
                pr = [v["ratio_real_to_matched"] for v in pos[c]["models"].values()]
                mac(f"{p}PosRatioMin", f"{min(pr):.1f}"); mac(f"{p}PosRatioMax", f"{max(pr):.1f}")
                mac(f"{p}PosExact", f"{100 * pos[c]['matching']['exact_distance_share']:.0f}")
                mac(f"{p}PosWithinOne", f"{100 * pos[c]['matching']['within_1_chunk_share']:.0f}")
                mac(f"{p}PosAdjacent", f"{100 * pos[c]['added_chunk_distance']['adjacent_share']:.0f}")
                mac(f"{p}PosAllCIPositive", "yes" if all(v["real_minus_matched_cluster95"][0] > 0
                                                        for v in pos[c]["models"].values()) else "no")
            pa = [v["ratio_real_to_matched"] for c in ("qasper", "nq") for v in pos[c]["models"].values()]
            mac("PosAllMin", f"{min(pa):.1f}"); mac("PosAllMax", f"{max(pa):.1f}")
        for p, c in (("Q", "qasper"), ("N", "nq")):
            hs = aud[c]["H_split_sensitivity"]
            mac(f"{p}SplitsDiffer", sum(1 for v in hs.values() if sorted(v["sig_first"]) != sorted(v["sig_union"])))
            mac(f"{p}SplitsN", len(hs))
            neural_more = all(v["union_minus_first_hit@1"][m] >= v["union_minus_first_hit@1"]["bm25"] - 1e-12
                              for v in hs.values() for m in v["union_minus_first_hit@1"] if m != "bm25")
            mac(f"{p}SplitsNeuralMore", "yes" if neural_more else "no")
        pw = aud["nq"]["F_power"]["first:bm25-minilm"]
        mac("NFirstPairPower", f"{100 * pw['power_at_observed']:.0f}")
        mac("NFirstPairMDD", f"{100 * pw['min_detectable_difference_80pct']:.1f}")
        pu = aud["nq"]["F_power"]["union:bm25-minilm"]
        mac("NUnionPairPower", f"{100 * pu['power_at_observed']:.0f}")
        # control table
        out = ["\\begin{table}[t]",
               "\\caption{Random-gold control. On the questions where the union adds chunks to the first annotator's "
               "gold, the gain in hit@1 when the added chunks are the other annotators' evidence (real) and when the "
               "same number of random non-gold chunks is added instead, drawn uniformly from the document (uniform) or "
               "at the same distance, in chunk positions, from the first annotator's gold as the real added chunks "
               "(position-matched). Random gains: mean and central 95\\% range over 200 draws. Gains in percentage "
               "points; ratio: real gain divided by the mean random gain.}",
               "\\label{tab:control}", "\\centering\\small", "\\begin{tabular}{llccccc}", "\\hline",
               " & & & \\multicolumn{2}{c}{Uniform} & \\multicolumn{2}{c}{Position-matched}\\\\",
               "Corpus & Model & Real & Random & Ratio & Random & Ratio\\\\", "\\hline"]
        for label, c in (("QASPER", "qasper"), ("NQ", "nq")):
            g = aud[c]["G_random_gold_control"]
            pm = pos[c]["models"] if pos else {}
            first = True
            for m in [x for x in ORDER if x in g["models"]]:
                v = g["models"][m]
                lo, hi = v["gain_random_extra_95"]
                row = (f"{label + ' ($n$=' + num(g['n_questions_union_adds_chunks']) + ')' if first else ''} & {NAMES[m]} & "
                       f"{100 * v['gain_real_union']:.1f} & {100 * v['gain_random_extra_mean']:.1f} "
                       f"[{100 * lo:.1f}, {100 * hi:.1f}] & {v['ratio_real_to_random']:.1f}")
                if m in pm:
                    w = pm[m]
                    plo, phi = w["gain_matched_95"]
                    row += f" & {100 * w['gain_matched_mean']:.1f} [{100 * plo:.1f}, {100 * phi:.1f}] & {w['ratio_real_to_matched']:.1f}"
                else:
                    row += " & -- & --"
                out.append(row + "\\\\")
                first = False
            out.append("\\hline")
        out += ["\\end{tabular}", "\\end{table}"]
        (GEN / "tab_control.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
        # robustness table: splits. Two panels so that nothing is clipped: (a) union - first per model,
        # one numeric column per model; (b) only the pairs whose significance differs between the two
        # references, in a wide wrapping column. Significance: primary cluster permutation test, Holm.
        short = {"bm25": "BM25", "minilm": "MiniLM", "ce_msmarco_minilm": "CE", "granite30m": "Granite",
                 "jina_turbo": "Jina", "colbert_small": "ColBERT", "bge_m3": "BGE-M3"}
        cols = [m for m in ORDER if m in qa["models"]]
        split_names = {"dev": "Dev.", "test": "Test", "file0": "File 0", "file1": "File 1"}
        out = ["\\begin{table}[t]",
               "\\caption{Robustness across sample splits (QASPER development and test questions; the two NQ validation "
               "files). (a) Union minus first-annotator hit@1 in points, per model ($n$: questions with gold under both "
               "references). (b) Model pairs whose significance differs between the first-annotator and union references "
               "(primary cluster permutation test, Holm over all pairs within the split); ``union only'' means significant "
               "under the union but not under the first-annotator reference. --: model not run on that corpus. "
               "Source: this study, computed from the released per-question orderings.}",
               "\\label{tab:splits}", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
               "\\textit{(a) Union $-$ first, hit@1 (points)}\\\\[2pt]",
               "\\begin{tabular}{lll" + "c" * len(cols) + "}", "\\hline",
               "Corpus & Split & $n$ & " + " & ".join(short[m] for m in cols) + "\\\\", "\\hline"]
        changes = []
        for label, c in (("QASPER", "qasper"), ("NQ", "nq")):
            for split, v in aud[c]["H_split_sensitivity"].items():
                d = v["union_minus_first_hit@1"]
                vals = " & ".join((f"{100 * d[m]:+.1f}".replace("-", "$-$") if m in d else "--") for m in cols)
                out.append(f"{label} & {split_names.get(split, split)} & {num(v['n'])} & {vals}\\\\")
                fs, us = set(v["sig_first"]), set(v["sig_union"])
                lab = lambda x: short[x.split('>')[0]] + " $>$ " + short[x.split('>')[1]]
                items = [lab(x) + " (union only)" for x in sorted(us - fs)] + [lab(x) + " (first only)" for x in sorted(fs - us)]
                changes.append((label, split_names.get(split, split), ", ".join(items) or "none"))
                label = ""
            out.append("\\hline")
        out += ["\\end{tabular}", "\\par\\medskip", "\\textit{(b) Pairs whose significance differs between the references}\\\\[2pt]",
                "\\begin{tabular}{llp{0.62\\textwidth}}", "\\hline", "Corpus & Split & Pairs\\\\", "\\hline"]
        prev = None
        for corp, split, items in changes:
            out.append(f"{corp if corp else ''} & {split} & {items}\\\\")
        out += ["\\hline", "\\end{tabular}", "\\end{table}"]
        (GEN / "tab_splits.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
        # neural-minus-BM25 excess of the reference effect (cluster_inference.json, primary test)
        ci_p = REV / "results/audit/cluster_inference.json"
        if ci_p.exists():
            ci = json.loads(ci_p.read_text(encoding="utf-8"))
            for p, c in (("Q", "qasper"), ("N", "nq")):
                g = ci[c]["neural_minus_bm25_reference_effect"]
                ex = [100 * v["delta"] for v in g.values()]
                mac(f"{p}ExcessMin", f"{min(ex):.1f}"); mac(f"{p}ExcessMax", f"{max(ex):.1f}")
                sig = [m for m in ORDER if m in g and g[m]["p_perm_holm"] < 0.05]
                mac(f"{p}ExcessSigN", len(sig)); mac(f"{p}ExcessN", len(g))
                def eng(xs):
                    xs = [NAMES[m] for m in xs]
                    return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
                mac(f"{p}ExcessSigModels", eng(sig))
                ns = [m for m in ORDER if m in g and g[m]["p_perm_holm"] >= 0.05]
                mac(f"{p}ExcessNonSigModels", eng(ns))
                mac(f"{p}ExcessMaxP", f"{max(g[m]['p_perm_holm'] for m in g):.3f}")
                lo = min(min(v["ci95_cluster"]) for v in g.values())
                mac(f"{p}ExcessCILoMin", f"{100 * lo:.1f}")
                mac(f"{p}ExcessAllCIPositive", "yes" if all(v["ci95_cluster"][0] > 0 for v in g.values()) else "no")
                # sample splits: do neural models gain more than BM25 in every split (point estimates)?
            ab = sum(1 for c in ("qasper", "nq") for s in ci[c]["splits"].values()
                     for m, x in s["union_minus_first_hit@1"].items() if m != "bm25" and x < s["union_minus_first_hit@1"]["bm25"])
            mac("SplitNeuralBelowBM", ab)
    tie_p = REV / "results/audit/tie_audit.json"
    if tie_p.exists():
        t = json.loads(tie_p.read_text(encoding="utf-8"))
        mac("QTiedTop", t["qasper"]["questions_with_tied_top_score"]); mac("NTiedTop", t["nq"]["questions_with_tied_top_score"])
        mx = max(abs(v["difference_points"]) for c in ("qasper", "nq") for v in t[c]["by_reference"].values())
        mac("TieMaxDiff", f"{mx:.2f}")
    adj_p = REV / "results/audit/adjudication_audit.json"
    if adj_p.exists():
        a = json.loads(adj_p.read_text(encoding="utf-8"))
        mac("AdjAUC", f"{a['auc_nli_vs_author']:.2f}")
        mac("AdjFlagNo", len(a["flag_author_no_but_nli_high"])); mac("AdjFlagYes", len(a["flag_author_yes_but_nli_low"]))
        mac("AdjUnreadHigh", len(a["unread_with_nli_high"]))
        u = a["under_coverage"]
        mac("AdjUpper", f"{100 * u['upper_bound_all_flags_and_unread_high']:.1f}")
        mac("AdjLower", f"{100 * u['if_flagged_yes_were_no']:.1f}")
    rows = []
    for label, f, key, sign in (("QASPER (290)", "qasper_sub290", "first_minus_union", -1),
                                ("NQ (300)", "nq_sub300", "union_minus_first", 1)):
        sp = REV / f"results/{f}/analysis.json"
        if not sp.exists():
            continue
        s_ = json.loads(sp.read_text(encoding="utf-8"))
        eff = s_["reference_effects"][key]["models"]
        first = True
        for m in [x for x in ORDER if x in s_["models"]]:
            lo, hi = sorted(sign * x for x in eff[m]["cluster95"])
            cell = f"{100 * sign * eff[m]['delta_hit@1']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]".replace("-", "$-$")
            rows.append(f"{label if first else ''} & {NAMES[m]} & {s_['schemes']['first']['metrics'][m]['hit@1']:.3f} & "
                        f"{s_['schemes']['union']['metrics'][m]['hit@1']:.3f} & {cell}\\\\")
            first = False
            if m == "bge_m3":
                mac("BgeMThreeShift", f"{100 * sign * eff[m]['delta_hit@1']:.1f}")
            if m == "colbert_small" and f == "nq_sub300":
                mac("ColbertNQShift", f"{100 * sign * eff[m]['delta_hit@1']:.1f}")
        rows.append("\\hline")
    if rows:
        out = ["\\begin{table}[t]",
               "\\caption{Every model on the earlier study's questions (290 QASPER, 300 NQ), including the models too slow "
               "for the full samples (BGE-M3 on QASPER; ColBERT-small on NQ): hit@1 under the first-annotator and union "
               "references and the paired union minus first difference (points; cluster bootstrap 95\\% interval, QASPER paper and NQ Wikipedia page).}",
               "\\label{tab:subsets}", "\\centering\\small", "\\begin{tabular}{llccc}", "\\hline",
               "Subset & Model & First & Union & Union $-$ first\\\\", "\\hline"] + rows + ["\\end{tabular}", "\\end{table}"]
        (GEN / "tab_subsets.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    qa, qraw = load("qasper")
    na, nraw = load("nq")
    corpus_macros("Q", qa, qraw)
    corpus_macros("N", na, nraw)
    adj = json.loads((REPO / "results/goldspan_adjudication/remaining37/estimate_combined.json").read_text(encoding="utf-8"))
    (GEN / "tab_main.tex").write_text(table_main(qa, na) + "\n", encoding="utf-8")
    (GEN / "tab_effects.tex").write_text(table_effects(qa, na) + "\n", encoding="utf-8")
    (GEN / "tab_conclusions.tex").write_text(table_conclusions(qa, na) + "\n", encoding="utf-8")
    (GEN / "tab_incomplete.tex").write_text(table_incomplete(qa, na, adj) + "\n", encoding="utf-8")
    (GEN / "tab_reach.tex").write_text(table_reach(qa, na) + "\n", encoding="utf-8")
    audit_outputs(qa, na)
    sensitivity_outputs()
    extra_figures(qa, na)
    seven, _ = seven_model()
    (GEN / "tab_seven.tex").write_text(seven + "\n", encoding="utf-8")
    pens = [v["share_of_misses_marked_by_another_annotator"] for a in (qa, na) for v in a["penalised_valid_evidence"].values()]
    mac("PenAllMin", f"{100 * min(pens):.0f}")
    mac("PenAllMax", f"{100 * max(pens):.0f}")
    draws = [v["draw_range"] for a in (qa, na) for v in a["single_annotator"]["hit@1"].values()]
    mac("DrawRangeAllMax", f"{100 * max(draws):.1f}")
    full = [m for m in ORDER if m in qa["models"] and m in na["models"]]
    names = [NAMES[m] for m in full]
    mac("FullModels", ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0])
    (GEN / "tab_models.tex").write_text(table_models(full) + "\n", encoding="utf-8")
    mac("ReproN", num(290 + 300))
    mac("TotalN", num(len(qraw["questions"]) + len(nraw["questions"])))
    figures(qa, na)
    add_table_sources()
    lines = ["% Generated by scripts/make_tables.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(macros)} macros; tables and figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
