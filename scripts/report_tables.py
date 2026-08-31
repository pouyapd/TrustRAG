"""Emit the documentation's result tables from the stored result files.

Every number in docs/EXPERIMENTS.md that describes a run comes from here rather
than from someone reading a JSON file and typing what they saw. Transcription is
where published tables quietly drift from the data that produced them, and a
reader has no way to catch it.

    python scripts/report_tables.py --table all

Each table prints as GitHub-flavoured Markdown on stdout. Runs that have not
been executed are reported as missing rather than omitted silently, so a
half-finished sweep cannot masquerade as a complete one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REPORTS = REPO / "reports" / "experiments"


def load(tag: str) -> dict | None:
    path = REPORTS / f"decomp_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["comparison"]


def pfmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1g}" if value >= 0.001 else f"{value:.1e}"


def step_cells(comparison: dict, step: str) -> tuple[str, str, str]:
    s = comparison["steps"][step]
    return (
        f"{s['absolute_gap_pp']:.1f} pp",
        f"{s['discordant_lost']}/{s['discordant_gained']}",
        pfmt(s["mcnemar"]["p_value"]),
    )


def table_embedders() -> str:
    from src.rag.embedders import EMBEDDERS

    out = []
    for corpus, base, step, name in (
        ("QASPER", "qasper_dev_300", "granularity_B_to_C", "granularity B→C"),
        ("HotpotQA", "hotpot_150", "quantifier_A_to_B", "quantifier A→B"),
    ):
        out.append(f"\n**{corpus} — {name}.** Corpus, questions, chunking (256/32) "
                   f"and k=5 held constant; only the embedder changes.\n")
        out.append("| Embedder | Family | dim | A | B | C | gap | discordant | McNemar p |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for key in ("minilm", "mpnet", "bge", "e5"):
            c = load(f"{base}_emb_{key}")
            spec = EMBEDDERS[key]
            short = spec.repo_id.split("/")[-1]
            if not c:
                out.append(f"| `{short}` | {spec.family} | {spec.dimension} "
                           f"| — | — | — | *not run* | — | — |")
                continue
            gap, disc, p = step_cells(c, step)
            cond = c["conditions"]
            out.append(
                f"| `{short}` | {spec.family} | {spec.dimension} "
                f"| {cond['A_document_any']:.3f} | {cond['B_document_quantified']:.3f} "
                f"| {cond['C_span_quantified']:.3f} | **{gap}** | {disc} | {p} |"
            )
    return "\n".join(out)


def table_topk() -> str:
    from scripts.reproduce_study import TOPK_VALUES

    out = []
    for corpus, base, step, name in (
        ("QASPER", "qasper_dev_300", "granularity_B_to_C", "granularity B→C"),
        ("Natural Questions", "nq_val_300_fixed", "granularity_B_to_C", "granularity B→C"),
        ("HotpotQA", "hotpot_150", "quantifier_A_to_B", "quantifier A→B"),
        ("2WikiMultihopQA", "twowiki_150", "quantifier_A_to_B", "quantifier A→B"),
    ):
        out.append(f"\n**{corpus} — {name}.** Every depth retrieved natively; "
                   f"corpus, chunking and embedder held constant.\n")
        out.append("| k | A | B | C | gap | discordant | McNemar p |")
        out.append("|---|---|---|---|---|---|---|")
        for k in TOPK_VALUES:
            c = load(f"{base}_topk_k{k}")
            if not c:
                out.append(f"| {k} | — | — | — | *not run* | — | — |")
                continue
            gap, disc, p = step_cells(c, step)
            cond = c["conditions"]
            out.append(
                f"| {k} | {cond['A_document_any']:.3f} | {cond['B_document_quantified']:.3f} "
                f"| {cond['C_span_quantified']:.3f} | **{gap}** | {disc} | {p} |"
            )
    return "\n".join(out)


def table_multihop() -> str:
    out = [
        "\n| | HotpotQA | 2WikiMultihopQA |",
        "|---|---|---|",
    ]
    a = load("hotpot_150")
    b = load("twowiki_150")
    if not (a and b):
        return "\n*One or both multi-hop runs are missing.*"

    def row(label, fn):
        return f"| {label} | {fn(a)} | {fn(b)} |"

    out.append(row("n", lambda c: str(c["n_paired"])))
    out.append(row("median chunks per gold document",
                   lambda c: str(c["median_chunks_per_relevant_document"])))
    out.append(row("evidence mode", lambda c: ", ".join(c["evidence_modes"])))
    out.append(row("A document, ANY", lambda c: f"{c['conditions']['A_document_any']:.3f}"))
    out.append(row("B document, quantified",
                   lambda c: f"**{c['conditions']['B_document_quantified']:.3f}**"))
    out.append(row("C span, quantified", lambda c: f"{c['conditions']['C_span_quantified']:.3f}"))
    out.append(row("**quantifier A→B**",
                   lambda c: f"**{step_cells(c, 'quantifier_A_to_B')[0]}** "
                             f"(p={step_cells(c, 'quantifier_A_to_B')[2]})"))
    out.append(row("discordant pairs",
                   lambda c: step_cells(c, "quantifier_A_to_B")[1]))
    out.append(row("granularity B→C",
                   lambda c: f"{step_cells(c, 'granularity_B_to_C')[0]} "
                             f"(p={step_cells(c, 'granularity_B_to_C')[2]})"))
    out.append(row("failures charged to retrieval, document-level",
                   lambda c: str(c["attribution"]["document_level"].get("retrieval", 0))))
    out.append(row("failures charged to retrieval, evidence-level",
                   lambda c: f"**{c['attribution']['evidence_level'].get('retrieval', 0)}**"))
    return "\n".join(out)


def table_attribution() -> str:
    out = [
        "\n| Corpus | n | retrieval (document-level) | retrieval (evidence-level) | change |",
        "|---|---|---|---|---|",
    ]
    for label, tag in (
        ("QASPER", "qasper_dev_300"),
        ("Natural Questions", "nq_val_300_fixed"),
        ("HotpotQA", "hotpot_150"),
        ("2WikiMultihopQA", "twowiki_150"),
    ):
        c = load(tag)
        if not c:
            out.append(f"| {label} | — | — | — | *not run* |")
            continue
        doc = c["attribution"]["document_level"].get("retrieval", 0)
        ev = c["attribution"]["evidence_level"].get("retrieval", 0)
        out.append(f"| {label} | {c['n_paired']} | {doc} | **{ev}** | ×{ev / doc:.1f} |"
                   if doc else
                   f"| {label} | {c['n_paired']} | {doc} | **{ev}** | — |")
    return "\n".join(out)


TABLES = {
    "embedders": table_embedders,
    "topk": table_topk,
    "multihop": table_multihop,
    "attribution": table_attribution,
}


BEGIN = "<!-- BEGIN generated: {name} -->"
END = "<!-- END generated: {name} -->"


def inject(path: Path) -> int:
    """Replace the content between each table's markers, in place.

    Idempotent: rerunning after more runs land refreshes the numbers without
    touching a word of the surrounding prose. A missing marker pair is reported
    rather than silently skipped, because a table that quietly stopped updating
    is exactly the failure this is meant to prevent.
    """
    text = path.read_text(encoding="utf-8")
    updated, missing = 0, []
    for name, builder in TABLES.items():
        begin, end = BEGIN.format(name=name), END.format(name=name)
        if begin not in text or end not in text:
            missing.append(name)
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + begin + "\n" + builder() + "\n" + end + tail
        updated += 1
    path.write_text(text, encoding="utf-8")
    print(f"updated {updated} table(s) in {path}")
    if missing:
        print(f"no markers for: {', '.join(missing)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit result tables as Markdown")
    parser.add_argument("--table", default="all",
                        choices=[*sorted(TABLES), "all"])
    parser.add_argument("--inject", default="",
                        help="rewrite the tables inside this Markdown file, in place")
    args = parser.parse_args()

    if args.inject:
        return inject(Path(args.inject))

    chosen = sorted(TABLES) if args.table == "all" else [args.table]
    for name in chosen:
        print(f"\n<!-- {name} -->")
        print(TABLES[name]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
