"""Reproduce the complete study from raw data to tables.

One command runs every experiment reported in docs/EXPERIMENTS.md, in the same
configuration, and regenerates the curated result files under `results/`.

    python scripts/reproduce_study.py --all            # the original study
    python scripts/reproduce_study.py --embedder-sweep # robustness to embedder
    python scripts/reproduce_study.py --topk-sweep     # robustness to retrieval depth
    python scripts/reproduce_study.py --multihop       # second multi-hop corpus
    python scripts/reproduce_study.py --everything     # all of the above

Nothing here needs an API key. The generator is a deterministic extractive
control and the embedders run locally, so the entire retrieval and evidence
measurement — which is what the reported findings are about — reproduces
offline once the corpora are downloaded. The optional real-language-model
experiment is a separate script (`scripts/run_llm_experiment.py`) precisely so
that this one stays runnable by anyone.

Prerequisites: `pip install -r requirements.txt`, then fetch the corpora as
documented in docs/DATASETS.md. The script checks for them and tells you what
is missing rather than failing obscurely partway through.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Fixed so a rerun samples the same questions. The loaders are deterministic
#: given (file, limit, split); nothing here draws random numbers, and the
#: bootstrap in the statistics layer carries its own fixed seed.
STUDY_SEED = 20260826


@dataclass(frozen=True)
class Experiment:
    """One configuration in the study."""

    tag: str
    dataset: str
    raw: str
    split: str
    limit: int
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 5
    embedder: str = "minilm"
    #: Whether this run backs a headline number or a robustness check.
    role: str = "headline"

    @property
    def out_dir(self) -> Path:
        return REPO / "reports" / "experiments" / self.tag


#: The three headline runs, then the chunk-size sensitivity sweep. Frozen: the
#: numbers in docs/EXPERIMENTS.md come from exactly these configurations.
EXPERIMENTS = [
    Experiment("qasper_dev_300", "qasper", "data/raw/qasper-dev-v0.3.json", "dev", 300),
    Experiment("nq_val_300_fixed", "nq", "data/raw/nq-validation-0.parquet", "validation", 300),
    Experiment("hotpot_150", "hotpotqa",
               "data/raw/hotpot-distractor-validation-0.parquet", "validation", 150),
    Experiment("qasper_c128", "qasper", "data/raw/qasper-dev-v0.3.json", "dev", 300,
               chunk_size=128, role="robustness"),
    Experiment("qasper_c512", "qasper", "data/raw/qasper-dev-v0.3.json", "dev", 300,
               chunk_size=512, role="robustness"),
]

#: Second multi-hop corpus. The quantifier effect was measured on HotpotQA
#: alone; 2WikiMultihopQA is structurally comparable, so the identical pipeline
#: runs over it and a difference in result is a difference in the data.
MULTIHOP_EXPERIMENTS = [
    Experiment("twowiki_150", "2wiki", "data/raw/2wiki-dev.parquet", "dev", 150,
               role="multihop"),
]

#: Embedder robustness. Held constant: dataset, questions, corpus, chunking,
#: retrieval depth, metrics. Varied: the embedding model, and nothing else.
#: Run on one granularity-dominated corpus and one quantifier-dominated corpus
#: so both effects are tested rather than only the headline one.
EMBEDDER_SWEEP_BASES = [
    ("qasper_dev_300", EXPERIMENTS[0]),
    ("hotpot_150", EXPERIMENTS[2]),
]
EMBEDDER_SWEEP_KEYS = ("minilm", "mpnet", "bge", "e5")

#: Retrieval-depth robustness. Every depth is retrieved *natively* — the query
#: is re-issued at each k rather than one deep ranking being truncated.
#:
#: Truncation was measured first and looked exactly equivalent on all three
#: study corpora (160 query x depth comparisons, zero disagreements). It was
#: still rejected: with the deterministic hash embedder it is possible to build
#: a corpus where two neighbours are near-tied and the approximate index orders
#: them differently depending on how many results were requested, so the
#: equivalence is a property of these corpora rather than a guarantee. The
#: corpus is embedded once and each depth re-queries it, which costs seconds and
#: removes the assumption.
TOPK_VALUES = (1, 3, 5, 10, 20)
TOPK_MAX = max(TOPK_VALUES)


def topk_bases() -> list[Experiment]:
    """The corpora the depth sweep runs over: two granularity, two quantifier."""
    return [EXPERIMENTS[0], EXPERIMENTS[1], EXPERIMENTS[2], MULTIHOP_EXPERIMENTS[0]]


def topk_runs() -> list[Experiment]:
    """The per-depth runs, one report directory each."""
    return [
        replace(base, tag=f"{base.tag}_topk_k{k}", top_k=k, role="topk")
        for base in topk_bases()
        for k in TOPK_VALUES
    ]


def embedder_sweep_runs() -> list[Experiment]:
    """Every embedder against every sweep corpus."""
    runs = []
    for _, base in EMBEDDER_SWEEP_BASES:
        for key in EMBEDDER_SWEEP_KEYS:
            runs.append(
                replace(base, tag=f"{base.tag}_emb_{key}", embedder=key, role="embedder")
            )
    return runs


def run(command: list[str]) -> int:
    """Run a step, streaming nothing but its exit status."""
    print(f"  $ {' '.join(command[1:])}")
    started = time.time()
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"    FAILED after {elapsed:.0f}s")
        print("    " + "\n    ".join(result.stderr.strip().splitlines()[-8:]))
    else:
        print(f"    ok ({elapsed:.0f}s)")
    return result.returncode


def check_corpora(experiments: list[Experiment]) -> list[str]:
    """Which raw files are missing. Checked up front, not halfway through."""
    missing = []
    for experiment in experiments:
        path = REPO / experiment.raw
        if not path.exists() and experiment.raw not in missing:
            missing.append(experiment.raw)
    return missing


def decomp_path(tag: str, k: int | None = None) -> Path:
    name = f"decomp_{tag}.json" if k is None else f"decomp_{tag}_at_k{k}.json"
    return REPO / "reports" / "experiments" / name


def run_experiment_step(experiment: Experiment, python: str, skip_existing: bool) -> int:
    """Inference for one configuration, unless its records already exist."""
    records = experiment.out_dir / "inference.jsonl"
    if skip_existing and records.exists():
        print("    reusing existing run")
        return 0
    return run([
        python, "scripts/run_experiment.py",
        "--dataset", experiment.dataset, "--raw", experiment.raw,
        "--split", experiment.split, "--limit", str(experiment.limit),
        "--top-k", str(experiment.top_k),
        "--chunk-size", str(experiment.chunk_size),
        "--chunk-overlap", str(experiment.chunk_overlap),
        "--embedder", experiment.embedder,
        "--out", str(experiment.out_dir.relative_to(REPO)),
        "--tag", experiment.tag,
    ])


def ablate(experiment: Experiment, python: str, k: int | None = None) -> int:
    """Apply the A/B/C decomposition to a finished run. No model is called."""
    records = experiment.out_dir / "inference.jsonl"
    if not records.exists():
        return 1
    command = [
        python, "scripts/run_ablation.py",
        "--records", str(records.relative_to(REPO)),
        "--out", str(decomp_path(experiment.tag, k).relative_to(REPO)),
        "--tag", experiment.tag if k is None else f"{experiment.tag}@k={k}",
    ]
    if k is not None:
        command += ["--k", str(k)]
    return run(command)


def read_decomp(tag: str, k: int | None = None) -> dict | None:
    path = decomp_path(tag, k)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["comparison"]


def print_abc_table(title: str, rows: list[tuple[str, dict]]) -> None:
    """The A/B/C table, in the layout the documentation quotes."""
    print(f"\n=== {title} ===")
    print(f"{'run':<26}{'n':>5}{'ch/doc':>8}{'A doc':>8}{'B quant':>9}"
          f"{'C span':>8}{'quant':>8}{'gran':>8}")
    for label, c in rows:
        if not c:
            continue
        cond, steps = c["conditions"], c["steps"]
        print(f"{label:<26}{c['n_paired']:>5}"
              f"{str(c['median_chunks_per_relevant_document']):>8}"
              f"{cond['A_document_any']:>8.3f}{cond['B_document_quantified']:>9.3f}"
              f"{cond['C_span_quantified']:>8.3f}"
              f"{steps['quantifier_A_to_B']['absolute_gap_pp']:>7.1f}p"
              f"{steps['granularity_B_to_C']['absolute_gap_pp']:>7.1f}p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the TrustRAG study")
    parser.add_argument("--all", action="store_true",
                        help="the original study: headline runs and chunk-size sweep")
    parser.add_argument("--headline-only", action="store_true",
                        help="skip the chunk-size robustness sweep")
    parser.add_argument("--embedder-sweep", action="store_true",
                        help="four embedding models, everything else held constant")
    parser.add_argument("--topk-sweep", action="store_true",
                        help=f"retrieval depth k in {list(TOPK_VALUES)}")
    parser.add_argument("--multihop", action="store_true",
                        help="the second multi-hop corpus (2WikiMultihopQA)")
    parser.add_argument("--everything", action="store_true",
                        help="every deterministic experiment in the repository")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse runs whose output already exists")
    args = parser.parse_args()

    do_study = args.all or args.headline_only or args.everything
    do_embedder = args.embedder_sweep or args.everything
    do_topk = args.topk_sweep or args.everything
    do_multihop = args.multihop or args.everything

    if not any((do_study, do_embedder, do_topk, do_multihop)):
        parser.print_help()
        return 1

    planned: list[Experiment] = []
    if do_study:
        planned += [e for e in EXPERIMENTS if not args.headline_only or e.role == "headline"]
    if do_multihop:
        planned += MULTIHOP_EXPERIMENTS
    if do_embedder:
        planned += embedder_sweep_runs()
    if do_topk:
        planned += topk_runs()

    missing = check_corpora(planned)
    if missing:
        print("Missing raw corpora. See docs/DATASETS.md for download commands:")
        for path in missing:
            print(f"  {path}")
        return 1

    python = sys.executable
    print(f"TrustRAG reproduction — {len(planned)} runs, seed {STUDY_SEED}")
    print("No API key required; the generator is a deterministic extractive control.\n")
    failures = 0

    # ---- the original study ----
    study = [e for e in EXPERIMENTS if not args.headline_only or e.role == "headline"]
    if do_study:
        for experiment in study:
            print(f"[{experiment.tag}] {experiment.dataset} {experiment.split} "
                  f"n={experiment.limit} chunk={experiment.chunk_size} k={experiment.top_k}")
            failures += bool(run_experiment_step(experiment, python, args.skip_existing))
            failures += bool(ablate(experiment, python))
            if experiment.role == "headline":
                failures += bool(run([
                    python, "scripts/curate_results.py",
                    "--summary", str((experiment.out_dir / "summary.json").relative_to(REPO)),
                    "--out", f"results/{experiment.tag}.json",
                ]))

    # ---- second multi-hop corpus ----
    if do_multihop:
        for experiment in MULTIHOP_EXPERIMENTS:
            print(f"[{experiment.tag}] {experiment.dataset} {experiment.split} "
                  f"n={experiment.limit} chunk={experiment.chunk_size} k={experiment.top_k}")
            failures += bool(run_experiment_step(experiment, python, args.skip_existing))
            failures += bool(ablate(experiment, python))
            failures += bool(run([
                python, "scripts/curate_results.py",
                "--summary", str((experiment.out_dir / "summary.json").relative_to(REPO)),
                "--out", f"results/{experiment.tag}.json",
            ]))

    # ---- embedder robustness ----
    if do_embedder:
        for experiment in embedder_sweep_runs():
            print(f"[{experiment.tag}] embedder={experiment.embedder}")
            failures += bool(run_experiment_step(experiment, python, args.skip_existing))
            failures += bool(ablate(experiment, python))

    # ---- retrieval-depth robustness ----
    if do_topk:
        for base in topk_bases():
            print(f"[{base.tag}] retrieval depths {list(TOPK_VALUES)}, native per k, one index")
            done = [
                (REPO / "reports" / "experiments" / f"{base.tag}_topk_k{k}"
                 / "inference.jsonl").exists()
                for k in TOPK_VALUES
            ]
            if args.skip_existing and all(done):
                print("    reusing existing runs")
            else:
                failures += bool(run([
                    python, "scripts/run_experiment.py",
                    "--dataset", base.dataset, "--raw", base.raw,
                    "--split", base.split, "--limit", str(base.limit),
                    "--chunk-size", str(base.chunk_size),
                    "--chunk-overlap", str(base.chunk_overlap),
                    "--embedder", base.embedder,
                    "--topk-values", ",".join(str(k) for k in TOPK_VALUES),
                    "--out", f"reports/experiments/{base.tag}_topk",
                    "--tag", f"{base.tag}_topk",
                ]))
            for k in TOPK_VALUES:
                failures += bool(ablate(
                    replace(base, tag=f"{base.tag}_topk_k{k}", top_k=k), python
                ))

    # ---- tables ----
    if do_study:
        print_abc_table("the study", [(e.tag, read_decomp(e.tag)) for e in study])
    if do_multihop:
        print_abc_table(
            "multi-hop corpora (quantifier effect)",
            [("hotpot_150 (original)", read_decomp("hotpot_150"))]
            + [(e.tag, read_decomp(e.tag)) for e in MULTIHOP_EXPERIMENTS],
        )
    if do_embedder:
        for base_tag, _ in EMBEDDER_SWEEP_BASES:
            print_abc_table(
                f"embedder robustness — {base_tag}",
                [(f"{key}", read_decomp(f"{base_tag}_emb_{key}"))
                 for key in EMBEDDER_SWEEP_KEYS],
            )
    if do_topk:
        for base in topk_bases():
            print_abc_table(
                f"retrieval depth — {base.tag}",
                [(f"k={k}", read_decomp(f"{base.tag}_topk_k{k}")) for k in TOPK_VALUES],
            )

    print(f"\n{'FAILED: ' + str(failures) + ' step(s)' if failures else 'All steps completed.'}")
    print("Curated results in results/, full reports in reports/experiments/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
