"""Reproduce the complete three-dataset study from raw data to tables.

One command runs every experiment reported in docs/EXPERIMENTS.md, in the same
configuration, and regenerates the curated result files under `results/`.

    python scripts/reproduce_study.py --all

Nothing here needs an API key. The generator is a deterministic extractive
control and the embedder runs locally, so the entire retrieval and evidence
measurement — which is what the reported findings are about — reproduces
offline once the corpora are downloaded.

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
from dataclasses import dataclass
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


#: The three headline runs, then the chunk-size sensitivity sweep.
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


def check_corpora() -> list[str]:
    """Which raw files are missing. Checked up front, not halfway through."""
    missing = []
    for experiment in EXPERIMENTS:
        path = REPO / experiment.raw
        if not path.exists() and experiment.raw not in missing:
            missing.append(experiment.raw)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the full TrustRAG study")
    parser.add_argument("--all", action="store_true", help="run every experiment")
    parser.add_argument("--headline-only", action="store_true",
                        help="skip the chunk-size robustness sweep")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse runs whose output already exists")
    args = parser.parse_args()

    if not (args.all or args.headline_only):
        parser.print_help()
        return 1

    missing = check_corpora()
    if missing:
        print("Missing raw corpora. See docs/DATASETS.md for download commands:")
        for path in missing:
            print(f"  {path}")
        return 1

    chosen = [e for e in EXPERIMENTS if args.all or e.role == "headline"]
    python = sys.executable

    print(f"TrustRAG study reproduction — {len(chosen)} experiments, seed {STUDY_SEED}")
    print("No API key required; the generator is a deterministic extractive control.\n")

    failures = 0
    for experiment in chosen:
        print(f"[{experiment.tag}] {experiment.dataset} {experiment.split} "
              f"n={experiment.limit} chunk={experiment.chunk_size} k={experiment.top_k}")
        records = experiment.out_dir / "inference.jsonl"

        if args.skip_existing and records.exists():
            print("    reusing existing run")
        else:
            failures += bool(run([
                python, "scripts/run_experiment.py",
                "--dataset", experiment.dataset, "--raw", experiment.raw,
                "--split", experiment.split, "--limit", str(experiment.limit),
                "--top-k", str(experiment.top_k),
                "--chunk-size", str(experiment.chunk_size),
                "--chunk-overlap", str(experiment.chunk_overlap),
                "--embedder", experiment.embedder,
                "--out", str(experiment.out_dir.relative_to(REPO)),
                "--tag", experiment.tag,
            ]))

        if records.exists():
            failures += bool(run([
                python, "scripts/run_ablation.py",
                "--records", str(records.relative_to(REPO)),
                "--out", f"reports/experiments/decomp_{experiment.tag}.json",
                "--tag", experiment.tag,
            ]))
            if experiment.role == "headline":
                failures += bool(run([
                    python, "scripts/curate_results.py",
                    "--summary", str((experiment.out_dir / "summary.json").relative_to(REPO)),
                    "--out", f"results/{experiment.tag}.json",
                ]))

    print("\n=== summary table ===")
    print(f"{'experiment':<20}{'chunks/doc':>11}{'A doc':>8}{'B quant':>9}"
          f"{'C span':>8}{'quant':>8}{'gran':>8}")
    for experiment in chosen:
        path = REPO / "reports" / "experiments" / f"decomp_{experiment.tag}.json"
        if not path.exists():
            continue
        c = json.loads(path.read_text(encoding="utf-8"))["comparison"]
        cond, steps = c["conditions"], c["steps"]
        print(f"{experiment.tag:<20}{str(c['median_chunks_per_relevant_document']):>11}"
              f"{cond['A_document_any']:>8.3f}{cond['B_document_quantified']:>9.3f}"
              f"{cond['C_span_quantified']:>8.3f}"
              f"{steps['quantifier_A_to_B']['absolute_gap_pp']:>7.1f}p"
              f"{steps['granularity_B_to_C']['absolute_gap_pp']:>7.1f}p")

    print(f"\n{'FAILED: ' + str(failures) + ' step(s)' if failures else 'All steps completed.'}")
    print("Curated results in results/, full reports in reports/experiments/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
