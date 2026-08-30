"""Copy the key numbers out of a run into a version-controlled results file.

`reports/` is git-ignored, so a published number would otherwise live only on
the machine that produced it. This extracts the summary fields a reader needs
to check a claim -- and the provenance needed to reproduce it -- into
`results/`, which is tracked.

Row-level data and inference records stay out: they are large, and for NQ they
contain substantial verbatim Wikipedia text that should not be redistributed
from this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Summary keys worth preserving. Row-level data is deliberately excluded.
KEEP = (
    "total", "precision_at_k_mean", "recall_at_k_mean", "mrr_mean",
    "token_overlap_mean", "faithfulness_mean", "failure_rate", "failure_modes",
    "failure_rate_v2", "failure_modes_v2", "attribution", "retrieval_corrected",
    "answer_corrected", "abstention", "evidence", "confidence_intervals",
    "statistical_notes", "experiment", "provenance", "taxonomy_v2",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate a run into results/")
    parser.add_argument("--summary", required=True, help="summary.json from a run")
    parser.add_argument("--out", required=True, help="destination JSON in results/")
    args = parser.parse_args()

    source = Path(args.summary)
    if not source.exists():
        print(f"no summary at {source}", file=sys.stderr)
        return 1

    report = json.loads(source.read_text(encoding="utf-8"))
    curated = {key: report[key] for key in KEEP if key in report}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(curated, indent=2), encoding="utf-8")
    print(f"curated {len(curated)} sections -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
