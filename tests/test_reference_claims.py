"""The localisation claims under harmonised evidence references (an exploratory analysis).

`scripts/reference_sensitivity_claims.py` re-evaluates the claims of
results/localisation/README.md when both corpora are scored against the same kind of
reference. Under the manuscript's own pairing every claim must hold and every number must
be the manuscript's, or the check itself is wrong.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

EXPLORATORY = REPO / "results" / "exploratory"
CLAIMS = EXPLORATORY / "reference_sensitivity_claims.json"


class TestPairReading:
    PROFILE = {"holm_significant": {"bm25_vs_bge": -1, "minilm_vs_bge_reranker_base": -1}}

    def test_a_pair_reads_the_same_in_either_order(self):
        import reference_sensitivity_claims as claims

        assert claims._pair(self.PROFILE, "bge", "bm25") == 1
        assert claims._pair(self.PROFILE, "bm25", "bge") == -1
        assert claims._pair(self.PROFILE, "bge", "e5") == 0

    def test_losers_are_the_models_beaten_after_correction(self):
        import reference_sensitivity_claims as claims

        assert claims._losers(self.PROFILE) == {"bm25", "minilm"}


needs_results = pytest.mark.skipif(
    not all((EXPLORATORY / d / "per_question.json").exists()
            for d in ("nq_reference_sensitivity", "qasper_reference_sensitivity")) or not CLAIMS.exists(),
    reason="the exploratory per-question results have not been produced")


@needs_results
def test_every_claim_holds_under_the_manuscript_pairing():
    stored = json.loads(CLAIMS.read_text(encoding="utf-8"))
    assert all(stored["manuscript_pairing_reproduces_summaries"].values())
    failing = [c["id"] for c in stored["views"]["manuscript"]["claims"] if not c["holds"]]
    assert not failing, failing


@needs_results
def test_the_manuscript_pairing_gives_the_manuscript_numbers():
    stored = json.loads(CLAIMS.read_text(encoding="utf-8"))["views"]["manuscript"]["profiles"]
    for corpus in ("qasper", "nq"):
        summary = json.loads((REPO / "results" / "localisation" / f"summary_{corpus}.json").read_text(encoding="utf-8"))
        for model, hit1 in stored[corpus]["hit@1"].items():
            assert hit1 == summary["models"][model]["hit@1"], (corpus, model)
        assert stored[corpus]["any_localiser_hit@1"] == summary["union"]["any_hit@1"]


@needs_results
def test_the_claims_recompute_from_the_per_question_files():
    import reference_sensitivity_claims as claims

    stored = json.loads(CLAIMS.read_text(encoding="utf-8"))
    fresh = claims.build(EXPLORATORY / "nq_reference_sensitivity", EXPLORATORY / "qasper_reference_sensitivity",
                         REPO / "results" / "localisation", stored["settings"]["n_boot"])
    assert fresh["manuscript_pairing_reproduces_summaries"] == stored["manuscript_pairing_reproduces_summaries"]
    for name, view in stored["views"].items():
        assert fresh["views"][name]["claims"] == view["claims"], name


@needs_results
def test_each_pairing_uses_the_references_it_declares():
    stored = json.loads(CLAIMS.read_text(encoding="utf-8"))["views"]
    for corpus, directory in (("nq", "nq_reference_sensitivity"), ("qasper", "qasper_reference_sensitivity")):
        summary = json.loads((EXPLORATORY / directory / "summary.json").read_text(encoding="utf-8"))
        for view in stored.values():
            scheme = view["references"][corpus]
            for model, hit1 in view["profiles"][corpus]["hit@1"].items():
                assert hit1 == summary["schemes"][scheme]["models"][model]["hit@1"], (corpus, scheme, model)
