"""Tests for retrieval depth: evaluating at k, and how the sweep obtains each k.

Two separate things are pinned here.

**Applying a depth consistently.** `run_ablation.py` can evaluate a stored run at
a shallower cutoff, and all three conditions must honour the same k. A version
that truncated A and B but let C see every retrieved chunk would silently report
a smaller granularity gap, because C would be scored with more evidence
available than the conditions it is compared against.

**Why the sweep does not rely on that.** The cheap design would retrieve once at
k=20 and derive every shallower depth by truncation. On the three study corpora
this looked exactly equivalent — 160 query x depth comparisons, zero
disagreements. It is still not a guarantee: the index is approximate, and
near-tied neighbours can be ordered differently depending on how many results
were requested. `TestWhyTheSweepRetrievesNatively` builds a corpus where that
actually happens, so the reason `--topk-values` re-queries at every depth is
held in place by a test rather than by a comment.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ablation import (  # noqa: E402
    compare,
    condition_a_document_any,
    condition_b_document_quantified,
    condition_c_span,
    effective_k,
    retrieved_prefix,
)
from src.evaluation.records import InferenceRecord, RetrievedChunk  # noqa: E402
from src.evaluation.runner import score_records  # noqa: E402


def chunk(rank: int, doc_id: str, start: int, end: int) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank, chunk_id=f"{doc_id}_{rank}", doc_id=doc_id, source=f"{doc_id}.md",
        score=1.0 - rank / 100, text="x", start_char=start, end_char=end,
    )


def record_with(retrieved, spans, mode="any_sufficient", top_k=20) -> InferenceRecord:
    """A record whose gold evidence and retrieval are both stated explicitly."""
    return InferenceRecord(
        index=1,
        question="q",
        reference_answer="a",
        relevant_doc_ids=sorted({s["doc_id"] for s in spans}),
        predicted_answer="a",
        retrieved=retrieved,
        faithfulness=1.0,
        latency_ms=1.0,
        top_k=top_k,
        n_relevant_chunks=1,
        metadata={"supporting_spans": spans, "evidence_mode": mode},
    )


SPAN_A = {"doc_id": "doc_a", "start_char": 0, "end_char": 50, "text": "x"}
SPAN_B = {"doc_id": "doc_b", "start_char": 0, "end_char": 50, "text": "x"}


class TestEffectiveK:
    def test_explicit_k_wins(self):
        assert effective_k(record_with([], [SPAN_A], top_k=20), 3) == 3

    def test_none_falls_back_to_the_run(self):
        assert effective_k(record_with([], [SPAN_A], top_k=7), None) == 7

    def test_prefix_is_rank_ordered_and_truncated(self):
        retrieved = [chunk(i, "doc_a", 0, 10) for i in range(1, 6)]
        prefix = retrieved_prefix(record_with(retrieved, [SPAN_A]), 2)
        assert [c.rank for c in prefix] == [1, 2]


class TestConditionsRespectK:
    def test_document_hit_outside_k_is_not_counted(self):
        """The gold document sits at rank 4; at k=3 retrieval has not found it."""
        retrieved = [chunk(1, "x", 0, 10), chunk(2, "y", 0, 10),
                     chunk(3, "z", 0, 10), chunk(4, "doc_a", 0, 60)]
        record = record_with(retrieved, [SPAN_A])
        assert condition_a_document_any(record, 3) is False
        assert condition_a_document_any(record, 4) is True

    def test_span_coverage_outside_k_is_not_counted(self):
        retrieved = [chunk(1, "x", 0, 10), chunk(2, "doc_a", 0, 60)]
        record = record_with(retrieved, [SPAN_A])
        assert condition_c_span(record, 1) is False
        assert condition_c_span(record, 2) is True

    def test_multi_hop_second_document_outside_k(self):
        """all_required: one of two gold documents inside k is still a failure."""
        retrieved = [chunk(1, "doc_a", 0, 60), chunk(2, "q", 0, 10), chunk(3, "doc_b", 0, 60)]
        record = record_with(retrieved, [SPAN_A, SPAN_B], mode="all_required")
        assert condition_b_document_quantified(record, 1) is False
        assert condition_b_document_quantified(record, 3) is True
        assert condition_c_span(record, 1) is False
        assert condition_c_span(record, 3) is True

    def test_deeper_k_is_monotone_for_document_retrieval(self):
        """Retrieving more can never lose a document you already had."""
        retrieved = [chunk(i, f"d{i}", 0, 10) for i in range(1, 11)]
        retrieved.append(chunk(11, "doc_a", 0, 60))
        record = record_with(retrieved, [SPAN_A])
        seen = [condition_a_document_any(record, k) for k in (1, 3, 5, 10, 11)]
        assert seen == sorted(seen, key=bool)  # never flips back to False


class TestCompareAtK:
    def _records(self):
        # One question whose gold document arrives at rank 4.
        deep = [chunk(1, "x", 0, 10), chunk(2, "y", 0, 10),
                chunk(3, "z", 0, 10), chunk(4, "doc_a", 0, 60)]
        return [record_with(deep, [SPAN_A])]

    def test_rates_change_with_k(self):
        records = self._records()
        rows = score_records(records)
        shallow = compare(records, rows, k=3)
        deep = compare(records, rows, k=4)
        assert shallow["conditions"]["A_document_any"] == 0.0
        assert deep["conditions"]["A_document_any"] == 1.0

    def test_k_is_recorded_in_the_output(self):
        records = self._records()
        rows = score_records(records)
        assert compare(records, rows, k=3)["top_k_evaluated"] == 3
        assert compare(records, rows)["top_k_evaluated"] == "as-retrieved"

    def test_default_matches_the_runs_own_k(self):
        """No --k must behave exactly as before this feature existed."""
        records = self._records()
        rows = score_records(records)
        assert compare(records, rows)["conditions"] == compare(records, rows, k=20)["conditions"]


class TestWhyTheSweepRetrievesNatively:
    """Records the measurement that decided how the depth sweep is built.

    The cheap design would retrieve once at k=20 and evaluate every shallower
    depth by truncating that ranking. On the three study corpora that looked
    exactly equivalent — 160 query x depth comparisons, zero disagreements —
    which is what made it tempting.

    It is not a guarantee. The index is approximate, and when two neighbours are
    near-tied it can order them differently depending on how many results were
    requested. This test builds such a corpus and asserts the disagreement is
    real, so the reason `--topk-values` re-queries at every depth is pinned by a
    test rather than left as a comment someone later "optimises" away.
    """

    def _store(self, tmp_path, texts):
        from src.rag.chunking import Chunk
        from src.rag.providers import HashEmbeddings
        from src.rag.vector_store import VectorStore

        store = VectorStore(
            HashEmbeddings(), persist_dir=str(tmp_path / "idx"), collection_name="prefix_test"
        )
        store.reset()
        store.add([
            Chunk(
                text=text, doc_id=f"doc_{i}", chunk_id=f"doc_{i}_0", source=f"doc_{i}.md",
                metadata={"chunk_index": 0, "token_count": 12,
                          "start_char": 0, "end_char": 60},
            )
            for i, text in enumerate(texts)
        ])
        return store

    def test_near_ties_can_reorder_with_requested_depth(self, tmp_path):
        """A corpus of near-identical vectors: the prefix property fails here.

        Whether any single index exhibits the reordering is itself approximate --
        HNSW graph construction is randomised, so one build may happen to agree with
        itself at every depth. Several independent indices are built and the property
        is asserted over their union; a single build made this test flaky.
        """
        n = 40
        disagreements = 0
        for build in range(5):
            store = self._store(
                tmp_path / f"build_{build}",
                [" ".join(["alpha"] * (i + 1) + ["beta"] * (n - i)) for i in range(n)],
            )
            for query in ("alpha", "beta", "alpha beta"):
                deep = [r.chunk_id for r in store.search(query, top_k=20)]
                for k in (1, 3, 5, 10):
                    if [r.chunk_id for r in store.search(query, top_k=k)] != deep[:k]:
                        disagreements += 1
            store.reset()
            if disagreements:
                break
        assert disagreements > 0, (
            "expected the approximate index to disagree with its own deeper ranking "
            "on a near-tied corpus, over 5 independent builds; if this now holds, "
            "truncation could be revisited"
        )

    def test_scores_are_non_increasing_within_one_query(self, tmp_path):
        """Whatever the depth, results come back best-first."""
        store = self._store(tmp_path, [
            "refund policy for annual subscribers",
            "invoice generation and billing cycles",
            "latency budget for retrieval",
            "chunking strategy and overlap",
            "evidence alignment by character offset",
        ])
        results = store.search("refund policy", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        store.reset()


@pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
def test_sweep_values_are_positive_and_ordered(k):
    from scripts.reproduce_study import TOPK_MAX, TOPK_VALUES

    assert k in TOPK_VALUES
    assert list(TOPK_VALUES) == sorted(TOPK_VALUES)
    assert max(TOPK_VALUES) == TOPK_MAX
