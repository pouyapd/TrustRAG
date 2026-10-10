"""Alternative evidence references for Natural Questions (an exploratory analysis).

The NQ loader takes the first annotator's long answer as the only gold evidence, although
the validation split is five-way annotated. These tests cover what rebuilds the reference
from every annotator: the loader's opt-in record of all annotators' spans, the reference
schemes, the chunk and rank arithmetic, and the tracked exploratory results, which must
give the manuscript's numbers back exactly under the first annotator.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.data.loaders.nq_parquet import NaturalQuestionsParquetLoader  # noqa: E402
from src.evaluation.references import (  # noqa: E402
    OFFICIAL_MIN_NON_NULL,
    SCHEMES,
    draw_single,
    first_gold_rank,
    gold_chunks,
    non_null_count,
    reference_annotators,
    reference_rank,
)

RESULTS = REPO / "results" / "exploratory" / "nq_reference_sensitivity"


def ok(i, start, end):
    return {"annotator": i, "status": "ok", "start_char": start, "end_char": end}


def null(i):
    return {"annotator": i, "status": "null"}


def markup(i):
    return {"annotator": i, "status": "not_representable", "token_range": [0, 1]}


# --- the schemes ---------------------------------------------------------------

class TestReferenceSchemes:
    # annotators 1 and 3 marked the same paragraph, annotator 2 a different one
    ANNS = [null(0), ok(1, 10, 20), ok(2, 30, 40), ok(3, 10, 20), null(4)]

    def test_first_is_the_first_annotator_who_marked_a_long_answer(self):
        assert reference_annotators(self.ANNS, "first") == [1]

    def test_union_takes_every_usable_span(self):
        assert reference_annotators(self.ANNS, "union") == [1, 2, 3]

    def test_official_matches_nq_threshold_of_two_annotators(self):
        assert OFFICIAL_MIN_NON_NULL == 2
        assert reference_annotators(self.ANNS, "official") == [1, 2, 3]
        assert reference_annotators([ok(0, 1, 5), null(1), null(2)], "official") is None

    def test_markup_only_answer_counts_toward_the_threshold_but_is_not_a_span(self):
        anns = [ok(0, 1, 5), markup(1)]
        assert non_null_count(anns) == 2
        assert reference_annotators(anns, "official") == [0]
        assert reference_annotators(anns, "union") == [0]

    def test_first_is_none_when_the_first_answer_is_pure_markup(self):
        """The loader skips such questions, so the study never used one."""
        anns = [markup(0), ok(1, 1, 5)]
        assert reference_annotators(anns, "first") is None
        assert reference_annotators(anns, "union") == [1]

    def test_plurality_takes_the_span_most_annotators_chose(self):
        assert reference_annotators(self.ANNS, "plurality") == [1, 3]

    def test_plurality_keeps_every_tied_span(self):
        assert reference_annotators([ok(0, 1, 2), ok(1, 3, 4)], "plurality") == [0, 1]

    def test_no_usable_annotation_means_no_reference(self):
        for scheme in SCHEMES:
            assert reference_annotators([null(0), markup(1)], scheme) is None

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            reference_annotators(self.ANNS, "majority_of_experts")

    def test_single_draws_are_seeded_and_uniform_over_annotators(self):
        draws = [draw_single(self.ANNS, random.Random(7))[0] for _ in range(3)]
        assert len(set(draws)) == 1                     # same seed, same draw
        rng = random.Random(0)
        counts = Counter(draw_single(self.ANNS, rng)[0] for _ in range(6000))
        assert set(counts) == {1, 2, 3}
        assert all(abs(c / 6000 - 1 / 3) < 0.03 for c in counts.values())

    def test_single_draw_needs_a_usable_annotation(self):
        with pytest.raises(ValueError):
            draw_single([null(0)], random.Random(0))


# --- chunk and rank arithmetic --------------------------------------------------

class TestRankArithmetic:
    CHUNKS = [(0, 10), (8, 20), (20, 30)]

    def test_overlap_is_positive_and_half_open(self):
        assert gold_chunks(self.CHUNKS, [(19, 21)]) == {1, 2}
        assert gold_chunks(self.CHUNKS, [(10, 20)]) == {1}          # touching is not overlapping
        assert gold_chunks(self.CHUNKS, [(30, 40)]) == set()

    def test_union_of_spans_is_union_of_chunk_sets(self):
        assert gold_chunks(self.CHUNKS, [(0, 5), (25, 26)]) == {0, 2}

    def test_first_gold_rank_is_one_based(self):
        assert first_gold_rank([2, 0, 1], {0}) == 2
        assert first_gold_rank([2, 0, 1], {2, 1}) == 1
        assert first_gold_rank([2, 0, 1], set()) is None

    def test_reference_rank_is_the_earliest_annotator_rank(self):
        assert reference_rank({1: 4, 2: 2, 3: 9}, [1, 3]) == 4
        assert reference_rank({1: 4, 2: 2, 3: 9}, [1, 2, 3]) == 2

    def test_a_union_rank_is_never_worse_than_any_member(self):
        rng = random.Random(3)
        for _ in range(200):
            order = list(range(12))
            rng.shuffle(order)
            sets = {a: set(rng.sample(range(12), rng.randint(1, 3))) for a in range(4)}
            ranks = {a: first_gold_rank(order, s) for a, s in sets.items()}
            union = first_gold_rank(order, set().union(*sets.values()))
            assert union == reference_rank(ranks, list(sets)) == min(ranks.values())


# --- the summary, on a synthetic per-question file --------------------------------

def _question(qid, doc, annotations, ranks):
    return {"question_id": qid, "example_id": qid, "doc_id": doc, "D": 10,
            "annotations": annotations, "ranks": ranks, "rank_source": {}}


class TestSummary:
    def test_span_relations(self):
        import nq_reference_sensitivity as analysis

        assert analysis.span_relation((10, 20), (10, 20)) == "identical"
        assert analysis.span_relation((10, 20), (20, 30)) == "disjoint"      # touching
        assert analysis.span_relation((10, 20), (12, 15)) == "inside_first"
        assert analysis.span_relation((10, 20), (5, 25)) == "contains_first"
        assert analysis.span_relation((10, 20), (15, 25)) == "partial_overlap"

    def test_schemes_on_a_synthetic_file(self):
        import nq_reference_sensitivity as analysis

        a = {**ok(0, 0, 10), "gold_chunks": [0]}
        b = {**ok(1, 50, 60), "gold_chunks": [5]}
        questions = [
            # annotators disagree: the first annotator's chunk is ranked 4th, the other's 1st
            _question("q1", "d1", [a, b], {"x": {"0": 4, "1": 1}, "y": {"0": 1, "1": 3}}),
            # one annotator only: below NQ's official threshold
            _question("q2", "d2", [a, null(1)], {"x": {"0": 2}, "y": {"0": 2}}),
            # two annotators on the same span
            _question("q3", "d3", [a, {**ok(1, 0, 10), "gold_chunks": [0]}],
                      {"x": {"0": 1, "1": 1}, "y": {"0": 5, "1": 5}}),
        ]
        out = analysis.summarise_questions(questions, ["x", "y"], n_boot=20, n_draws=10)
        first, union = out["schemes"]["first"], out["schemes"]["union"]
        assert first["n_questions"] == union["n_questions"] == 3
        assert first["models"]["x"]["hit@1"] == round(1 / 3, 4)              # only q3
        assert union["models"]["x"]["hit@1"] == round(2 / 3, 4)              # q1 and q3
        assert out["schemes"]["official"]["n_questions"] == 2                # q2 dropped
        assert out["schemes"]["agreement_subset"]["n_questions"] == 2        # q2 and q3
        assert out["union_vs_first"]["x"]["hit@1_gained"] == 1
        assert first["any_localiser_hit@1"] == round(2 / 3, 4)               # y on q1, x on q3
        assert union["any_localiser_hit@1"] == round(2 / 3, 4)
        profile = out["annotation_profile"]
        assert profile["questions_where_spans_cover_different_chunks"] == 1
        assert profile["questions_by_most_consequential_relation"] == {
            "disjoint": 1, "identical": 1, "single_annotator": 1}


# --- the loader's opt-in record of every annotator -------------------------------

TOKENS = ["<P>", "Alpha", "beta", "</P>", "<P>", "Gamma", "delta", "</P>", "<Table>", "</Table>"]
IS_HTML = [True, False, False, True, True, False, False, True, True, True]


def _long(start, end):
    return {"candidate_index": 0, "start_token": start, "end_token": end, "start_byte": 0, "end_byte": 0}


def _short(starts, ends):
    return {"start_token": starts, "end_token": ends, "start_byte": [0] * len(starts),
            "end_byte": [0] * len(ends), "text": [""] * len(starts)}


def _nq_parquet(tmp_path):
    """One five-way annotated row in the HuggingFace NQ parquet shape."""
    row = {
        "id": "42",
        "document": {"html": "", "title": "Page", "url": "http://wiki/Page",
                     "tokens": {"token": TOKENS, "is_html": IS_HTML,
                                "start_byte": [0] * len(TOKENS), "end_byte": [0] * len(TOKENS)}},
        "question": {"text": "which greek letters", "tokens": ["which", "greek", "letters"]},
        "long_answer_candidates": {"start_token": [], "end_token": [], "start_byte": [],
                                   "end_byte": [], "top_level": []},
        "annotations": {
            "id": ["a0", "a1", "a2", "a3", "a4"],
            "long_answer": [_long(-1, -1), _long(0, 4), _long(4, 8), _long(8, 10), _long(0, 4)],
            "short_answers": [_short([], []), _short([1], [2]), _short([5], [6]), _short([], []),
                              _short([1], [2])],
            "yes_no_answer": [-1, -1, -1, -1, -1],
        },
    }
    path = tmp_path / "nq.parquet"
    pq.write_table(pa.Table.from_pylist([row]), str(path))
    return path


class TestLoaderAnnotatorSpans:
    def test_off_by_default_and_records_unchanged(self, tmp_path):
        loader = NaturalQuestionsParquetLoader()
        assert loader.keep_annotator_spans is False
        record = loader.load(_nq_parquet(tmp_path)).questions[0]
        assert set(record.metadata) == {"example_id", "n_annotations", "n_reference_answers",
                                        "document_url"}

    def test_every_annotator_is_kept_in_annotation_order(self, tmp_path):
        loader = NaturalQuestionsParquetLoader()
        loader.keep_annotator_spans = True
        result = loader.load(_nq_parquet(tmp_path))
        record, text = result.questions[0], result.documents[0].text
        entries = record.metadata["annotator_long_answers"]
        assert [e["annotator"] for e in entries] == [0, 1, 2, 3, 4]
        assert [e["status"] for e in entries] == ["null", "ok", "ok", "not_representable", "ok"]
        assert text[entries[1]["start_char"]:entries[1]["end_char"]] == "Alpha beta"
        assert text[entries[2]["start_char"]:entries[2]["end_char"]] == "Gamma delta"
        assert entries[3]["token_range"] == [8, 10]       # an empty table: markup only

    def test_the_evidence_is_still_the_first_annotator_either_way(self, tmp_path):
        path = _nq_parquet(tmp_path)
        plain = NaturalQuestionsParquetLoader().load(path).questions[0]
        loader = NaturalQuestionsParquetLoader()
        loader.keep_annotator_spans = True
        kept = loader.load(path).questions[0]
        assert kept.question_id == plain.question_id
        assert [(s.start_char, s.end_char) for s in kept.supporting_spans] == \
               [(s.start_char, s.end_char) for s in plain.supporting_spans]
        first = kept.metadata["annotator_long_answers"][reference_annotators(
            kept.metadata["annotator_long_answers"], "first")[0]]
        assert (first["start_char"], first["end_char"]) == \
               (plain.supporting_spans[0].start_char, plain.supporting_spans[0].end_char)

    def test_switching_it_on_for_one_loader_leaves_others_off(self, tmp_path):
        loader = NaturalQuestionsParquetLoader()
        loader.keep_annotator_spans = True
        assert NaturalQuestionsParquetLoader().keep_annotator_spans is False


# --- the tracked exploratory results ----------------------------------------------

needs_results = pytest.mark.skipif(not (RESULTS / "per_question.json").exists(),
                                   reason="the exploratory NQ results have not been produced")


def _load():
    import nq_reference_sensitivity as analysis

    data = json.loads((RESULTS / "per_question.json").read_text(encoding="utf-8"))
    return analysis, data["questions"], [m for m in analysis.MODELS if m in data["localisers"]]


@needs_results
def test_first_annotator_gives_back_the_manuscript_ranks():
    from localisation_robustness import load_ranks

    analysis, questions, models = _load()
    frozen = load_ranks(REPO / "results" / "localisation", "nq")
    rows = analysis.scheme_rows(questions, models, "first")
    assert len(rows) == 300
    for row in rows:
        for m in models:
            assert row["rank"][m] == frozen[m][row["question_id"]], (m, row["question_id"])


@needs_results
def test_union_never_ranks_worse_than_the_first_annotator():
    analysis, questions, models = _load()
    first = {r["question_id"]: r for r in analysis.scheme_rows(questions, models, "first")}
    for row in analysis.scheme_rows(questions, models, "union"):
        for m in models:
            assert row["rank"][m] <= first[row["question_id"]]["rank"][m]


@needs_results
def test_summary_recomputes_from_the_per_question_file():
    analysis, questions, models = _load()
    stored = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    fresh = analysis.summarise_questions(questions, models, n_boot=50, n_draws=20)
    for scheme, block in stored["schemes"].items():
        assert fresh["schemes"][scheme]["n_questions"] == block["n_questions"]
        assert fresh["schemes"][scheme]["models"] == block["models"]
        for pair, test in block["pairwise"].items():
            for key in ("hit@1_diff", "mcnemar_exact_p", "mcnemar_holm_adjusted_p"):
                assert fresh["schemes"][scheme]["pairwise"][pair][key] == test[key]
    assert fresh["holm_significant_pairs"] == stored["holm_significant_pairs"]
    assert fresh["annotation_profile"] == stored["annotation_profile"]


@needs_results
def test_the_manuscript_check_passed():
    check = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["manuscript_check"]
    assert check["metrics_identical"] and check["pairwise_tests_identical"], check["differences"]


@needs_results
def test_no_dataset_text_is_stored():
    """NQ is CC BY-SA 3.0; the tracked files hold ids, character ranges and ranks only."""
    data = json.loads((RESULTS / "per_question.json").read_text(encoding="utf-8"))
    allowed_question = {"question_id", "example_id", "doc_id", "D", "annotations", "ranks", "rank_source"}
    allowed_annotation = {"annotator", "status", "start_char", "end_char", "token_range", "gold_chunks"}
    for q in data["questions"]:
        assert set(q) <= allowed_question
        for a in q["annotations"]:
            assert set(a) <= allowed_annotation
