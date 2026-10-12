"""Alternative evidence references for QASPER (an exploratory analysis).

The QASPER loader merges every annotator's evidence paragraphs into one reference, while the
NQ loader keeps only the first annotator's long answer. These tests cover what rebuilds the
QASPER reference per annotator: the loader's opt-in record of each annotator's evidence, the
evidence-set schemes, the summary arithmetic, and the tracked exploratory results, which
must give the manuscript's numbers back exactly under the union of annotators.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.data.loaders.qasper import QasperLoader  # noqa: E402
from src.evaluation.references import (  # noqa: E402
    EVIDENCE_SET_SCHEMES,
    draw_single_evidence,
    evidence_spans,
    reference_spans,
)

RESULTS = REPO / "results" / "exploratory" / "qasper_reference_sensitivity"


def ok(i, *spans):
    return {"annotator": i, "status": "ok", "spans": [list(s) for s in spans]}


def status(i, name):
    return {"annotator": i, "status": name}


# --- the schemes ---------------------------------------------------------------

class TestEvidenceSetSchemes:
    # annotators 1 and 2 both marked (10, 20); annotator 2 also marked (30, 40)
    ANNS = [status(0, "unanswerable"), ok(1, (10, 20)), ok(2, (30, 40), (10, 20)), status(3, "no_evidence")]

    def test_union_takes_every_located_paragraph(self):
        assert reference_spans(self.ANNS, "union") == [(10, 20), (30, 40)]

    def test_first_is_the_first_annotator_who_gave_evidence(self):
        assert reference_spans(self.ANNS, "first") == [(10, 20)]

    def test_first_is_none_when_that_evidence_is_not_in_the_text(self):
        """As for NQ: the next annotator does not stand in for the first."""
        anns = [status(0, "not_representable"), ok(1, (10, 20))]
        assert reference_spans(anns, "first") is None
        assert reference_spans(anns, "union") == [(10, 20)]

    def test_plurality_takes_the_paragraphs_most_annotators_marked(self):
        assert reference_spans(self.ANNS, "plurality") == [(10, 20)]

    def test_plurality_is_the_union_when_no_paragraph_is_shared(self):
        anns = [ok(0, (1, 2)), ok(1, (3, 4), (5, 6))]
        assert reference_spans(anns, "plurality") == reference_spans(anns, "union") == [(1, 2), (3, 4), (5, 6)]

    def test_agreement_needs_a_paragraph_marked_twice(self):
        assert reference_spans(self.ANNS, "agreement") == [(10, 20)]
        assert reference_spans([ok(0, (1, 2)), ok(1, (3, 4))], "agreement") is None
        assert reference_spans([ok(0, (1, 2))], "agreement") is None

    def test_an_annotator_repeating_a_paragraph_counts_once(self):
        assert reference_spans([ok(0, (1, 2), (1, 2)), ok(1, (3, 4))], "agreement") is None

    def test_no_usable_annotation_means_no_reference(self):
        for scheme in EVIDENCE_SET_SCHEMES:
            assert reference_spans([status(0, "unanswerable"), status(1, "not_representable")], scheme) is None

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            reference_spans(self.ANNS, "official")

    def test_evidence_spans_are_empty_unless_located(self):
        assert evidence_spans(ok(0, (5, 9))) == [(5, 9)]
        assert evidence_spans(status(0, "not_representable")) == []

    def test_single_draws_are_seeded_and_uniform_over_annotators(self):
        draws = [draw_single_evidence(self.ANNS, random.Random(7)) for _ in range(3)]
        assert draws[0] == draws[1] == draws[2]           # same seed, same draw
        rng = random.Random(0)
        counts = Counter(tuple(draw_single_evidence(self.ANNS, rng)) for _ in range(4000))
        assert set(counts) == {((10, 20),), ((10, 20), (30, 40))}
        assert all(abs(c / 4000 - 1 / 2) < 0.03 for c in counts.values())

    def test_single_draw_needs_a_usable_annotation(self):
        with pytest.raises(ValueError):
            draw_single_evidence([status(0, "unanswerable")], random.Random(0))


# --- the summary, on a synthetic per-question file --------------------------------

def _span(start, end, annotators, chunks):
    return {"start_char": start, "end_char": end, "annotators": annotators, "gold_chunks": chunks}


def _question(qid, doc, annotations, spans, ranks, invariant):
    return {"question_id": qid, "paper_id": doc, "doc_id": doc, "D": 10, "annotations": annotations,
            "spans": spans, "reference_invariant": invariant, "ranks": ranks, "rank_source": {}}


def _synthetic():
    p0, p1 = (0, 10), (50, 60)
    return [
        # annotators disagree: the first annotator's paragraph is ranked 4th by x, the other's 1st
        _question("q1", "d1", [ok(0, p0), ok(1, p1)],
                  [_span(*p0, [0], [0]), _span(*p1, [1], [5])],
                  {"x": {"union": 1, "spans": [4, 1]}, "y": {"union": 1, "spans": [1, 3]}}, False),
        # one annotator only: every reference is the union
        _question("q2", "d2", [ok(0, p0), status(1, "unanswerable")], [_span(*p0, [0], [0])],
                  {"x": {"union": 2, "spans": [2]}, "y": {"union": 2, "spans": None}}, True),
        # both annotators marked p0, the second also p1: plurality and agreement keep p0 only
        _question("q3", "d3", [ok(0, p0), ok(1, p0, p1)],
                  [_span(*p0, [0, 1], [0]), _span(*p1, [1], [5])],
                  {"x": {"union": 1, "spans": [3, 1]}, "y": {"union": 5, "spans": [5, 6]}}, False),
    ]


class TestSummary:
    def test_schemes_on_a_synthetic_file(self):
        import qasper_reference_sensitivity as analysis

        out = analysis.summarise_questions(_synthetic(), ["x", "y"], n_boot=20, n_draws=10)
        union, first = out["schemes"]["union"], out["schemes"]["first"]
        assert union["n_questions"] == first["n_questions"] == 3
        assert union["models"]["x"]["hit@1"] == round(2 / 3, 4)             # q1 and q3
        assert first["models"]["x"]["hit@1"] == 0.0                          # p0 is 4th, 2nd, 3rd
        assert out["schemes"]["agreement"]["n_questions"] == 1               # q3 only
        assert out["schemes"]["agreement"]["models"]["x"]["hit@1"] == 0.0   # p0 alone is 3rd
        assert out["schemes"]["plurality"]["models"]["x"]["hit@1"] == round(1 / 3, 4)  # q1 = union
        assert out["schemes"]["agreement_subset"]["n_questions"] == 1        # q2
        assert out["first_vs_union"]["x"]["hit@1_lost"] == 2
        assert union["mean_gold_spans_per_question"] == round(5 / 3, 3)
        profile = out["annotation_profile"]
        assert profile["questions_with_two_or_more_usable_annotators"] == 2
        assert profile["of_which_share_a_paragraph"] == 1
        assert profile["questions_where_some_reference_differs_from_union"] == 2

    def test_stored_invariance_flags_are_checked(self):
        import qasper_reference_sensitivity as analysis

        questions = _synthetic()
        questions[0]["reference_invariant"] = True
        with pytest.raises(ValueError):
            analysis.summarise_questions(questions, ["x", "y"], n_boot=20, n_draws=10)

    def test_a_missing_paragraph_rank_is_an_error_not_a_guess(self):
        import qasper_reference_sensitivity as analysis

        q1 = _synthetic()[0]
        q1["ranks"]["y"]["spans"] = None
        with pytest.raises(ValueError):
            analysis.question_rank(q1, "y", [(0, 10)])
        assert analysis.question_rank(q1, "y", [(0, 10), (50, 60)]) == 1     # the union's own rank

    def test_annotators_covering_the_same_chunks_can_still_disagree_on_paragraphs(self):
        import qasper_reference_sensitivity as analysis

        # p0 and p1 fall in chunk 0, p2 in chunk 5. Both annotators cover chunks {0, 5}, but
        # the only paragraph they share is p2, so agreement and plurality narrow the gold to
        # chunk 5: the question is not reference-invariant.
        p0, p1, p2 = (0, 10), (12, 20), (50, 60)
        q = _question("q", "d", [ok(0, p0, p2), ok(1, p1, p2)],
                      [_span(*p0, [0], [0]), _span(*p1, [1], [0]), _span(*p2, [0, 1], [5])], {}, False)
        assert analysis.reference_invariant(q) is False
        # different paragraphs in the same chunk, none shared: every reference is chunk 0
        q2 = _question("q", "d", [ok(0, p0), ok(1, p1)],
                       [_span(*p0, [0], [0]), _span(*p1, [1], [0])], {}, True)
        assert analysis.reference_invariant(q2) is True


# --- the loader's opt-in record of every annotator -------------------------------

PARAGRAPH_A = "Alpha paragraph about the method."
PARAGRAPH_B = "Beta paragraph about the results."


def _qasper_json(tmp_path):
    """One paper, one question, five answer blocks covering every annotator status."""
    def answer(evidence, unanswerable=False):
        return {"answer": {"unanswerable": unanswerable, "extractive_spans": [] if unanswerable else ["x"],
                           "yes_no": None, "free_form_answer": "", "evidence": evidence}}

    paper = {
        "title": "A paper", "abstract": "The abstract.",
        "full_text": [{"section_name": "Intro", "paragraphs": [PARAGRAPH_A, PARAGRAPH_B]}],
        "qas": [{"question": "what method?", "question_id": "q1", "answers": [
            answer([], unanswerable=True),
            answer([PARAGRAPH_A, "FLOAT SELECTED: Table 1"]),
            answer([PARAGRAPH_B, PARAGRAPH_A]),
            answer([]),
            answer(["FLOAT SELECTED: Figure 2", "text that is in no paragraph"]),
        ]}],
    }
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps({"0000.00001": paper}), encoding="utf-8")
    return path


class TestLoaderAnnotatorEvidence:
    def test_off_by_default_and_records_unchanged(self, tmp_path):
        loader = QasperLoader()
        assert loader.keep_annotator_spans is False
        record = loader.load(_qasper_json(tmp_path)).questions[0]
        assert set(record.metadata) == {"paper_id", "n_annotations", "n_evidence_spans",
                                        "unresolved_float_evidence", "unresolved_unmatched_evidence"}

    def test_every_annotator_is_kept_in_answer_order(self, tmp_path):
        loader = QasperLoader()
        loader.keep_annotator_spans = True
        result = loader.load(_qasper_json(tmp_path))
        record, body = result.questions[0], result.documents[0].text
        entries = record.metadata["annotator_evidence"]
        assert [e["annotator"] for e in entries] == [0, 1, 2, 3, 4]
        assert [e["status"] for e in entries] == ["unanswerable", "ok", "ok", "no_evidence", "not_representable"]
        assert [body[s:e] for s, e in entries[1]["spans"]] == [PARAGRAPH_A]
        assert [body[s:e] for s, e in entries[2]["spans"]] == [PARAGRAPH_B, PARAGRAPH_A]
        assert (entries[1]["unresolved_float"], entries[4]["unresolved_float"], entries[4]["unresolved_unmatched"]) == (1, 1, 1)
        assert "spans" not in entries[4]

    def test_the_union_of_annotators_is_the_loaders_evidence(self, tmp_path):
        path = _qasper_json(tmp_path)
        plain = QasperLoader().load(path).questions[0]
        loader = QasperLoader()
        loader.keep_annotator_spans = True
        kept = loader.load(path).questions[0]
        assert kept.question_id == plain.question_id
        loader_spans = [(s.start_char, s.end_char) for s in plain.supporting_spans]
        assert [(s.start_char, s.end_char) for s in kept.supporting_spans] == loader_spans
        assert reference_spans(kept.metadata["annotator_evidence"], "union") == sorted(loader_spans)
        assert {k: v for k, v in kept.metadata.items() if k != "annotator_evidence"} == plain.metadata

    def test_switching_it_on_for_one_loader_leaves_others_off(self):
        loader = QasperLoader()
        loader.keep_annotator_spans = True
        assert QasperLoader().keep_annotator_spans is False


# --- the tracked exploratory results ----------------------------------------------

needs_results = pytest.mark.skipif(not (RESULTS / "per_question.json").exists(),
                                   reason="the exploratory QASPER results have not been produced")


def _load():
    import qasper_reference_sensitivity as analysis

    data = json.loads((RESULTS / "per_question.json").read_text(encoding="utf-8"))
    return analysis, data["questions"], [m for m in analysis.MODELS if m in data["localisers"]]


@needs_results
def test_union_gives_back_the_manuscript_ranks():
    from localisation_robustness import load_ranks

    analysis, questions, models = _load()
    frozen = load_ranks(REPO / "results" / "localisation", "qasper")
    rows = analysis.scheme_rows(questions, models, "union")
    assert len(rows) == 290
    for row in rows:
        for m in models:
            assert row["rank"][m] == frozen[m][row["question_id"]], (m, row["question_id"])


@needs_results
def test_no_narrower_reference_ranks_better_than_the_union():
    analysis, questions, models = _load()
    union = {r["question_id"]: r for r in analysis.scheme_rows(questions, models, "union")}
    for scheme in ("first", "plurality", "agreement"):
        for row in analysis.scheme_rows(questions, models, scheme):
            for m in models:
                assert row["rank"][m] >= union[row["question_id"]]["rank"][m]
    for q in questions:
        for a in analysis.usable_annotations(q["annotations"]):
            row = analysis._row(q, models, sorted(evidence_spans(a)))
            for m in models:
                assert row["rank"][m] >= union[q["question_id"]]["rank"][m]


@needs_results
def test_stored_ranks_are_consistent():
    analysis, questions, models = _load()
    for q in questions:
        for m in models:
            ranks, source = q["ranks"][m], q["rank_source"][m]
            assert source in {"recomputed", "stored_rank_reference_invariant"}
            if source == "recomputed":
                assert ranks["union"] == min(ranks["spans"])
                assert len(ranks["spans"]) == len(q["spans"])
            else:
                assert ranks["spans"] is None and q["reference_invariant"]


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
    """The tracked files hold ids, character ranges, counts and ranks only."""
    data = json.loads((RESULTS / "per_question.json").read_text(encoding="utf-8"))
    allowed_question = {"question_id", "paper_id", "doc_id", "D", "annotations", "spans",
                        "reference_invariant", "ranks", "rank_source"}
    allowed_annotation = {"annotator", "status", "n_evidence", "unresolved_float", "unresolved_unmatched", "spans"}
    allowed_span = {"start_char", "end_char", "annotators", "gold_chunks"}
    for q in data["questions"]:
        assert set(q) <= allowed_question
        for a in q["annotations"]:
            assert set(a) <= allowed_annotation
        for s in q["spans"]:
            assert set(s) <= allowed_span
