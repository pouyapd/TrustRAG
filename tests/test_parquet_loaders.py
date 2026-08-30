"""Unit tests for the parquet dataset loaders.

These loaders were originally validated only by running them over the real
corpora, which catches gross failures but not quiet ones. It missed a real
defect: the HuggingFace parquet distribution encodes `yes_no_answer` as a
ClassLabel index, not the strings the older JSON release used, so every yes/no
question was silently dropped and a 300-question run reported zero of them.
Then the obvious integer mapping was the wrong way round.

The tests build small parquet files in the datasets' genuine native shapes, so
they exercise the same code path the real data does without needing the data.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.loaders.hotpot_parquet import HotpotQaParquetLoader, _join_sentences
from src.data.loaders.nq_parquet import (
    NaturalQuestionsParquetLoader,
    _clean_text_with_spans,
    _tokens_to_char_span,
    _yes_no_label,
)
from src.data.schema import index_documents, validate_dataset


def write_parquet(rows: list[dict], path):
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def nq_row(example_id, question, tokens, is_html, long_answer, short_answers=None,
           yes_no=None, url="http://wiki/Page", title="Page"):
    """One row in the shape the HuggingFace NQ parquet actually has."""
    return {
        "id": example_id,
        "document": {
            "html": "", "title": title, "url": url,
            "tokens": {
                "token": tokens, "is_html": is_html,
                "start_byte": [0] * len(tokens), "end_byte": [0] * len(tokens),
            },
        },
        "question": {"text": question, "tokens": question.split()},
        "long_answer_candidates": {
            "start_token": [], "end_token": [], "start_byte": [],
            "end_byte": [], "top_level": [],
        },
        "annotations": {
            "id": ["0"],
            "long_answer": [long_answer],
            "short_answers": [short_answers or {
                "start_token": [], "end_token": [], "start_byte": [],
                "end_byte": [], "text": [],
            }],
            "yes_no_answer": yes_no if yes_no is not None else [-1],
        },
    }


NO_LONG_ANSWER = {"candidate_index": -1, "start_token": -1, "end_token": -1,
                  "start_byte": -1, "end_byte": -1}


class TestYesNoEncoding:
    """The defect that motivated this file."""

    def test_classlabel_indices_map_per_the_schema(self):
        """Schema metadata says names = ["NO", "YES"], so 0 is No and 1 is Yes."""
        assert _yes_no_label(0) == "No"
        assert _yes_no_label(1) == "Yes"

    def test_sentinel_means_no_annotation(self):
        assert _yes_no_label(-1) is None

    def test_string_form_from_older_releases_still_works(self):
        assert _yes_no_label("YES") == "Yes"
        assert _yes_no_label("NO") == "No"
        assert _yes_no_label("NONE") is None

    def test_booleans_are_accepted(self):
        assert _yes_no_label(True) == "Yes"
        assert _yes_no_label(False) == "No"

    def test_a_yes_no_question_survives_loading(self, tmp_path):
        tokens = ["Ebola", "killed", "two", "people", "in", "the", "US", "."]
        row = nq_row(
            "1", "did anyone die from ebola in the us", tokens, [False] * len(tokens),
            {"candidate_index": 0, "start_token": 0, "end_token": 8,
             "start_byte": -1, "end_byte": -1},
            yes_no=[1, 1, -1],
        )
        path = write_parquet([row], tmp_path / "nq.parquet")
        result = NaturalQuestionsParquetLoader().load(path)
        assert len(result.questions) == 1
        assert result.questions[0].answers == ["Yes"]
        assert str(result.questions[0].question_type) == "yes_no"


class TestNqTokenMapping:
    def test_html_tokens_are_dropped_from_the_text(self):
        text, spans = _clean_text_with_spans(["<P>", "Hello", "world"], [True, False, False])
        assert text == "Hello world"
        assert spans[0] is None
        assert text[spans[1][0] : spans[1][1]] == "Hello"
        assert text[spans[2][0] : spans[2][1]] == "world"

    def test_offsets_survive_dropped_tokens(self):
        """The point of the index map: annotation offsets must still resolve."""
        tokens = ["<Table>", "The", "answer", "</Table>", "is", "42"]
        is_html = [True, False, False, True, False, False]
        text, spans = _clean_text_with_spans(tokens, is_html)
        assert text == "The answer is 42"
        # Original token range [1,3) is "The answer" even though token 0 is gone.
        span = _tokens_to_char_span((1, 3), spans)
        assert text[span[0] : span[1]] == "The answer"

    def test_range_of_pure_markup_has_no_char_span(self):
        _, spans = _clean_text_with_spans(["<P>", "</P>"], [True, True])
        assert _tokens_to_char_span((0, 2), spans) is None


class TestNqLoading:
    def _answerable_row(self, example_id, url, tokens=None):
        tokens = tokens or ["Paris", "is", "the", "capital", "of", "France", "."]
        return nq_row(
            example_id, "what is the capital of france", tokens, [False] * len(tokens),
            {"candidate_index": 0, "start_token": 0, "end_token": len(tokens),
             "start_byte": -1, "end_byte": -1},
            short_answers={"start_token": [0], "end_token": [1], "start_byte": [],
                           "end_byte": [], "text": []},
            url=url,
        )

    def test_spans_resolve_against_the_built_document(self, tmp_path):
        path = write_parquet([self._answerable_row("1", "http://wiki/France")],
                             tmp_path / "nq.parquet")
        result = NaturalQuestionsParquetLoader().load(path)
        assert validate_dataset(result.questions, index_documents(result.documents)) == {}

    def test_page_scoped_null_is_skipped_not_marked_unanswerable(self, tmp_path):
        """"No answer on this page" is not "the corpus cannot answer this"."""
        tokens = ["Some", "unrelated", "text", "."]
        row = nq_row("1", "an unanswered question", tokens, [False] * len(tokens),
                     NO_LONG_ANSWER)
        path = write_parquet([row], tmp_path / "nq.parquet")
        result = NaturalQuestionsParquetLoader().load(path)
        assert result.questions == []
        assert result.skipped["no_long_answer_page_scoped_null"] == 1
        # Crucially: it did NOT become an unanswerable question.
        assert result.counts["unanswerable"] == 0

    def test_two_questions_on_one_page_share_one_document(self, tmp_path):
        rows = [self._answerable_row("1", "http://wiki/France"),
                self._answerable_row("2", "http://wiki/France")]
        path = write_parquet(rows, tmp_path / "nq.parquet")
        result = NaturalQuestionsParquetLoader().load(path)
        assert len(result.questions) == 2
        assert len(result.documents) == 1, "identical pages must deduplicate"

    def test_different_pages_stay_separate(self, tmp_path):
        rows = [
            self._answerable_row("1", "http://wiki/A", ["Alpha", "is", "first", "."]),
            self._answerable_row("2", "http://wiki/B", ["Beta", "is", "second", "."]),
        ]
        path = write_parquet(rows, tmp_path / "nq.parquet")
        result = NaturalQuestionsParquetLoader().load(path)
        assert len(result.documents) == 2

    def test_limit_is_respected(self, tmp_path):
        rows = [self._answerable_row(str(i), f"http://wiki/{i}") for i in range(5)]
        path = write_parquet(rows, tmp_path / "nq.parquet")
        assert len(NaturalQuestionsParquetLoader().load(path, limit=2).questions) == 2

    def test_question_ids_are_deterministic_across_loads(self, tmp_path):
        path = write_parquet([self._answerable_row("1", "http://wiki/France")],
                             tmp_path / "nq.parquet")
        first = NaturalQuestionsParquetLoader().load(path).questions[0].question_id
        second = NaturalQuestionsParquetLoader().load(path).questions[0].question_id
        assert first == second


def hotpot_row(item_id, question, answer, titles, paragraphs, fact_titles, fact_ids,
               kind="bridge", level="hard"):
    return {
        "id": item_id, "question": question, "answer": answer,
        "type": kind, "level": level,
        "supporting_facts": {"title": fact_titles, "sent_id": fact_ids},
        "context": {"title": titles, "sentences": paragraphs},
    }


class TestHotpotLoading:
    def _two_hop(self):
        return hotpot_row(
            "h1", "Were A and B both directors?", "yes",
            ["Alpha", "Beta", "Distractor"],
            [["Alpha is a director.", " Alpha was born in 1950."],
             ["Beta is a director.", " Beta was born in 1960."],
             ["Something unrelated entirely."]],
            ["Alpha", "Beta"], [0, 0],
        )

    def test_multi_hop_item_is_all_required(self, tmp_path):
        path = write_parquet([self._two_hop()], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader().load(path)
        assert len(result.questions) == 1
        question = result.questions[0]
        assert str(question.evidence_mode) == "all_required"
        assert question.hops == 2
        assert len(question.relevant_doc_ids) == 2

    def test_spans_resolve_and_distractors_are_kept(self, tmp_path):
        path = write_parquet([self._two_hop()], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader().load(path)
        documents = index_documents(result.documents)
        assert validate_dataset(result.questions, documents) == {}
        # The distractor paragraph stays: it is what makes retrieval non-trivial.
        assert len(result.documents) == 3

    def test_single_hop_item_is_filtered_by_default(self, tmp_path):
        row = hotpot_row("h2", "Who is Alpha?", "a director",
                         ["Alpha", "Distractor"],
                         [["Alpha is a director."], ["Unrelated."]],
                         ["Alpha"], [0])
        path = write_parquet([row], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader().load(path)
        assert result.questions == []
        assert result.skipped["single_document_evidence"] == 1

    def test_single_hop_kept_when_filter_disabled(self, tmp_path):
        row = hotpot_row("h2", "Who is Alpha?", "a director",
                         ["Alpha", "Distractor"],
                         [["Alpha is a director."], ["Unrelated."]],
                         ["Alpha"], [0])
        path = write_parquet([row], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader(multi_hop_only=False).load(path)
        assert len(result.questions) == 1
        assert str(result.questions[0].evidence_mode) == "any_sufficient"

    def test_out_of_range_supporting_fact_is_counted(self, tmp_path):
        row = hotpot_row("h3", "q", "a", ["Alpha", "Beta"],
                         [["One sentence."], ["Another."]], ["Alpha", "Beta"], [99, 0])
        path = write_parquet([row], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader().load(path)
        assert result.skipped["supporting_fact_out_of_range"] == 1

    def test_sentences_join_verbatim_so_offsets_stay_exact(self):
        text, offsets = _join_sentences(["First.", " Second.", " Third."])
        assert text == "First. Second. Third."
        for start, end in offsets:
            assert text[start:end] in {"First.", " Second.", " Third."}

    def test_answer_is_kept_as_the_reference(self, tmp_path):
        path = write_parquet([self._two_hop()], tmp_path / "h.parquet")
        result = HotpotQaParquetLoader().load(path)
        assert result.questions[0].answers == ["yes"]


class TestRegistry:
    def test_both_parquet_loaders_are_registered(self):
        from src.data.loaders import available_loaders, get_loader

        assert "nq" in available_loaders()
        assert "hotpotqa" in available_loaders()
        assert isinstance(get_loader("nq"), NaturalQuestionsParquetLoader)
        assert isinstance(get_loader("hotpotqa"), HotpotQaParquetLoader)

    def test_unknown_loader_names_the_alternatives(self):
        from src.data.loaders import get_loader

        with pytest.raises(KeyError, match="available"):
            get_loader("does_not_exist")
