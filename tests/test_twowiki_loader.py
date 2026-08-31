"""Unit tests for the 2WikiMultihopQA loader.

The dataset is the replication target for the quantifier effect, so the loader
has to be right in the ways that would silently corrupt that result rather than
crash: dropping multi-hop items down to single-hop, mis-resolving a supporting
sentence to the wrong character range, or letting distractor paragraphs fall
out of the corpus and make retrieval trivially easy.

Fixtures are built in the distribution's genuine native shape — nested fields
JSON-encoded as *strings*, context as `[title, [sentences]]` pairs — so the
same parsing path the real file takes is exercised without needing the file.
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.loaders.twowiki_parquet import TwoWikiMultihopParquetLoader, _maybe_json
from src.data.schema import index_documents, validate_dataset


def write_parquet(rows: list[dict], path):
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def twowiki_row(
    item_id="a1",
    question="Who is the mother of the director of film X?",
    answer="Jane Doe",
    context=None,
    supporting_facts=None,
    kind="compositional",
    evidences=None,
    as_json=True,
):
    """One row in the shape the HuggingFace parquet actually has."""
    context = context if context is not None else [
        ["Film X", ["Film X is a 2009 film directed by John Roe.", " It was shot in Poland."]],
        ["John Roe", ["John Roe is a director.", " He is the son of actress Jane Doe."]],
        ["Distractor", ["Something entirely unrelated."]],
    ]
    supporting_facts = supporting_facts if supporting_facts is not None else [
        ["Film X", 0], ["John Roe", 1],
    ]
    evidences = evidences if evidences is not None else [
        ["Film X", "director", "John Roe"], ["John Roe", "mother", "Jane Doe"],
    ]
    enc = json.dumps if as_json else (lambda v: v)
    return {
        "_id": item_id,
        "type": kind,
        "question": question,
        "answer": answer,
        "context": enc(context),
        "supporting_facts": enc(supporting_facts),
        "evidences": enc(evidences),
    }


class TestJsonStringFields:
    """This mirror stores nested fields as strings; both forms must work."""

    def test_json_string_is_parsed(self):
        assert _maybe_json('[["a", 1]]') == [["a", 1]]

    def test_native_list_passes_through(self):
        assert _maybe_json([["a", 1]]) == [["a", 1]]

    def test_malformed_json_is_not_fatal(self):
        assert _maybe_json("{not json") == []

    def test_row_with_native_lists_builds_the_same_record(self):
        """A distribution that kept real nested values must not break the loader.

        Exercised through `_build` rather than through a parquet file: the
        `[title, [sentences]]` shape mixes a string and a list in one position,
        which Arrow cannot represent as a uniform type. That is precisely why
        this distribution encodes the field as a JSON string, so the native form
        can only reach the loader in memory.
        """
        loader = TwoWikiMultihopParquetLoader()
        from_json = loader._build(twowiki_row(), {}, "dev", __import__("collections").Counter())
        from_native = loader._build(
            twowiki_row(as_json=False), {}, "dev", __import__("collections").Counter()
        )
        assert from_native is not None
        assert from_native.question == from_json.question
        assert from_native.hops == from_json.hops
        assert [s.text for s in from_native.supporting_spans] == [
            s.text for s in from_json.supporting_spans
        ]


class TestMultiHopStructure:
    def test_item_is_all_required(self, tmp_path):
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        result = TwoWikiMultihopParquetLoader().load(path)
        question = result.questions[0]
        assert str(question.evidence_mode) == "all_required"
        assert question.hops == 2
        assert len(question.metadata["gold_doc_ids"]) == 2

    def test_distractors_are_kept_in_the_corpus(self, tmp_path):
        """Removing them would make retrieval trivial and inflate every rate."""
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        result = TwoWikiMultihopParquetLoader().load(path)
        assert len(result.documents) == 3
        assert len(result.questions[0].metadata["distractor_doc_ids"]) == 1

    def test_four_hop_item_requires_four_documents(self, tmp_path):
        context = [[f"D{i}", [f"Sentence about entity {i}."]] for i in range(4)]
        facts = [[f"D{i}", 0] for i in range(4)]
        path = write_parquet(
            [twowiki_row(context=context, supporting_facts=facts, kind="bridge_comparison")],
            tmp_path / "t.parquet",
        )
        question = TwoWikiMultihopParquetLoader().load(path).questions[0]
        assert question.hops == 4
        assert len(question.metadata["gold_doc_ids"]) == 4

    def test_single_document_evidence_is_filtered_by_default(self, tmp_path):
        path = write_parquet(
            [twowiki_row(supporting_facts=[["Film X", 0], ["Film X", 1]])],
            tmp_path / "t.parquet",
        )
        result = TwoWikiMultihopParquetLoader().load(path)
        assert result.questions == []
        assert result.skipped["single_document_evidence"] == 1

    def test_single_hop_kept_when_filter_disabled(self, tmp_path):
        path = write_parquet(
            [twowiki_row(supporting_facts=[["Film X", 0]])], tmp_path / "t.parquet"
        )
        result = TwoWikiMultihopParquetLoader(multi_hop_only=False).load(path)
        assert str(result.questions[0].evidence_mode) == "any_sufficient"


class TestSpanResolution:
    def test_spans_resolve_exactly_against_the_document(self, tmp_path):
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        result = TwoWikiMultihopParquetLoader().load(path)
        assert validate_dataset(result.questions, index_documents(result.documents)) == {}

    def test_span_text_matches_the_named_sentence(self, tmp_path):
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        result = TwoWikiMultihopParquetLoader().load(path)
        docs = index_documents(result.documents)
        texts = {s.text for s in result.questions[0].supporting_spans}
        assert "Film X is a 2009 film directed by John Roe." in texts
        assert " He is the son of actress Jane Doe." in texts
        for span in result.questions[0].supporting_spans:
            assert docs[span.doc_id].text[span.start_char : span.end_char] == span.text

    def test_out_of_range_sentence_index_is_counted_not_crashed(self, tmp_path):
        path = write_parquet(
            [twowiki_row(supporting_facts=[["Film X", 0], ["John Roe", 99]])],
            tmp_path / "t.parquet",
        )
        result = TwoWikiMultihopParquetLoader().load(path)
        assert result.skipped["supporting_fact_out_of_range"] == 1

    def test_supporting_fact_naming_absent_title_is_counted(self, tmp_path):
        path = write_parquet(
            [twowiki_row(supporting_facts=[["Film X", 0], ["Nowhere", 0]])],
            tmp_path / "t.parquet",
        )
        result = TwoWikiMultihopParquetLoader().load(path)
        assert result.skipped["supporting_fact_without_context"] == 1


class TestRecordFields:
    def test_answer_and_type_are_recorded(self, tmp_path):
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        question = TwoWikiMultihopParquetLoader().load(path).questions[0]
        assert question.answers == ["Jane Doe"]
        assert question.metadata["twowiki_type"] == "compositional"
        assert question.metadata["n_evidence_triples"] == 2

    def test_question_ids_are_deterministic(self, tmp_path):
        path = write_parquet([twowiki_row()], tmp_path / "t.parquet")
        first = TwoWikiMultihopParquetLoader().load(path).questions[0].question_id
        second = TwoWikiMultihopParquetLoader().load(path).questions[0].question_id
        assert first == second

    def test_limit_is_respected(self, tmp_path):
        rows = [twowiki_row(item_id=f"a{i}") for i in range(5)]
        path = write_parquet(rows, tmp_path / "t.parquet")
        assert len(TwoWikiMultihopParquetLoader().load(path, limit=2).questions) == 2

    def test_empty_question_and_answer_are_skipped(self, tmp_path):
        rows = [twowiki_row(item_id="a", question=""), twowiki_row(item_id="b", answer="")]
        path = write_parquet(rows, tmp_path / "t.parquet")
        result = TwoWikiMultihopParquetLoader().load(path)
        assert result.questions == []
        assert result.skipped["question_empty"] == 1
        assert result.skipped["answer_empty"] == 1

    def test_licence_and_source_are_declared(self):
        loader = TwoWikiMultihopParquetLoader()
        assert loader.license_spdx == "Apache-2.0"
        assert "2WikiMultihopQA" in loader.source_url


class TestRegistry:
    def test_loader_is_registered_under_2wiki(self):
        from src.data.loaders import available_loaders, get_loader

        assert "2wiki" in available_loaders()
        assert isinstance(get_loader("2wiki"), TwoWikiMultihopParquetLoader)

    def test_unknown_name_still_lists_alternatives(self):
        from src.data.loaders import get_loader

        with pytest.raises(KeyError, match="available"):
            get_loader("2wikimultihop")
