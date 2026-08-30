"""End-to-end integration across the whole research path (W5).

    raw dataset file -> loader -> schema -> corpus -> chunking -> vector store
    -> retrieval -> generation -> inference records -> evidence alignment
    -> failure taxonomy -> aggregation

These tests use real loaders reading real dataset files written to a temp
directory in each dataset's genuine native format, a real chunker, a real
ChromaDB store and real retrieval. Only the generator is a deterministic stand
-in, because a language model would make the test non-deterministic and
network-dependent; every stage whose correctness this project depends on is
exercised for real.

The point is to catch the class of defect that unit tests structurally cannot:
an id that does not survive ingestion, an offset that does not survive the
vector store, a gold span that cannot be matched to the chunk that carries it.
"""
import json

import pytest

from src.data.corpus import build_corpus, chunk_documents
from src.data.loaders import get_loader
from src.data.schema import index_documents, validate_dataset
from src.evaluation.evidence import EvidenceStatus
from src.evaluation.records import write_records
from src.evaluation.runner import aggregate, run_inference, score_records
from src.evaluation.taxonomy import TaxonomyConfig
from src.rag.chunking import DocumentChunker, _WordTokenizer
from src.rag.mock_llm import MockExtractiveLLM
from src.rag.pipeline import RAGPipeline
from src.rag.providers import HashEmbeddings
from src.rag.vector_store import VectorStore

PARA_METHOD = (
    "We fine-tune a BERT-base encoder on the training split for three epochs "
    "using the Adam optimiser with a learning rate of 3e-5."
)
PARA_RESULTS = (
    "Our system reaches an F1 score of 88.2 on the held-out test set, "
    "outperforming the previous best result of 85.7 reported by prior work."
)
PARA_RELATED = (
    "Earlier approaches relied on hand-engineered features and conditional "
    "random fields rather than pretrained transformer encoders."
)


@pytest.fixture
def qasper_file(tmp_path):
    """A real QASPER-format file: JSON keyed by paper id."""
    payload = {
        "1901.00001": {
            "title": "A Study of Encoders",
            "abstract": "We study transformer encoders for extraction tasks.",
            "full_text": [
                {"section_name": "Method", "paragraphs": [PARA_METHOD]},
                {"section_name": "Results", "paragraphs": [PARA_RESULTS]},
                {"section_name": "Related Work", "paragraphs": [PARA_RELATED]},
            ],
            "qas": [
                {
                    "question": "What F1 score does the system reach on the test set?",
                    "question_id": "q_f1",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["88.2"],
                                "yes_no": None,
                                "free_form_answer": "",
                                "evidence": [PARA_RESULTS],
                            }
                        }
                    ],
                },
                {
                    "question": "Which optimiser is used for fine-tuning?",
                    "question_id": "q_opt",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["Adam"],
                                "yes_no": None,
                                "free_form_answer": "",
                                "evidence": [PARA_METHOD],
                            }
                        }
                    ],
                },
                {
                    "question": "What was the total compute budget in GPU hours?",
                    "question_id": "q_budget",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": True,
                                "extractive_spans": [],
                                "yes_no": None,
                                "free_form_answer": "",
                                "evidence": [],
                            }
                        }
                    ],
                },
            ],
        }
    }
    path = tmp_path / "qasper-sample.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def offline_chunker(chunk_size=40, chunk_overlap=5):
    """Chunker pinned to the offline tokenizer so the test needs no network."""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker.encoder = _WordTokenizer()
    return chunker


def run_pipeline(questions, documents, tmp_path, name, top_k=3):
    """Index a corpus and run every question through real retrieval."""
    store = VectorStore(
        HashEmbeddings(),
        persist_dir=str(tmp_path / f"chroma_{name}"),
        collection_name=f"integration_{name}",
    )
    stats = build_corpus(documents, store, offline_chunker(), reset=True)
    pipeline = RAGPipeline(vector_store=store, llm=MockExtractiveLLM())
    records = run_inference(
        [q.to_experiment_item() for q in questions],
        pipeline,
        top_k=top_k,
        doc_chunk_counts=store.doc_chunk_counts(),
    )
    return records, stats, store


class TestQasperEndToEnd:
    def test_full_path_from_raw_file_to_aggregate(self, qasper_file, tmp_path):
        result = get_loader("qasper").load(qasper_file, split="dev")
        assert result.questions, "loader produced nothing from a valid QASPER file"

        # Spans must resolve against the documents the loader itself built.
        documents = index_documents(result.documents)
        assert validate_dataset(result.questions, documents) == {}

        records, stats, _ = run_pipeline(result.questions, result.documents, tmp_path, "qasper")
        assert stats.offset_mismatches == 0
        assert len(records) == len(result.questions)

        rows = score_records(records, TaxonomyConfig())
        report = aggregate(rows)

        assert report["total"] == len(rows)
        assert "evidence" in report
        assert report["evidence"]["n_with_gold_evidence"] >= 2

    def test_gold_evidence_is_actually_found_by_retrieval(self, qasper_file, tmp_path):
        """The end-to-end claim: a gold span maps onto the chunk carrying it."""
        result = get_loader("qasper").load(qasper_file, split="dev")
        records, _, _ = run_pipeline(result.questions, result.documents, tmp_path, "qfind")
        rows = score_records(records, TaxonomyConfig())

        answerable = [r for r in rows if r.n_gold_spans > 0]
        assert answerable
        # With a three-paragraph paper and top_k=3 the evidence is reachable,
        # so at least one question must have its evidence located exactly.
        assert any(r.evidence_status == str(EvidenceStatus.COMPLETE) for r in answerable)

    def test_unanswerable_question_survives_the_whole_path(self, qasper_file, tmp_path):
        result = get_loader("qasper").load(qasper_file, split="dev")
        unanswerable = [q for q in result.questions if not q.is_answerable]
        assert len(unanswerable) == 1

        records, _, _ = run_pipeline(result.questions, result.documents, tmp_path, "qunans")
        rows = score_records(records, TaxonomyConfig())
        target = [r for r in rows if "compute budget" in r.question][0]

        assert target.is_answerable is False
        assert target.n_gold_spans == 0
        assert target.evidence_status == str(EvidenceStatus.NOT_APPLICABLE)
        # It either abstained correctly or failed to abstain; both are handled,
        # and neither may be charged to retrieval.
        assert target.attribution_stage in {"none", "abstention"}

    def test_ids_and_offsets_survive_ingestion(self, qasper_file, tmp_path):
        """Namespaced ids and character offsets must reach the records intact."""
        result = get_loader("qasper").load(qasper_file, split="dev")
        documents = index_documents(result.documents)
        records, _, _ = run_pipeline(result.questions, result.documents, tmp_path, "qids")

        for record in records:
            for chunk in record.retrieved:
                assert chunk.doc_id.startswith("qasper:")
                assert chunk.start_char is not None and chunk.end_char is not None
                source_text = documents[chunk.doc_id].text
                assert source_text[chunk.start_char : chunk.end_char] == chunk.text

    def test_records_round_trip_through_disk(self, qasper_file, tmp_path):
        """Scoring a reloaded run must reproduce the labels exactly."""
        from src.evaluation.records import read_records

        result = get_loader("qasper").load(qasper_file, split="dev")
        records, _, _ = run_pipeline(result.questions, result.documents, tmp_path, "qround")

        path = write_records(records, tmp_path / "inference.jsonl")
        reloaded = read_records(path)

        original = score_records(records, TaxonomyConfig())
        restored = score_records(reloaded, TaxonomyConfig())
        assert [r.failure_mode_v2 for r in restored] == [r.failure_mode_v2 for r in original]
        assert [r.evidence_status for r in restored] == [r.evidence_status for r in original]
        assert [r.attribution_stage for r in restored] == [r.attribution_stage for r in original]


class TestCorpusBridge:
    def test_offset_verification_rejects_a_corrupt_corpus(self, tmp_path):
        """build_corpus must refuse to index a corpus whose offsets are wrong."""
        from src.data.schema import Document

        class BadChunker(DocumentChunker):
            def chunk_text(self, text, doc_id, source):
                chunks = super().chunk_text(text, doc_id, source)
                for chunk in chunks:
                    chunk.metadata["start_char"] = 0
                    chunk.metadata["end_char"] = 1  # deliberately wrong
                return chunks

        bad = BadChunker(chunk_size=20, chunk_overlap=0)
        bad.encoder = _WordTokenizer()
        store = VectorStore(
            HashEmbeddings(),
            persist_dir=str(tmp_path / "bad"),
            collection_name="integration_bad",
        )
        documents = [Document(doc_id="d1", text="alpha beta gamma delta epsilon", source="d.md")]

        with pytest.raises(ValueError, match="do not slice back to their source"):
            build_corpus(documents, store, bad, reset=True)

    def test_chunk_documents_reports_per_document_counts(self, tmp_path):
        from src.data.schema import Document

        documents = [
            Document(doc_id="a", text="one two three four five six", source="a.md"),
            Document(doc_id="b", text="seven eight nine", source="b.md"),
        ]
        chunks, stats = chunk_documents(documents, offline_chunker(chunk_size=3, chunk_overlap=0))
        assert stats.n_documents == 2
        assert stats.n_chunks == len(chunks)
        assert stats.offset_mismatches == 0
        assert set(stats.chunks_per_document) == {"a", "b"}
        assert stats.total_characters == sum(len(d.text) for d in documents)
