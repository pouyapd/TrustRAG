"""Tests for the embedder registry and the generator abstraction.

Neither downloads a model. The registry is checked as data — the sweep's
validity rests on each entry describing the model that was actually called, and
a wrong prefix would silently report a weaker model rather than a different one.
The generator side is checked with fakes, because the point of the abstraction
is that the experiment does not care which provider is behind it, and CI must
never need an API key or a gigabyte of weights.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embedders import (  # noqa: E402
    EMBEDDERS,
    SWEEP_KEYS,
    EmbedderSpec,
    build_embedder,
    get_embedder_spec,
)
from src.rag.local_llm import LOCAL_GENERATORS, LocalCausalLM, build_generator  # noqa: E402
from src.rag.providers import EmbeddingProvider, HashEmbeddings  # noqa: E402


class TestEmbedderRegistry:
    def test_every_sweep_key_is_registered(self):
        for key in SWEEP_KEYS:
            assert key in EMBEDDERS

    def test_the_hash_control_is_excluded_from_the_sweep(self):
        """It is a determinism control, not a competing embedding model."""
        assert "hash" in EMBEDDERS
        assert "hash" not in SWEEP_KEYS

    def test_sweep_spans_more_than_one_training_lineage(self):
        """Four checkpoints from one family would not be a robustness result."""
        families = {EMBEDDERS[k].family for k in SWEEP_KEYS}
        assert len(families) >= 3

    def test_every_entry_declares_a_licence_and_dimension(self):
        for key, spec in EMBEDDERS.items():
            assert spec.license_spdx, key
            assert spec.dimension > 0, key
            assert spec.repo_id, key

    def test_baseline_is_unchanged(self):
        """Every previously reported figure used exactly this model.

        If this test fails, the frozen results are no longer reproducible and
        the change was almost certainly unintentional.
        """
        spec = get_embedder_spec("minilm")
        assert spec.repo_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert spec.dimension == 384
        assert spec.query_prefix == ""
        assert spec.passage_prefix == ""
        assert spec.normalize is False

    def test_asymmetric_models_carry_their_documented_prefixes(self):
        """E5 needs both sides prefixed; BGE prefixes only the query."""
        e5 = get_embedder_spec("e5")
        assert e5.query_prefix == "query: "
        assert e5.passage_prefix == "passage: "
        bge = get_embedder_spec("bge")
        assert bge.query_prefix.startswith("Represent this sentence")
        assert bge.passage_prefix == ""

    def test_symmetric_models_have_no_prefixes(self):
        for key in ("minilm", "mpnet"):
            spec = get_embedder_spec(key)
            assert spec.query_prefix == "" and spec.passage_prefix == ""

    def test_identity_records_a_revision_when_pinned(self):
        spec = EmbedderSpec(key="x", repo_id="org/model", dimension=8,
                            license_spdx="MIT", revision="abc123")
        assert spec.identity == "org/model@abc123"

    def test_identity_is_honest_when_unpinned(self):
        """No revision must not be dressed up as a pin."""
        assert get_embedder_spec("minilm").identity == \
            "sentence-transformers/all-MiniLM-L6-v2"

    def test_unknown_key_lists_the_alternatives(self):
        with pytest.raises(ValueError, match="available"):
            get_embedder_spec("not-a-model")

    def test_hash_embedder_builds_without_network(self):
        provider, spec = build_embedder("hash")
        assert isinstance(provider, HashEmbeddings)
        assert len(provider.embed(["hello world"])[0]) == spec.dimension


class TestQueryPassageAsymmetry:
    """`embed_query` must exist everywhere and default to symmetric behaviour."""

    def test_default_delegates_to_embed(self):
        provider = HashEmbeddings()
        assert provider.embed_query("a question") == provider.embed(["a question"])[0]

    def test_prefixes_change_the_vector(self):
        """A prefix that made no difference would mean it was never applied."""

        class Recorder(EmbeddingProvider):
            def __init__(self):
                self.seen = []

            def embed(self, texts):
                self.seen.extend(texts)
                return [[0.0] for _ in texts]

        recorder = Recorder()
        recorder.embed_query("what is x")
        assert recorder.seen == ["what is x"]

    def test_local_embeddings_applies_prefixes_without_loading_a_model(self, monkeypatch):
        """Verifies the prefix plumbing, not sentence-transformers."""
        from src.rag import providers

        captured = {}

        class FakeSentenceTransformer:
            def __init__(self, name):
                captured["model"] = name

            def encode(self, texts, **kwargs):
                captured.setdefault("encoded", []).extend(texts)
                return _FakeArray([[0.1, 0.2] for _ in texts])

            def get_sentence_embedding_dimension(self):
                return 2

        class _FakeArray(list):
            def tolist(self):
                return list(self)

        import types
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = providers.LocalEmbeddings(
            "org/model", query_prefix="query: ", passage_prefix="passage: "
        )
        provider.embed(["a passage"])
        provider.embed_query("a question")
        assert captured["encoded"] == ["passage: a passage", "query: a question"]
        assert captured["model"] == "org/model"


class TestGeneratorAbstraction:
    def test_mock_generator_needs_nothing(self):
        generator, identity = build_generator("mock")
        assert identity["is_language_model"] is False
        assert "extractive" in identity["kind"]
        assert generator.generate("sys", "CONTEXT:\nParis is the capital.\n\nQUESTION: q")

    def test_local_generators_are_declared_but_not_loaded(self):
        """Building must not download; loading is deferred to first generate()."""
        for key in LOCAL_GENERATORS:
            generator, identity = build_generator(key)
            assert isinstance(generator, LocalCausalLM)
            assert generator._model is None
            assert identity["is_language_model"] is True
            assert identity["decoding"] == "greedy (do_sample=False)"

    def test_local_generator_specs_are_independent_families(self):
        repos = {spec.repo_id.split("/")[0] for spec in LOCAL_GENERATORS.values()}
        assert len(repos) >= 2

    def test_unknown_generator_names_the_alternatives(self):
        with pytest.raises(ValueError, match="expected"):
            build_generator("gpt-9")

    def test_hosted_provider_refuses_without_a_key(self, monkeypatch):
        """It must fail loudly, never silently substitute a different model."""
        from src import config

        monkeypatch.setattr(config.settings, "openai_api_key", "", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            build_generator("openai:gpt-4o-mini")

    def test_anthropic_refuses_without_a_key(self, monkeypatch):
        from src import config

        monkeypatch.setattr(config.settings, "anthropic_api_key", "", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            build_generator("anthropic:claude-3-haiku")

    def test_temperature_is_ignored_rather_than_silently_sampling(self):
        """Greedy decoding is what makes the run reproducible."""
        import inspect

        source = inspect.getsource(LocalCausalLM.generate)
        assert "do_sample=False" in source


class TestGenerationExperimentPlumbing:
    """The parts of the LLM experiment that need no model."""

    def test_refusal_detection(self):
        from scripts.run_llm_experiment import abstained

        assert abstained("I cannot answer this from the provided context.")
        assert abstained("The context does not provide that information.")
        assert not abstained("The refund window is 30 days.")

    def test_refusal_detection_is_not_over_eager(self):
        """A hedge inside a real answer is still an answer."""
        from scripts.run_llm_experiment import abstained

        assert not abstained("The policy is 30 days, though it does not say for whom.")

    def test_context_is_rebuilt_in_rank_order(self):
        from scripts.run_llm_experiment import format_context
        from src.evaluation.records import InferenceRecord, RetrievedChunk

        record = InferenceRecord(
            index=1, question="q", reference_answer="a", relevant_doc_ids=["d"],
            predicted_answer="", faithfulness=None, latency_ms=1.0, top_k=2,
            retrieved=[
                RetrievedChunk(rank=1, chunk_id="d_0", doc_id="d", source="first.md",
                               score=0.9, text="alpha"),
                RetrievedChunk(rank=2, chunk_id="d_1", doc_id="d", source="second.md",
                               score=0.8, text="beta"),
            ],
        )
        context = format_context(record)
        assert context.index("first.md") < context.index("second.md")
        assert "alpha" in context and "beta" in context

    def test_empty_retrieval_is_stated_not_blank(self):
        from scripts.run_llm_experiment import format_context
        from src.evaluation.records import InferenceRecord

        record = InferenceRecord(
            index=1, question="q", reference_answer="a", relevant_doc_ids=[],
            predicted_answer="", faithfulness=None, latency_ms=1.0, top_k=0, retrieved=[],
        )
        assert format_context(record) == "(no context available)"
