"""Tests for character offsets on chunks (W1).

The single invariant everything else rests on:

    document_text[chunk.start_char:chunk.end_char] == chunk.text

It must hold for every chunk, on every tokenizer path, including documents that
repeat themselves and documents with irregular whitespace. If it ever fails,
gold evidence spans cannot be mapped onto retrieved chunks and every span-level
retrieval metric built on top would be silently wrong.
"""
import pytest

from src.rag.chunking import (
    Chunk,
    DocumentChunker,
    _WordTokenizer,
    token_char_spans,
)

SIMPLE = "Hello world. This is a test."
REPEATED = "The model was trained. The model was trained. The model was trained."
WHITESPACE = "Alpha   beta\t\tgamma\n\n\nDelta  \t epsilon\r\n\r\nZeta"


def word_chunker(chunk_size: int, chunk_overlap: int = 0) -> DocumentChunker:
    """Chunker pinned to the offline word tokenizer."""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker.encoder = _WordTokenizer()
    return chunker


def tiktoken_chunker(chunk_size: int, chunk_overlap: int = 0) -> DocumentChunker:
    """Chunker pinned to tiktoken, skipping the test when it is unavailable."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        encoder.encode("probe")
    except Exception as e:  # pragma: no cover - environment without BPE files
        pytest.skip(f"tiktoken encoding unavailable: {e}")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker.encoder = encoder
    return chunker


#: Both supported tokenizer paths, used wherever a test must cover TEST 4.
BUILDERS = [
    pytest.param(word_chunker, id="word-fallback"),
    pytest.param(tiktoken_chunker, id="tiktoken"),
]


def assert_offsets_exact(document: str, chunks: list[Chunk]) -> None:
    """The core invariant, plus the structural guarantees around it."""
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.start_char is not None
        assert chunk.end_char is not None
        # Half-open, in range, non-empty.
        assert 0 <= chunk.start_char < chunk.end_char <= len(document)
        # The invariant.
        assert document[chunk.start_char : chunk.end_char] == chunk.text


# ---------------------------------------------------------------
# TEST 1 — simple document
# ---------------------------------------------------------------

class TestSimpleDocument:
    @pytest.mark.parametrize("build", BUILDERS)
    def test_offsets_reproduce_chunk_text(self, build):
        chunks = build(4).chunk_text(SIMPLE, doc_id="d", source="d.md")
        assert_offsets_exact(SIMPLE, chunks)

    @pytest.mark.parametrize("build", BUILDERS)
    def test_first_chunk_starts_at_first_token(self, build):
        chunks = build(4).chunk_text(SIMPLE, doc_id="d", source="d.md")
        assert chunks[0].start_char == 0

    @pytest.mark.parametrize("build", BUILDERS)
    def test_last_chunk_ends_at_last_token(self, build):
        chunks = build(4).chunk_text(SIMPLE, doc_id="d", source="d.md")
        assert chunks[-1].end_char == len(SIMPLE.rstrip())

    def test_single_chunk_covers_whole_document(self):
        chunks = word_chunker(1000).chunk_text(SIMPLE, doc_id="d", source="d.md")
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(SIMPLE)
        assert chunks[0].text == SIMPLE


# ---------------------------------------------------------------
# TEST 2 — repeated text
# ---------------------------------------------------------------

class TestRepeatedText:
    """The case that makes `str.find()` recovery unsound."""

    @pytest.mark.parametrize("build", BUILDERS)
    def test_invariant_holds(self, build):
        chunks = build(4).chunk_text(REPEATED, doc_id="d", source="d.md")
        assert_offsets_exact(REPEATED, chunks)

    @pytest.mark.parametrize("build", BUILDERS)
    def test_identical_text_gets_distinct_offsets(self, build):
        chunks = build(4).chunk_text(REPEATED, doc_id="d", source="d.md")
        starts = [c.start_char for c in chunks]
        assert len(set(starts)) == len(starts), "chunks must not share a start offset"

    @pytest.mark.parametrize("build", BUILDERS)
    def test_offsets_are_strictly_increasing(self, build):
        chunks = build(4).chunk_text(REPEATED, doc_id="d", source="d.md")
        starts = [c.start_char for c in chunks]
        assert starts == sorted(starts)
        assert all(b > a for a, b in zip(starts, starts[1:], strict=False))

    def test_offsets_point_at_the_right_occurrence_not_the_first(self):
        """A later repetition must resolve to its own position.

        `REPEATED` contains the same sentence three times, so `find()` would
        return offset 0 for all three. Chunking at sentence granularity must
        produce three different starts, each slicing back to its own copy.
        """
        chunks = word_chunker(4).chunk_text(REPEATED, doc_id="d", source="d.md")
        assert len(chunks) == 3
        assert [c.text for c in chunks] == ["The model was trained."] * 3
        assert [c.start_char for c in chunks] == [0, 23, 46]
        # Each slice really is the copy at that position, not the first one.
        for chunk in chunks:
            assert REPEATED[chunk.start_char : chunk.end_char] == chunk.text
        assert REPEATED.find(chunks[2].text) == 0  # what find() would have said


# ---------------------------------------------------------------
# TEST 3 — whitespace-heavy input
# ---------------------------------------------------------------

class TestWhitespaceHeavy:
    @pytest.mark.parametrize("build", BUILDERS)
    def test_invariant_holds_with_irregular_whitespace(self, build):
        chunks = build(3).chunk_text(WHITESPACE, doc_id="d", source="d.md")
        assert_offsets_exact(WHITESPACE, chunks)

    def test_original_whitespace_is_preserved_in_chunk_text(self):
        """The offline path must not collapse runs of whitespace.

        `_WordTokenizer.decode` joins tokens with single spaces, which is why
        chunk text is sliced from the source instead of decoded.
        """
        chunks = word_chunker(3).chunk_text(WHITESPACE, doc_id="d", source="d.md")
        assert "   " in chunks[0].text or "\t" in chunks[0].text
        joined = _WordTokenizer.decode(_WordTokenizer.encode(WHITESPACE))
        assert chunks[0].text != joined[: len(chunks[0].text)]

    def test_leading_and_trailing_whitespace_is_excluded(self):
        document = "\n\n   Alpha beta gamma   \n\n"
        chunks = word_chunker(10).chunk_text(document, doc_id="d", source="d.md")
        assert chunks[0].text == "Alpha beta gamma"
        assert document[chunks[0].start_char : chunks[0].end_char] == "Alpha beta gamma"

    @pytest.mark.parametrize("build", BUILDERS)
    def test_multibyte_characters_stay_aligned(self, build):
        """Character offsets, not byte offsets."""
        document = "Café serves 日本茶 daily. Naïve café patrons agree."
        chunks = build(3).chunk_text(document, doc_id="d", source="d.md")
        assert_offsets_exact(document, chunks)


# ---------------------------------------------------------------
# TEST 4 — tokenizer paths and their span functions
# ---------------------------------------------------------------

class TestTokenizerPaths:
    def test_word_spans_match_str_split_boundaries(self):
        spans = _WordTokenizer.char_spans(WHITESPACE)
        assert [WHITESPACE[s:e] for s, e in spans] == WHITESPACE.split()

    def test_word_span_count_matches_encode(self):
        assert len(_WordTokenizer.char_spans(WHITESPACE)) == len(
            _WordTokenizer.encode(WHITESPACE)
        )

    def test_tiktoken_span_count_matches_token_count(self):
        chunker = tiktoken_chunker(4)
        spans = token_char_spans(SIMPLE, chunker.encoder)
        assert len(spans) == len(chunker.encoder.encode(SIMPLE))

    def test_tiktoken_spans_are_ordered_and_cover_the_text(self):
        chunker = tiktoken_chunker(4)
        spans = token_char_spans(SIMPLE, chunker.encoder)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(SIMPLE)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert next_start >= prev_end - 1  # boundaries may share a snapped char

    @pytest.mark.parametrize("build", BUILDERS)
    def test_chunk_count_is_unchanged_by_offset_tracking(self, build):
        """Token boundaries must not move: only how text is materialised changed."""
        chunker = build(4)
        spans = token_char_spans(SIMPLE, chunker.encoder)
        chunks = chunker.chunk_text(SIMPLE, doc_id="d", source="d.md")
        expected = -(-len(spans) // 4)  # ceil division, no overlap
        assert len(chunks) == expected

    def test_tokenizer_without_offsets_is_rejected_not_approximated(self):
        """An approximate mapping would silently misalign gold evidence."""

        class OpaqueTokenizer:
            def encode(self, text):
                return text.split()

            def decode(self, tokens):
                return " ".join(tokens)

        with pytest.raises(TypeError, match="cannot provide exact character offsets"):
            token_char_spans(SIMPLE, OpaqueTokenizer())


# ---------------------------------------------------------------
# TEST 5 — vector store round trip
# ---------------------------------------------------------------

class TestVectorStoreRoundTrip:
    """Round-trip tests.

    ChromaDB caches its client at process level, so a collection can still hold
    vectors written by an earlier test in the same session even though this
    fixture supplies a fresh directory. Each test therefore resets the
    collection on entry and on exit: it must neither read another test's
    vectors nor leave its own behind.
    """

    @staticmethod
    def _empty_store():
        from src.rag.providers import HashEmbeddings
        from src.rag.vector_store import VectorStore

        store = VectorStore(HashEmbeddings())
        store.reset()
        return store

    def test_offsets_survive_store_and_retrieve(self, tmp_chroma_dir):
        document = (
            "Annual subscribers have a 30-day refund window. "
            "Monthly plans are refundable for 14 days. "
            "Enterprise plans include a 99.9% uptime SLA."
        )
        chunks = word_chunker(6).chunk_text(document, doc_id="policy", source="policy.md")
        by_id = {c.chunk_id: c for c in chunks}

        store = self._empty_store()
        try:
            store.add(chunks)
            results = store.search("refund window", top_k=len(chunks))
            assert results
            assert {r.chunk_id for r in results} == set(by_id)

            for result in results:
                original = by_id[result.chunk_id]
                assert result.start_char == original.start_char
                assert result.end_char == original.end_char
                # Offsets still slice the source correctly after the round trip.
                assert document[result.start_char : result.end_char] == original.text
        finally:
            store.reset()

    def test_offsets_reach_the_inference_record(self, tmp_chroma_dir):
        """RetrievedChunk carries the offsets through to stored records."""
        from src.evaluation.records import RetrievedChunk

        document = "Alpha beta gamma delta epsilon zeta eta theta."
        chunks = word_chunker(3).chunk_text(document, doc_id="d", source="d.md")
        store = self._empty_store()
        try:
            store.add(chunks)
            result = store.search("gamma", top_k=1)[0]
        finally:
            store.reset()
        record_chunk = RetrievedChunk(
            rank=1,
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            source=result.source,
            score=result.score,
            text=result.text,
            start_char=result.start_char,
            end_char=result.end_char,
        )
        restored = RetrievedChunk.from_dict(record_chunk.as_dict())
        assert restored.start_char == result.start_char
        assert restored.end_char == result.end_char
        assert document[restored.start_char : restored.end_char] == restored.text

    def test_missing_offsets_read_as_none_not_zero(self):
        """Vectors written before W1 must not look like they start at char 0."""
        from src.evaluation.records import RetrievedChunk

        legacy = RetrievedChunk.from_dict(
            {"rank": 1, "chunk_id": "c", "doc_id": "d", "source": "s", "score": 0.5, "text": "t"}
        )
        assert legacy.start_char is None
        assert legacy.end_char is None


# ---------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------

class TestBackwardCompatibility:
    @pytest.mark.parametrize("build", BUILDERS)
    def test_existing_metadata_keys_are_preserved(self, build):
        chunks = build(4).chunk_text(SIMPLE, doc_id="d", source="d.md")
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == idx
            assert chunk.metadata["token_count"] > 0

    @pytest.mark.parametrize("build", BUILDERS)
    def test_chunk_id_scheme_is_unchanged(self, build):
        chunks = build(4).chunk_text(SIMPLE, doc_id="mydoc", source="d.md")
        assert [c.chunk_id for c in chunks] == [f"mydoc_{i}" for i in range(len(chunks))]

    def test_token_count_matches_the_token_slice(self):
        chunker = word_chunker(3)
        chunks = chunker.chunk_text(WHITESPACE, doc_id="d", source="d.md")
        assert sum(c.metadata["token_count"] for c in chunks) == len(WHITESPACE.split())

    def test_empty_and_whitespace_only_documents_yield_no_chunks(self):
        assert word_chunker(4).chunk_text("", doc_id="d", source="s") == []
        assert word_chunker(4).chunk_text("   \n\t  ", doc_id="d", source="s") == []

    @pytest.mark.parametrize("build", BUILDERS)
    def test_overlapping_chunks_still_overlap_in_character_space(self, build):
        document = " ".join(f"word{i}" for i in range(40))
        chunks = build(10, chunk_overlap=3).chunk_text(document, doc_id="d", source="d.md")
        assert len(chunks) > 1
        assert_offsets_exact(document, chunks)
        # Overlap on tokens must show up as overlap on characters.
        assert chunks[1].start_char < chunks[0].end_char
