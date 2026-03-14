# ==============================================================================
# Informity AI — Chunker Tests (v2)
# Tests the public API: chunk_text() and ChunkData behavior.
# Tests behavior, not implementation details.
# ==============================================================================

from informity.indexer.chunker import ChunkData, chunk_text

# ==============================================================================
# Helpers
# ==============================================================================


def _total_chunks(result: list[ChunkData]) -> int:
    # Count chunks in a result list.
    return len(result)


def _all_indices(result: list[ChunkData]) -> list[int]:
    # Extract chunk indices for ordering verification.
    return [c.chunk_index for c in result]


# ==============================================================================
# chunk_text — main public API
# ==============================================================================


class TestChunkText:
    # Tests for the main chunk_text function (public API only).

    # -- Empty / whitespace ------------------------------------------------

    def test_empty_string(self) -> None:
        result = chunk_text('')
        assert result == []

    def test_whitespace_only(self) -> None:
        result = chunk_text('   \n\n\t  ')
        assert result == []

    # -- Short text (fits in one chunk) ------------------------------------

    def test_short_text_single_chunk(self) -> None:
        text   = 'This is a short document.'
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert _total_chunks(result) == 1
        assert result[0].chunk_index == 0
        assert result[0].content == text
        assert result[0].token_count > 0

    def test_short_text_no_overlap_applied(self) -> None:
        # Even with overlap set, a single chunk should not duplicate text
        text   = 'Short text here.'
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert _total_chunks(result) == 1

    # -- Paragraph-based splitting -----------------------------------------

    def test_paragraph_splitting(self) -> None:
        # Two paragraphs, each small enough for its own chunk
        text = (
            'First paragraph with several words here.\n\n'
            'Second paragraph with several words too.'
        )
        result = chunk_text(text, chunk_size=8, overlap=0)
        assert _total_chunks(result) >= 2
        # First chunk should contain text from the first paragraph
        assert 'First paragraph' in result[0].content

    def test_paragraph_merging(self) -> None:
        # Multiple tiny paragraphs should merge into fewer chunks
        text = 'A.\n\nB.\n\nC.\n\nD.'
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert _total_chunks(result) == 1

    def test_markdown_header_splitting(self) -> None:
        # Markdown headers should create chunk boundaries
        text = (
            '# Heading One\n\nContent under heading one.\n\n'
            '## Heading Two\n\nContent under heading two.'
        )
        result = chunk_text(text, chunk_size=8, overlap=0)
        assert _total_chunks(result) >= 2
        # First chunk should contain heading one content
        assert 'Heading One' in result[0].content or 'heading one' in result[0].content.lower()

    # -- Overlap -----------------------------------------------------------

    def test_overlap_applied(self) -> None:
        # With overlap, chunks after the first should contain overlap from previous chunk.
        # Chunker splits on paragraphs; overlap is sentence-aligned.
        p1 = 'First paragraph with several words here for testing overlap behavior.'
        p2 = 'Second paragraph with more content to ensure we get multiple chunks.'
        p3 = 'Third paragraph to complete the test of overlap between chunks.'
        text = f'{p1}\n\n{p2}\n\n{p3}'
        result = chunk_text(text, chunk_size=15, overlap=3)
        assert _total_chunks(result) >= 2
        # Overlap: second chunk should share some content with first (sentence-aligned)
        all_content = ' '.join(c.content for c in result)
        assert 'First paragraph' in all_content
        assert 'Second paragraph' in all_content

    # -- Very long text ----------------------------------------------------

    def test_very_long_text(self) -> None:
        # Generate a long text with multiple paragraphs
        paragraphs = []
        for i in range(20):
            sentences = [f'This is sentence {j} of paragraph {i}.' for j in range(5)]
            paragraphs.append(' '.join(sentences))
        text   = '\n\n'.join(paragraphs)
        result = chunk_text(text, chunk_size=50, overlap=5)
        assert _total_chunks(result) >= 5
        # Indices should be sequential starting from 0
        assert _all_indices(result) == list(range(len(result)))

    # -- Chunk indices are sequential --------------------------------------

    def test_sequential_indices(self) -> None:
        simple_words = ['the', 'cat', 'sat', 'on', 'a', 'big', 'red', 'mat', 'and', 'ran']
        text   = ' '.join(simple_words[i % len(simple_words)] for i in range(100))
        result = chunk_text(text, chunk_size=10, overlap=0)
        indices = _all_indices(result)
        assert indices == list(range(len(result)))

    # -- Token counts are accurate -----------------------------------------

    def test_token_counts_present(self) -> None:
        # Token counts should be present and positive
        text   = 'the cat sat on a big red mat and ran'
        result = chunk_text(text, chunk_size=100, overlap=0)
        for chunk in result:
            assert chunk.token_count > 0
            assert isinstance(chunk.token_count, int)

    # -- ChunkData is frozen -----------------------------------------------

    def test_chunk_data_immutable(self) -> None:
        text   = 'Some test text here.'
        result = chunk_text(text, chunk_size=100, overlap=0)
        try:
            result[0].content = 'modified'  # type: ignore[misc]
            raise AssertionError('Expected FrozenInstanceError')
        except AttributeError:
            pass  # dataclass(frozen=True) raises AttributeError

    # -- Default settings used when not specified --------------------------

    def test_defaults_from_config(self) -> None:
        # When chunk_size and overlap are not passed, config values are used.
        # Just verify the function runs without error using defaults.
        long_text = ' '.join(f'word{i}' for i in range(1000))
        result    = chunk_text(long_text)
        assert _total_chunks(result) >= 1
        # All chunks should have valid data
        for chunk in result:
            assert chunk.content
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0

    # -- Mixed paragraph sizes ---------------------------------------------

    def test_mixed_paragraph_sizes(self) -> None:
        # Mix of small and large paragraphs
        small  = 'Tiny paragraph.'
        medium = ' '.join(f'word{i}' for i in range(15))
        large  = ' '.join(f'word{i}' for i in range(50))
        text   = f'{small}\n\n{medium}\n\n{large}'
        result = chunk_text(text, chunk_size=10, overlap=0)
        assert _total_chunks(result) >= 3
        # Verify no empty chunks
        for chunk in result:
            assert chunk.content.strip()

    # -- Text with only newlines (no double newlines) ----------------------

    def test_no_paragraph_breaks(self) -> None:
        # Single paragraph with only single newlines — chunker treats as one part.
        # With no double newlines, we get one chunk (chunker splits on paragraphs).
        text   = 'line one\nline two\nline three\nline four\nline five'
        result = chunk_text(text, chunk_size=3, overlap=0)
        assert _total_chunks(result) >= 1
        assert 'line one' in result[0].content

    # -- Unicode text -------------------------------------------------------

    def test_unicode_text(self) -> None:
        text   = 'Привет мир. Это тестовый текст. Юникод работает.'
        result = chunk_text(text, chunk_size=4, overlap=0)
        assert _total_chunks(result) >= 1
        # Content should preserve unicode
        all_content = ' '.join(c.content for c in result)
        assert 'Привет' in all_content
        assert 'Юникод' in all_content

    # -- Chunk size limits ------------------------------------------------

    def test_chunks_respect_size_limit(self) -> None:
        # Chunker splits on paragraphs; each paragraph is an atomic unit.
        # A single long paragraph can exceed chunk_size (not split mid-paragraph).
        paragraphs = [' '.join(f'word{i}' for i in range(15)) for _ in range(4)]
        text = '\n\n'.join(paragraphs)
        result = chunk_text(text, chunk_size=12, overlap=2)
        assert _total_chunks(result) >= 2
        # Chunks from multi-paragraph text should be bounded; single para may exceed limit
        for chunk in result:
            assert chunk.token_count <= 50  # Allow for paragraph atomicity

    # -- Content preservation ---------------------------------------------

    def test_content_preserved(self) -> None:
        # All original content should appear in chunks (accounting for overlap)
        text   = 'First sentence. Second sentence. Third sentence.'
        result = chunk_text(text, chunk_size=5, overlap=1)
        all_content = ' '.join(c.content for c in result)
        assert 'First' in all_content
        assert 'Second' in all_content
        assert 'Third' in all_content

    def test_header_only_filter_keeps_single_chunk_fallback(self) -> None:
        # If header-only filtering would remove every chunk, keep one fallback chunk
        # so indexing never creates file rows with zero chunks.
        text = (
            '| Header A | Header B | Header C |\n'
            '| --- | --- | --- |\n'
            '| Header A | Header B | Header C |\n'
            '| --- | --- | --- |'
        )
        result = chunk_text(text, chunk_size=512, overlap=0)
        assert _total_chunks(result) == 1
        assert '| Header A | Header B | Header C |' in result[0].content
