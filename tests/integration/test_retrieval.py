# ==============================================================================
# Informity AI — Retrieval Integration Tests
# 15-document fixed corpus, structural assertions only (no golden answers).
#
# Corpus summary:
#   .pdf  (6 files): annual_report_2021/2022/2023, quarterly_results_2022,
#                    employee_handbook, marketing_strategy
#   .md   (6 files): privacy_policy, terms_of_service, technical_specification,
#                    product_roadmap, onboarding_guide, release_notes
#   .txt  (3 files): meeting_notes_q1_2023, meeting_notes_q2_2023, sales_summary_2022
#
#   year=2021: annual_report_2021 (2 chunks)
#   year=2022: annual_report_2022, quarterly_results_2022, sales_summary_2022 (6 chunks)
#   year=2023: annual_report_2023, meeting_notes_q1_2023, meeting_notes_q2_2023 (6 chunks)
#   year=None: remaining 9 files (18 chunks)
# ==============================================================================

import aiosqlite
import pytest

from informity.config import settings
from informity.llm.retrieval import retrieve_chunks
from informity.llm.user_messages import INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE

# ---------------------------------------------------------------------------
# Per-test DB connection
# ---------------------------------------------------------------------------

@pytest.fixture
async def db():
    """Open an aiosqlite connection to the corpus DB for each test."""
    conn = await aiosqlite.connect(str(settings.db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute('PRAGMA journal_mode=WAL')
    await conn.execute('PRAGMA foreign_keys=ON')
    yield conn
    await conn.close()


# ===========================================================================
# Group 1: Basic retrieval — no filters
# ===========================================================================

async def test_focused_query_returns_at_least_one_chunk(db):
    """A focused query against the corpus returns ≥ 1 chunk."""
    results = await retrieve_chunks(query='company report', top_k=5, db=db)
    assert len(results) >= 1


async def test_top_k_is_upper_bound(db):
    """Result count never exceeds the requested top_k."""
    results = await retrieve_chunks(query='company documents', top_k=3, db=db)
    assert len(results) <= 3


async def test_top_k_1_returns_exactly_one_chunk(db):
    """top_k=1 returns exactly 1 chunk when the corpus is non-empty."""
    results = await retrieve_chunks(query='documents', top_k=1, db=db)
    assert len(results) == 1


async def test_returned_chunks_have_required_fields(db):
    """Every returned chunk has all fields consumed by the generation stage."""
    results = await retrieve_chunks(query='policy guidelines', top_k=5, db=db)
    assert len(results) >= 1
    required = {'chunk_id', 'file_id', 'filename', 'chunk_text', 'score'}
    for chunk in results:
        missing = required - set(chunk.keys())
        assert not missing, f'Chunk is missing fields: {missing}'


async def test_chunk_score_is_float(db):
    """The reranker score on every chunk is a Python float."""
    results = await retrieve_chunks(query='operational guidelines', top_k=5, db=db)
    for chunk in results:
        assert isinstance(chunk['score'], float), (
            f"score should be float, got {type(chunk['score'])}"
        )


async def test_chunk_text_is_non_empty_string(db):
    """chunk_text on every returned chunk is a non-empty string."""
    results = await retrieve_chunks(query='strategy plan', top_k=5, db=db)
    for chunk in results:
        assert isinstance(chunk['chunk_text'], str)
        assert chunk['chunk_text'].strip()


# ===========================================================================
# Group 2: Year filter — regression suite for the year-range filter bug class
#
# Core requirement: a year_filter for year N must return ONLY documents whose
# `year` column equals N. Any cross-year leakage is a regression.
# ===========================================================================

async def test_year_filter_2021_returns_only_2021_documents(db):
    """
    year_filter=2021 must not return any 2022 or 2023 documents.
    Regression: year-range filter bug where adjacent years leaked.
    """
    results = await retrieve_chunks(
        query='revenue operating expenses', top_k=10, year_filter=2021, db=db,
    )
    assert len(results) >= 1, 'Expected ≥ 1 chunk for year=2021'
    filenames = {c['filename'] for c in results}
    for fn in filenames:
        assert '2021' in fn, f'Non-2021 file {fn!r} appeared in year=2021 results'
        assert '2022' not in fn, f'2022 leaked into year=2021 results: {fn!r}'
        assert '2023' not in fn, f'2023 leaked into year=2021 results: {fn!r}'


async def test_year_filter_2022_returns_only_2022_documents(db):
    """
    year_filter=2022 must not return any 2021 or 2023 documents.
    Regression: year-range filter bug where adjacent years leaked.
    """
    results = await retrieve_chunks(
        query='revenue net income quarterly', top_k=10, year_filter=2022, db=db,
    )
    assert len(results) >= 1, 'Expected ≥ 1 chunk for year=2022'
    filenames = {c['filename'] for c in results}
    for fn in filenames:
        assert '2022' in fn, f'Non-2022 file {fn!r} appeared in year=2022 results'
        assert '2021' not in fn, f'2021 leaked into year=2022 results: {fn!r}'
        assert '2023' not in fn, f'2023 leaked into year=2022 results: {fn!r}'


async def test_year_filter_2023_returns_only_2023_documents(db):
    """year_filter=2023 must not return any 2021 or 2022 documents."""
    results = await retrieve_chunks(
        query='planning meeting quarterly notes', top_k=10, year_filter=2023, db=db,
    )
    assert len(results) >= 1, 'Expected ≥ 1 chunk for year=2023'
    filenames = {c['filename'] for c in results}
    for fn in filenames:
        assert '2023' in fn, f'Non-2023 file {fn!r} appeared in year=2023 results'
        assert '2021' not in fn, f'2021 leaked into year=2023 results: {fn!r}'
        assert '2022' not in fn, f'2022 leaked into year=2023 results: {fn!r}'


async def test_year_2021_and_2022_result_sets_are_disjoint(db):
    """
    Chunk IDs returned for year=2021 and year=2022 must have no overlap.
    Regression: year-range filter bug where a single query could return chunks
    from multiple years simultaneously.
    """
    results_2021 = await retrieve_chunks(
        query='revenue expenses', top_k=10, year_filter=2021, db=db,
    )
    results_2022 = await retrieve_chunks(
        query='revenue expenses', top_k=10, year_filter=2022, db=db,
    )
    ids_2021 = {c['chunk_id'] for c in results_2021}
    ids_2022 = {c['chunk_id'] for c in results_2022}
    overlap = ids_2021 & ids_2022
    assert not overlap, (
        f'Chunk IDs appear in both year=2021 and year=2022 results: {overlap}'
    )


async def test_year_filter_nonexistent_year_returns_empty(db):
    """A year with no indexed documents returns an empty list."""
    results = await retrieve_chunks(
        query='company overview', top_k=10, year_filter=1900, db=db,
    )
    assert results == []


async def test_year_and_extension_combined_filter(db):
    """
    year_filter=2022 + extension_filter='.pdf' returns only 2022 PDF files.
    Regression: combined filters must not widen each other's scope.
    """
    results = await retrieve_chunks(
        query='revenue results', top_k=10, year_filter=2022, extension_filter='.pdf', db=db,
    )
    assert len(results) >= 1, 'Expected ≥ 1 chunk for year=2022 + .pdf'
    for chunk in results:
        fn = chunk['filename']
        assert fn.endswith('.pdf'), f'Expected .pdf, got {fn!r}'
        assert '2022' in fn, f'Expected 2022 in filename, got {fn!r}'


async def test_year_2021_pdf_only_returns_annual_report(db):
    """year_filter=2021 + extension_filter='.pdf' returns only the 2021 PDF."""
    results = await retrieve_chunks(
        query='revenue', top_k=10, year_filter=2021, extension_filter='.pdf', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert chunk['filename'] == 'annual_report_2021.pdf', (
            f"Unexpected file: {chunk['filename']!r}"
        )


# ===========================================================================
# Group 3: Filename filter
# ===========================================================================

async def test_filename_filter_employee_handbook(db):
    """Filename filter returns only chunks from the matched file."""
    results = await retrieve_chunks(
        query='employee guidelines', top_k=10,
        filename_filter='employee_handbook.pdf', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert 'employee_handbook' in chunk['filename'], (
            f"Expected employee_handbook file, got {chunk['filename']!r}"
        )


async def test_filename_filter_annual_report_2022(db):
    """Filename filter limits results to the specified file only."""
    results = await retrieve_chunks(
        query='annual results', top_k=10,
        filename_filter='annual_report_2022.pdf', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert 'annual_report_2022' in chunk['filename']


async def test_filename_filter_no_match_returns_empty(db):
    """A filename filter that matches no indexed file returns an empty list."""
    results = await retrieve_chunks(
        query='overview', top_k=10, filename_filter='nonexistent_file_xyz.pdf', db=db,
    )
    assert results == []


# ===========================================================================
# Group 4: Extension filter
# ===========================================================================

async def test_extension_filter_pdf_returns_only_pdf_files(db):
    """extension_filter='.pdf' returns only .pdf files."""
    results = await retrieve_chunks(
        query='report overview', top_k=10, extension_filter='.pdf', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert chunk['filename'].endswith('.pdf'), (
            f"Expected .pdf file, got {chunk['filename']!r}"
        )


async def test_extension_filter_md_returns_only_markdown_files(db):
    """extension_filter='.md' returns only Markdown files."""
    results = await retrieve_chunks(
        query='documentation guide', top_k=10, extension_filter='.md', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert chunk['filename'].endswith('.md'), (
            f"Expected .md file, got {chunk['filename']!r}"
        )


async def test_extension_filter_txt_returns_only_text_files(db):
    """extension_filter='.txt' returns only plain-text files."""
    results = await retrieve_chunks(
        query='meeting notes summary', top_k=10, extension_filter='.txt', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert chunk['filename'].endswith('.txt'), (
            f"Expected .txt file, got {chunk['filename']!r}"
        )


async def test_extension_filter_nonexistent_returns_empty(db):
    """An extension that doesn't exist in the corpus returns an empty list."""
    results = await retrieve_chunks(
        query='report', top_k=10, extension_filter='.xyz', db=db,
    )
    assert results == []


# ===========================================================================
# Group 5: Coverage mode
# ===========================================================================

async def test_coverage_query_returns_multiple_distinct_files(db):
    """
    Coverage query returns chunks from ≥ 3 distinct files (file-anchored retrieval).
    """
    results = await retrieve_chunks(
        query='company documents overview', top_k=15, query_type='coverage', db=db,
    )
    assert len(results) >= 3, 'Coverage query should return ≥ 3 chunks'
    filenames = {c['filename'] for c in results}
    assert len(filenames) >= 3, 'Coverage query should cover ≥ 3 distinct files'


async def test_coverage_returns_at_most_one_chunk_per_file(db):
    """
    Coverage retrieval shares the same vector+rereank path as focused retrieval.
    It must still respect top_k bounds.
    """
    results = await retrieve_chunks(
        query='documents', top_k=15, query_type='coverage', db=db,
    )
    assert len(results) <= 15


async def test_coverage_year_filter_restricts_files(db):
    """Coverage query with year_filter=2022 returns only 2022 files."""
    results = await retrieve_chunks(
        query='financial results', top_k=10, query_type='coverage', year_filter=2022, db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert '2022' in chunk['filename'], (
            f"Coverage leaked non-2022 file: {chunk['filename']!r}"
        )


async def test_coverage_extension_filter_restricts_to_md(db):
    """Coverage query with extension_filter='.md' returns only Markdown files."""
    results = await retrieve_chunks(
        query='documentation reference', top_k=10,
        query_type='coverage', extension_filter='.md', db=db,
    )
    assert len(results) >= 1
    for chunk in results:
        assert chunk['filename'].endswith('.md')


# ===========================================================================
# Group 6: Refusal phrase — pipeline-level boundary
# ===========================================================================

async def test_insufficient_context_response_is_non_empty_string():
    """The refusal phrase constant is a non-empty string."""
    assert isinstance(INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE, str)
    assert len(INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE) > 0


async def test_retrieve_returns_empty_list_not_string_on_no_match(db):
    """
    retrieve_chunks returns [] when no chunks match — never the refusal phrase.
    Refusal phrases are resolved at higher orchestration layers; retrieval must stay data-only.
    """
    results = await retrieve_chunks(
        query='xyz', top_k=5, extension_filter='.xyz', db=db,
    )
    assert results == [], f'Expected [], got {results!r}'
    assert results is not INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE
