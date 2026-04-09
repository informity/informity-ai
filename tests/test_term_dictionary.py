from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from informity.config import settings
from informity.db.models import Chunk, FileCategory, IndexedFile
from informity.db.sqlite import (
    get_active_term_alias_rows,
    get_connection,
    init_db,
    insert_chunks_batch,
    insert_file,
)
from informity.indexer import term_dictionary_builder as term_builder
from informity.indexer.term_dictionary_builder import rebuild_term_dictionary
from informity.indexer.term_dictionary_quality import evaluate_term_dictionary_quality
from informity.llm import term_dictionary


@pytest.mark.asyncio
async def test_expand_query_for_retrieval_applies_tiered_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', True)

    async def _fake_version(_db: aiosqlite.Connection) -> int:
        return 3

    async def _fake_rows(_db: aiosqlite.Connection) -> list[dict]:
        return [
            {
                'alias': 'roi',
                'normalized_alias': 'roi',
                'alias_type': 'observed',
                'alias_confidence': 0.9,
                'canonical_term': 'return on investment',
                'normalized_term': 'return on investment',
                'term_type': 'acronym',
                'term_confidence': 0.92,
            },
            {
                'alias': 'narrative brief',
                'normalized_alias': 'narrative brief',
                'alias_type': 'observed',
                'alias_confidence': 0.7,
                'canonical_term': 'evidence summary',
                'normalized_term': 'evidence summary',
                'term_type': 'domain_term',
                'term_confidence': 0.7,
            },
        ]

    monkeypatch.setattr(term_dictionary, 'get_term_dictionary_current_version', _fake_version)
    monkeypatch.setattr(term_dictionary, 'get_active_term_alias_rows', _fake_rows)

    db = await aiosqlite.connect(':memory:')
    try:
        expansion = await term_dictionary.expand_query_for_retrieval(
            db=db,
            query='Show ROI and narrative brief by year',
        )
    finally:
        await db.close()

    assert expansion.dictionary_version == 3
    # High confidence term appears in embedding expansion.
    assert any(term == 'return on investment' for term in expansion.embedding_terms)
    # Medium confidence term appears in FTS expansion only.
    assert any(term == 'evidence summary' for term in expansion.fts_terms)
    assert all(term != 'evidence summary' for term in expansion.embedding_terms)


@pytest.mark.asyncio
async def test_rebuild_term_dictionary_extracts_acronym_definition_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', True)
    monkeypatch.setattr(settings, 'term_dictionary_build_enabled', True)

    db_path = tmp_path / 'term-dictionary.db'
    monkeypatch.setattr(settings, 'db_path', db_path)
    await init_db()

    db = await get_connection()
    try:
        indexed = await insert_file(
            db,
            IndexedFile(
                path=str(tmp_path / 'doc1.txt'),
                filename='doc1.txt',
                extension='.txt',
                size_bytes=120,
                content_hash='hash-doc1',
                extracted_text_preview='Return on Investment (ROI) discussed.',
                category=FileCategory.PLAINTEXT,
                tags=[],
                year=2025,
                modified_at=datetime.now(UTC),
            ),
        )
        await insert_chunks_batch(
            db,
            indexed.id or 0,
            [
                Chunk(
                    file_id=indexed.id or 0,
                    chunk_index=0,
                    content='Return on Investment (ROI) improved in Q4.',
                    token_count=12,
                )
            ],
        )

        result = await rebuild_term_dictionary(db, run_id='term-dict-test-run')
        aliases = await get_active_term_alias_rows(db)
    finally:
        await db.close()

    assert result['status'] == 'completed'
    assert result['terms_inserted'] >= 1
    assert any(row['normalized_alias'] == 'roi' for row in aliases)
    assert any('return on investment' in row['normalized_term'] for row in aliases)


def test_expand_query_for_routing_applies_high_confidence_and_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', True)
    monkeypatch.setattr(settings, 'term_dictionary_routing_enabled', True)
    monkeypatch.setattr(settings, 'term_dictionary_max_routing_expansions', 1)

    def _fake_rows() -> tuple[int, list[dict]]:
        return 2, [
            {
                'normalized_alias': 'roi',
                'canonical_term': 'return on investment',
                'term_confidence': 0.92,
            },
            {
                'normalized_alias': 'narrative brief',
                'canonical_term': 'evidence summary',
                'term_confidence': 0.91,
            },
        ]

    monkeypatch.setattr(term_dictionary, '_get_active_term_alias_rows_sync', _fake_rows)
    expansion = term_dictionary.expand_query_for_routing('Show ROI and narrative brief')

    assert expansion.dictionary_version == 2
    assert len(expansion.canonical_terms) == 1
    assert expansion.canonical_terms[0] == 'return on investment'
    assert 'return on investment' in expansion.expanded_query


@pytest.mark.asyncio
async def test_expand_query_for_retrieval_fuzzy_cap_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', True)
    monkeypatch.setattr(settings, 'term_dictionary_max_fuzzy_expansions', 1)
    monkeypatch.setattr(settings, 'term_dictionary_max_fuzzy_per_canonical', 1)

    async def _fake_version(_db: aiosqlite.Connection) -> int:
        return 7

    async def _fake_rows(_db: aiosqlite.Connection) -> list[dict]:
        return [
            {
                'alias': 'mortgage',
                'normalized_alias': 'mortgage',
                'alias_type': 'observed',
                'alias_confidence': 0.9,
                'canonical_term': 'mortgage loan',
                'normalized_term': 'mortgage loan',
                'term_type': 'domain_term',
                'term_confidence': 0.9,
            },
            {
                'alias': 'amortization',
                'normalized_alias': 'amortization',
                'alias_type': 'observed',
                'alias_confidence': 0.9,
                'canonical_term': 'amortization schedule',
                'normalized_term': 'amortization schedule',
                'term_type': 'domain_term',
                'term_confidence': 0.9,
            },
        ]

    monkeypatch.setattr(term_dictionary, 'get_term_dictionary_current_version', _fake_version)
    monkeypatch.setattr(term_dictionary, 'get_active_term_alias_rows', _fake_rows)

    db = await aiosqlite.connect(':memory:')
    try:
        expansion = await term_dictionary.expand_query_for_retrieval(
            db=db,
            query='show mortgagf and amortizatiom details',
        )
    finally:
        await db.close()

    assert expansion.dictionary_version == 7
    assert expansion.fuzzy_cap_reached is True
    fuzzy_matches = [match for match in expansion.matches if match.match_type == 'fuzzy']
    assert len(fuzzy_matches) == 1


@pytest.mark.asyncio
async def test_expand_query_for_retrieval_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', False)
    db = await aiosqlite.connect(':memory:')
    try:
        expansion = await term_dictionary.expand_query_for_retrieval(
            db=db,
            query='ROI summary',
        )
    finally:
        await db.close()

    assert expansion.dictionary_version == 0
    assert expansion.embedding_query == 'ROI summary'
    assert expansion.fts_query == 'ROI summary'
    assert expansion.matches == []


def test_expand_query_for_routing_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'term_dictionary_enabled', False)
    monkeypatch.setattr(settings, 'term_dictionary_routing_enabled', True)
    expansion = term_dictionary.expand_query_for_routing('ROI summary')
    assert expansion.dictionary_version == 0
    assert expansion.expanded_query == 'ROI summary'
    assert expansion.canonical_terms == []


def test_term_dictionary_builder_rejects_boilerplate_canonical_phrase() -> None:
    out: dict[tuple[str, str], term_builder._Candidate] = {}
    term_builder._extract_candidates_from_chunk(
        content='See instructions (SI) for all details.',
        file_id=1,
        chunk_id=1,
        out=out,
    )
    assert out == {}


def test_term_dictionary_builder_excludes_person_name_pairs_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'entity_extract_acronym', True)
    monkeypatch.setattr(settings, 'entity_extract_person_name', False)
    out: dict[tuple[str, str], term_builder._Candidate] = {}
    term_builder._extract_candidates_from_chunk(
        content='Jane Doe (JD) approved the request.',
        file_id=1,
        chunk_id=1,
        out=out,
    )
    assert out == {}


def test_term_dictionary_builder_ocr_normalization_extracts_split_acronym() -> None:
    out: dict[tuple[str, str], term_builder._Candidate] = {}
    term_builder._extract_candidates_from_chunk(
        content='Return on Investment (R O I) improved in Q4.',
        file_id=1,
        chunk_id=1,
        out=out,
    )
    candidate = out.get(('acronym', 'return on investment'))
    assert candidate is not None
    assert 'roi' in candidate.aliases


def test_term_dictionary_builder_person_name_extractor_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'entity_extract_acronym', False)
    monkeypatch.setattr(settings, 'entity_extract_person_name', True)
    out: dict[tuple[str, str], term_builder._Candidate] = {}
    term_builder._extract_candidates_from_chunk(
        content='Dr. Maria Garcia met with Maria Garcia yesterday.',
        file_id=1,
        chunk_id=1,
        out=out,
    )
    assert ('person_name', 'maria garcia') in out


def test_term_dictionary_quality_gate_fails_for_high_noise_rate() -> None:
    result = evaluate_term_dictionary_quality(
        total_candidates=100,
        kept_candidates=20,
        noise_rate_threshold=0.55,
        min_candidates_for_gate=20,
        gate_enabled=True,
    )
    assert result.passed is False
    assert result.reason.startswith('noise_rate_exceeded:')
