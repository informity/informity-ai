# ==============================================================================
# Informity AI — Adaptive RAG Tuning Tests
# Tests get_corpus_stats, calculate_adaptive_top_k, cache, and get_retrieval_top_k.
# ==============================================================================

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from informity.db.sqlite import get_connection, get_corpus_stats
from informity.indexer.adaptive_tuning import (
    calculate_adaptive_top_k,
    get_effective_top_k,
    invalidate_tuning_cache,
    update_tuning_cache,
)

# ==============================================================================
# get_corpus_stats
# ==============================================================================


@pytest.mark.asyncio
async def test_get_corpus_stats_returns_valid_structure() -> None:
    conn = await get_connection()
    try:
        stats = await get_corpus_stats(conn)
        assert isinstance(stats['total_files'], int) and stats['total_files'] >= 0
        assert isinstance(stats['total_parent_chunks'], int) and stats['total_parent_chunks'] >= 0
        assert isinstance(stats['total_child_chunks'], int) and stats['total_child_chunks'] >= 0
        assert stats['last_scan_at'] is None or hasattr(stats['last_scan_at'], 'isoformat')
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_get_corpus_stats_returns_dict_keys() -> None:
    conn = await get_connection()
    try:
        stats = await get_corpus_stats(conn)
        assert set(stats.keys()) == {
            'total_files',
            'total_parent_chunks',
            'total_child_chunks',
            'last_scan_at',
        }
    finally:
        await conn.close()


# ==============================================================================
# calculate_adaptive_top_k
# ==============================================================================


class TestCalculateAdaptiveTopK:
    def test_coverage_empty_corpus_returns_profile_base(self) -> None:
        result = calculate_adaptive_top_k(0, 0, 'coverage', 18)
        assert result == 18

    def test_focused_empty_corpus_returns_profile_base(self) -> None:
        result = calculate_adaptive_top_k(0, 0, 'focused', 10)
        assert result == 10

    def test_coverage_small_corpus(self) -> None:
        # 83 files * 0.25 = 20, max(20, 18) = 20
        result = calculate_adaptive_top_k(83, 247, 'coverage', 18)
        assert result == 20

    def test_coverage_capped_at_30(self) -> None:
        result = calculate_adaptive_top_k(200, 5000, 'coverage', 18)
        assert result == 30

    def test_focused_small_corpus_under_500(self) -> None:
        # total_parent_chunks=247 < 500: max(10, min(10, 12)) = 10
        result = calculate_adaptive_top_k(83, 247, 'focused', 10)
        assert 10 <= result <= 12

    def test_focused_medium_corpus(self) -> None:
        # 1000 parent chunks: log curve
        result = calculate_adaptive_top_k(100, 1000, 'focused', 10)
        assert 10 <= result <= 25

    def test_focused_large_corpus(self) -> None:
        result = calculate_adaptive_top_k(500, 5000, 'focused', 10)
        assert 10 <= result <= 25

    def test_profile_base_as_floor(self) -> None:
        # Very small corpus: coverage should not go below profile
        result = calculate_adaptive_top_k(10, 50, 'coverage', 18)
        assert result >= 18


# ==============================================================================
# Cache and get_effective_top_k
# ==============================================================================


class TestAdaptiveTuningCache:
    def test_get_effective_top_k_returns_none_when_disabled(self) -> None:
        invalidate_tuning_cache()
        with patch('informity.indexer.adaptive_tuning.settings') as mock:
            mock.adaptive_rag_tuning = False
            assert get_effective_top_k('focused') is None
            assert get_effective_top_k('coverage') is None

    def test_get_effective_top_k_returns_none_when_cache_invalid(self) -> None:
        invalidate_tuning_cache()
        with patch('informity.indexer.adaptive_tuning.settings') as mock:
            mock.adaptive_rag_tuning = True
            assert get_effective_top_k('focused') is None
            assert get_effective_top_k('coverage') is None

    @pytest.mark.asyncio
    async def test_update_tuning_cache_does_not_raise(self) -> None:
        invalidate_tuning_cache()
        conn = await get_connection()
        try:
            await update_tuning_cache(conn)
            # Cache may or may not be populated depending on settings and DB state
            # Main assertion: no exception (logic works)
        finally:
            await conn.close()
        invalidate_tuning_cache()

    @pytest.mark.asyncio
    async def test_profile_change_triggers_cache_refresh(self) -> None:
        invalidate_tuning_cache()
        conn = await get_connection()
        try:
            profile_a = SimpleNamespace(name='profile-a', rag_top_k=10, coverage_top_k=18)
            profile_b = SimpleNamespace(name='profile-b', rag_top_k=12, coverage_top_k=20)
            stats = {
                'total_files': 0,
                'total_parent_chunks': 0,
            }

            with patch('informity.indexer.adaptive_tuning.get_profile', side_effect=[profile_a, profile_a]), \
                 patch('informity.indexer.adaptive_tuning.get_corpus_stats', AsyncMock(return_value=stats)):
                await update_tuning_cache(conn, force_recompute=True)
                assert get_effective_top_k('focused') == 10
                assert get_effective_top_k('coverage') == 18

            with patch('informity.indexer.adaptive_tuning.get_profile', side_effect=[profile_b, profile_b]), \
                 patch('informity.indexer.adaptive_tuning.get_corpus_stats', AsyncMock(return_value=stats)):
                await update_tuning_cache(conn)
                assert get_effective_top_k('focused') == 12
                assert get_effective_top_k('coverage') == 20
        finally:
            await conn.close()
        invalidate_tuning_cache()

    @pytest.mark.asyncio
    async def test_file_count_delta_triggers_cache_refresh(self) -> None:
        invalidate_tuning_cache()
        conn = await get_connection()
        try:
            profile = SimpleNamespace(name='profile-a', rag_top_k=8, coverage_top_k=5)
            stats_initial = {
                'total_files': 40,
                'total_parent_chunks': 1000,
            }
            stats_changed = {
                'total_files': 60,
                'total_parent_chunks': 1000,
            }

            with patch('informity.indexer.adaptive_tuning.get_profile', side_effect=[profile, profile]), \
                 patch('informity.indexer.adaptive_tuning.get_corpus_stats', AsyncMock(return_value=stats_initial)):
                await update_tuning_cache(conn, force_recompute=True)
                assert get_effective_top_k('coverage') == 10

            with patch('informity.indexer.adaptive_tuning.get_profile', side_effect=[profile, profile]), \
                 patch('informity.indexer.adaptive_tuning.get_corpus_stats', AsyncMock(return_value=stats_changed)):
                await update_tuning_cache(conn)
                assert get_effective_top_k('coverage') == 15
        finally:
            await conn.close()
        invalidate_tuning_cache()
