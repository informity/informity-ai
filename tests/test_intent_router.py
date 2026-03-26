from __future__ import annotations

import pytest

from informity.llm.intent_router import IntentPrediction, get_intent_router
from informity.llm.promptcue_adapter import (
    _BROAD_FLIPS,
    _DEFAULT_INTENT,
    PromptCueIntentAdapter,
    _map_intent,
)

# ==============================================================================
# Type → intent mapping
# ==============================================================================

class TestMapIntent:
    def test_chitchat_always_simple(self) -> None:
        assert _map_intent('chitchat', 'broad')    == 'simple'
        assert _map_intent('chitchat', 'focused')  == 'simple'
        assert _map_intent('chitchat', 'unknown')  == 'simple'

    def test_lookup_focused_by_default(self) -> None:
        assert _map_intent('lookup', 'focused')  == 'focused'
        assert _map_intent('lookup', 'unknown')  == 'focused'

    def test_lookup_flips_to_coverage_on_broad(self) -> None:
        assert _map_intent('lookup', 'broad') == 'coverage'

    def test_analysis_coverage_by_default(self) -> None:
        assert _map_intent('analysis', 'broad')   == 'coverage'
        assert _map_intent('analysis', 'unknown') == 'coverage'

    def test_analysis_stays_coverage_on_focused_scope(self) -> None:
        assert _map_intent('analysis', 'focused') == 'coverage'

    def test_comparison_stays_coverage_on_focused_scope(self) -> None:
        assert _map_intent('comparison', 'focused') == 'coverage'

    def test_summarization_stays_coverage_on_focused_scope(self) -> None:
        assert _map_intent('summarization', 'focused') == 'coverage'

    def test_generation_coverage_by_default(self) -> None:
        assert _map_intent('generation', 'broad') == 'coverage'

    def test_update_flips_to_coverage_on_broad(self) -> None:
        assert _map_intent('update', 'broad')   == 'coverage'
        assert _map_intent('update', 'focused') == 'focused'

    def test_unknown_type_defaults_to_focused(self) -> None:
        assert _map_intent('unknown_type', 'broad') == 'focused'

    def test_all_twelve_types_are_mapped(self) -> None:
        expected_types = {
            'chitchat', 'lookup', 'procedure', 'troubleshooting',
            'recommendation', 'validation', 'update',
            'analysis', 'comparison', 'summarization', 'coverage', 'generation',
        }
        assert set(_DEFAULT_INTENT.keys()) == expected_types

    def test_broad_flip_set_contains_focused_defaults(self) -> None:
        # Every type in _BROAD_FLIPS must have 'focused' as its base default.
        for query_type in _BROAD_FLIPS:
            assert _DEFAULT_INTENT[query_type] == 'focused', query_type

    def test_coverage_defaults_remain_coverage_without_focused_flip_set(self) -> None:
        coverage_defaults = {'analysis', 'comparison', 'summarization', 'coverage', 'generation'}
        for query_type in coverage_defaults:
            assert _DEFAULT_INTENT[query_type] == 'coverage', query_type
            assert _map_intent(query_type, 'focused') == 'coverage'


# ==============================================================================
# Default router is PromptCueIntentAdapter
# ==============================================================================

class TestDefaultRouterIsPromptCue:
    def test_get_intent_router_returns_adapter(self) -> None:
        router = get_intent_router()
        assert isinstance(router, PromptCueIntentAdapter)

    def test_empty_query_returns_simple(self) -> None:
        router = get_intent_router()
        assert isinstance(router, PromptCueIntentAdapter)
        prediction = router.classify_intent('')
        assert isinstance(prediction, IntentPrediction)
        assert prediction.intent == 'simple'
        assert prediction.confidence == pytest.approx(1.0)
        assert 'empty_query_default' in prediction.reason_codes

    def test_classify_intent_fallback_returns_intent_prediction(self) -> None:
        """classify_intent() must always return an IntentPrediction, even on error."""
        from unittest.mock import patch

        adapter = PromptCueIntentAdapter()
        # Force _get_analyzer to raise so the fallback path is exercised.
        with patch.object(adapter, '_get_analyzer', side_effect=RuntimeError('model unavailable')):
            prediction = adapter.classify_intent('What files are indexed?')
        assert isinstance(prediction, IntentPrediction)
        assert prediction.intent == 'focused'
        assert 'promptcue_adapter_failed' in prediction.reason_codes
