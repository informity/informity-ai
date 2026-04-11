# ==============================================================================
# Informity AI — Model Profile Tests
# Tests profile detection, selection, stop sequences, reasoning mode, prompt
# format, max tokens, and model-specific behavior for Qwen3.5 35B A3B,
# Qwen3.5 9B, Qwen3 14B, and the default profile.
# ==============================================================================

import pytest

from informity.llm.model_adapter import (
    DEFAULT_PROFILE,
    QWEN3_5_9B_PROFILE,
    QWEN3_5_35B_A3B_PROFILE,
    QWEN3_14B_PROFILE,
    ModelFamily,
    ModelProfile,
    PromptFormat,
    ReasoningMode,
    get_profile_for_filename,
    get_retrieval_top_k,
)

# ==============================================================================
# Profile Detection (filename -> profile)
# ==============================================================================


class TestGetProfileForFilename:
    def test_qwen3_5_35b_a3b_detected(self) -> None:
        profile = get_profile_for_filename('Qwen3.5-35B-A3B-Q4_K_M.gguf')
        assert profile is QWEN3_5_35B_A3B_PROFILE
        assert profile.name == 'Qwen3.5 35B A3B'

    def test_qwen3_5_35b_a3b_lowercase(self) -> None:
        profile = get_profile_for_filename('qwen3.5-35b-a3b-q4_k_m.gguf')
        assert profile is QWEN3_5_35B_A3B_PROFILE

    def test_qwen2_5_3b_returns_default(self) -> None:
        # No dedicated Qwen2.5-3B profile; falls through to default
        profile = get_profile_for_filename('Qwen2.5-3B-Instruct-Q4_K_M.gguf')
        assert profile is DEFAULT_PROFILE

    def test_qwen3_14b_detected(self) -> None:
        # Qwen3 14B has a dedicated analysis profile
        profile = get_profile_for_filename('Qwen3-14B-Q4_K_M.gguf')
        assert profile is QWEN3_14B_PROFILE

    def test_qwen3_8b_returns_default(self) -> None:
        profile = get_profile_for_filename('Qwen3-8B-Q5_K_M.gguf')
        assert profile is DEFAULT_PROFILE

    def test_unknown_returns_default(self) -> None:
        profile = get_profile_for_filename('custom-model.gguf')
        assert profile is DEFAULT_PROFILE
        assert profile.name == 'Unknown (ChatML default)'

    def test_llama_returns_default(self) -> None:
        # Llama is no longer a supported profile; use default
        profile = get_profile_for_filename('Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
        assert profile is DEFAULT_PROFILE

    def test_mistral_nemo_returns_default(self) -> None:
        # Mistral Nemo is no longer a supported profile; falls through to default
        profile = get_profile_for_filename('Mistral-Nemo-Instruct-2407-Q4_K_M.gguf')
        assert profile is DEFAULT_PROFILE

    def test_phi4_returns_default(self) -> None:
        profile = get_profile_for_filename('Phi-4-mini-reasoning-Q8_0.gguf')
        assert profile is DEFAULT_PROFILE

    def test_gemma_returns_default(self) -> None:
        profile = get_profile_for_filename('Gemma-2-9B-It-Q4_K_M.gguf')
        assert profile is DEFAULT_PROFILE


# ==============================================================================
# Qwen3.5 35B A3B Profile (primary large model)
# ==============================================================================


class TestQwen3535BA3BProfile:
    @pytest.fixture
    def profile(self) -> ModelProfile:
        return QWEN3_5_35B_A3B_PROFILE

    def test_identity(self, profile: ModelProfile) -> None:
        assert profile.name == 'Qwen3.5 35B A3B'
        assert profile.family == ModelFamily.CHATML
        assert profile.supports_think_blocks is True

    def test_reasoning_disabled(self, profile: ModelProfile) -> None:
        assert profile.reasoning_mode == ReasoningMode.NEVER
        assert profile.get_reasoning_enabled('simple') is False
        assert profile.get_reasoning_enabled('focused') is False
        assert profile.get_reasoning_enabled('coverage') is False

    def test_chat_template_kwargs(self, profile: ModelProfile) -> None:
        assert profile.chat_template_kwargs == {'enable_thinking': False}
        assert profile.no_think_token is None

    def test_prompt_format_always_native(self, profile: ModelProfile) -> None:
        assert profile.get_prompt_format('simple') == PromptFormat.NATIVE_GGUF
        assert profile.get_prompt_format('focused') == PromptFormat.NATIVE_GGUF
        assert profile.get_prompt_format('coverage') == PromptFormat.NATIVE_GGUF

    def test_max_tokens(self, profile: ModelProfile) -> None:
        assert profile.get_max_tokens('simple') == 3072
        assert profile.get_max_tokens('focused') == 3072
        assert profile.get_max_tokens('coverage') == 3072

    def test_retrieval_top_k_fields(self, profile: ModelProfile) -> None:
        assert profile.retrieval_top_k_candidates > 0
        assert profile.retrieval_top_k_final > 0

    def test_requested_tuning_values(self, profile: ModelProfile) -> None:
        assert profile.max_tokens == 3072
        assert profile.coverage_top_k == 18
        assert profile.timeout_seconds == 900
        assert profile.context_length == 24576
        assert profile.temperature == 0.2
        assert profile.rag_top_k == 10
        assert profile.rag_max_score == 0.90
        assert profile.rag_context_ratio == 0.65
        assert profile.retrieval_top_k_final == 12

    def test_stop_sequences_include_chatml(self, profile: ModelProfile) -> None:
        stops = profile.get_stop_sequences(reasoning_enabled=True)
        assert '<|im_end|>' in stops
        assert '<|im_start|>' in stops
        assert '<|endoftext|>' in stops

    def test_stop_sequences_no_reasoning_does_not_stop_on_think(self, profile: ModelProfile) -> None:
        no_reasoning = profile.get_stop_sequences(reasoning_enabled=False)
        assert '<think>' not in no_reasoning

    def test_prepare_messages_does_not_append_no_think(self, profile: ModelProfile) -> None:
        messages = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'What is X?'},
        ]
        result = profile.prepare_messages(messages, 'simple')
        assert result[-1]['content'] == messages[-1]['content']


# ==============================================================================
# Qwen3.5 9B Profile (non-thinking template mode)
# ==============================================================================


class TestQwen359BProfile:
    @pytest.fixture
    def profile(self) -> ModelProfile:
        return QWEN3_5_9B_PROFILE

    def test_reasoning_disabled_by_profile(self, profile: ModelProfile) -> None:
        assert profile.reasoning_mode == ReasoningMode.NEVER
        assert profile.no_think_token is None

    def test_sampling_defaults_for_non_thinking_mode(self, profile: ModelProfile) -> None:
        assert profile.temperature == 0.7
        assert profile.top_p == 0.8

    def test_template_kwargs_disable_thinking(self, profile: ModelProfile) -> None:
        assert profile.chat_template_kwargs == {'enable_thinking': False}

    def test_prepare_messages_does_not_append_no_think_token(self, profile: ModelProfile) -> None:
        messages = [
            {'role': 'system', 'content': 'System.'},
            {'role': 'user', 'content': 'Summarize the indexed evidence.'},
        ]
        result = profile.prepare_messages(messages, 'focused')
        assert result[-1]['content'] == messages[-1]['content']


# ==============================================================================
# ==============================================================================
# Default Profile
# ==============================================================================


class TestDefaultProfile:
    def test_conservative_settings(self) -> None:
        assert DEFAULT_PROFILE.supports_think_blocks is True
        assert DEFAULT_PROFILE.reasoning_mode == ReasoningMode.FOCUSED_ONLY

    def test_retrieval_top_k_standard(self) -> None:
        assert DEFAULT_PROFILE.retrieval_top_k_candidates == 25
        assert DEFAULT_PROFILE.retrieval_top_k_final == 12

    def test_no_think_token(self) -> None:
        # Default also supports /no_think (safe for ChatML models)
        assert DEFAULT_PROFILE.no_think_token == '/no_think'

    def test_stop_sequences_returned_when_no_reasoning(self) -> None:
        stops = DEFAULT_PROFILE.get_stop_sequences(reasoning_enabled=False)
        assert isinstance(stops, list)
        assert len(stops) > 0


# ==============================================================================
# ModelProfile Methods (generic)
# ==============================================================================


class TestModelProfileMethods:
    def test_get_max_tokens_unknown_type_returns_focused(self) -> None:
        profile = QWEN3_5_35B_A3B_PROFILE
        assert profile.get_max_tokens('unknown') == profile.max_tokens

    def test_get_prompt_format_unknown_type_returns_default(self) -> None:
        profile = QWEN3_5_35B_A3B_PROFILE
        assert profile.get_prompt_format('unknown') == profile.prompt_format

    def test_prepare_messages_no_mutation(self) -> None:
        messages = [{'role': 'user', 'content': 'Test'}]
        _ = QWEN3_5_35B_A3B_PROFILE.prepare_messages(messages, 'simple')
        assert messages[0]['content'] == 'Test'  # Original not mutated

    def test_to_display_dict_contains_all_keys(self) -> None:
        display = QWEN3_5_35B_A3B_PROFILE.to_display_dict()
        expected_keys = {
            'name', 'family', 'supports_reasoning', 'reasoning_mode',
            'max_tokens', 'coverage_top_k', 'min_tokens_coverage',
            'prompt_format', 'coverage_prompt_format', 'context_length',
            'temperature', 'top_p', 'rag_top_k', 'retrieval_top_k_candidates', 'retrieval_top_k_final',
            'rag_top_k_simple', 'rag_top_k_focused', 'rag_top_k_coverage',
            'rag_max_score', 'rag_context_ratio', 'timeout_seconds',
        }
        assert expected_keys == set(display.keys())

    def test_default_model_uses_qwen3_5_35b_profile(self) -> None:
        # Default config points at Qwen3.5 35B A3B
        from informity.config import _DEFAULT_LLM_MODEL_FILENAME
        profile = get_profile_for_filename(_DEFAULT_LLM_MODEL_FILENAME)
        assert profile is QWEN3_5_35B_A3B_PROFILE


# ==============================================================================
# get_retrieval_top_k (override point for adaptive tuning)
# ==============================================================================


class TestGetRetrievalTopK:
    def test_returns_profile_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = monkeypatch
        from informity.llm.model_adapter import get_profile

        profile = get_profile()
        expected_focused = profile.rag_top_k_focused or profile.retrieval_top_k_final
        expected_coverage = profile.rag_top_k_coverage or profile.coverage_top_k
        assert get_retrieval_top_k('focused') == expected_focused
        assert get_retrieval_top_k('coverage') == expected_coverage
