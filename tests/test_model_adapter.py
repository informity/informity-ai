# ==============================================================================
# Informity AI — Model Profile Tests
# Tests profile detection, selection, stop sequences, reasoning mode, prompt
# format, max tokens, and model-specific behavior for Qwen3.6 35B A3B,
# Qwen3.5 9B, Qwen3 14B, and the default profile.
# ==============================================================================

"""Test module for tests test model adapter."""

# pylint: disable=redefined-outer-name

import pytest

from informity.llm.model_adapter import (
    DEFAULT_PROFILE,
    OLLAMA_DEFAULT_PROFILE,
    QWEN3_5_4B_ROUTER_PROFILE,
    QWEN3_5_9B_PROFILE,
    QWEN3_6_35B_A3B_PROFILE,
    QWEN3_14B_PROFILE,
    ModelFamily,
    ModelProfile,
    PromptFormat,
    ReasoningMode,
    get_profile,
    get_profile_for_filename,
    get_retrieval_top_k,
    infer_model_id_from_ollama_model,
)

# ==============================================================================
# Profile Detection (filename -> profile)
# ==============================================================================


class TestGetProfileForFilename:
    """Class docstring."""
    def test_qwen3_5_35b_a3b_detected(self) -> None:
        """Test qwen3 5 35b a3b detected."""
        profile = get_profile_for_filename("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        assert profile is QWEN3_6_35B_A3B_PROFILE
        assert profile.name == "Qwen3.6 35B A3B"

    def test_qwen3_5_35b_a3b_lowercase(self) -> None:
        """Test qwen3 5 35b a3b lowercase."""
        profile = get_profile_for_filename("qwen3.6-35b-a3b-q4_k_m.gguf")
        assert profile is QWEN3_6_35B_A3B_PROFILE

    def test_qwen3_5_4b_classifier_detected(self) -> None:
        """Test qwen3 5 4b classifier detected."""
        profile = get_profile_for_filename("Qwen3.5-4B-Q4_K_M.gguf")
        assert profile is QWEN3_5_4B_ROUTER_PROFILE

    def test_qwen2_5_3b_returns_default(self) -> None:
        # No dedicated Qwen2.5-3B profile; falls through to default
        """Test qwen2 5 3b returns default."""
        profile = get_profile_for_filename("Qwen2.5-3B-Instruct-Q4_K_M.gguf")
        assert profile is DEFAULT_PROFILE

    def test_qwen3_14b_detected(self) -> None:
        # Qwen3 14B has a dedicated analysis profile
        """Test qwen3 14b detected."""
        profile = get_profile_for_filename("Qwen3-14B-Q4_K_M.gguf")
        assert profile is QWEN3_14B_PROFILE

    def test_qwen3_8b_returns_default(self) -> None:
        """Test qwen3 8b returns default."""
        profile = get_profile_for_filename("Qwen3-8B-Q5_K_M.gguf")
        assert profile is DEFAULT_PROFILE

    def test_unknown_returns_default(self) -> None:
        """Test unknown returns default."""
        profile = get_profile_for_filename("custom-model.gguf")
        assert profile is DEFAULT_PROFILE
        assert profile.name == "Unknown (ChatML default)"

    def test_llama_returns_default(self) -> None:
        # Llama is no longer a supported profile; use default
        """Test llama returns default."""
        profile = get_profile_for_filename("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        assert profile is DEFAULT_PROFILE

    def test_mistral_nemo_returns_default(self) -> None:
        # Mistral Nemo is no longer a supported profile; falls through to default
        """Test mistral nemo returns default."""
        profile = get_profile_for_filename("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf")
        assert profile is DEFAULT_PROFILE

    def test_phi4_returns_default(self) -> None:
        """Test phi4 returns default."""
        profile = get_profile_for_filename("Phi-4-mini-reasoning-Q8_0.gguf")
        assert profile is DEFAULT_PROFILE

    def test_gemma_returns_default(self) -> None:
        """Test gemma returns default."""
        profile = get_profile_for_filename("Gemma-2-9B-It-Q4_K_M.gguf")
        assert profile is DEFAULT_PROFILE


# ==============================================================================
# Qwen3.6 35B A3B Profile (primary large model)
# ==============================================================================


class TestQwen3535BA3BProfile:
    """Class docstring."""
    @pytest.fixture
    def profile(self) -> ModelProfile:
        """Profile."""
        return QWEN3_6_35B_A3B_PROFILE

    def test_identity(self, profile: ModelProfile) -> None:
        """Test identity."""
        assert profile.name == "Qwen3.6 35B A3B"
        assert profile.family == ModelFamily.CHATML
        assert profile.supports_think_blocks is True

    def test_reasoning_disabled(self, profile: ModelProfile) -> None:
        """Test reasoning disabled."""
        assert profile.reasoning_mode == ReasoningMode.NEVER
        assert profile.get_reasoning_enabled("simple") is False
        assert profile.get_reasoning_enabled("focused") is False
        assert profile.get_reasoning_enabled("coverage") is False

    def test_chat_template_kwargs(self, profile: ModelProfile) -> None:
        """Test chat template kwargs."""
        assert profile.chat_template_kwargs == {"enable_thinking": False}
        assert profile.no_think_token is None

    def test_prompt_format_always_native(self, profile: ModelProfile) -> None:
        """Test prompt format always native."""
        assert profile.get_prompt_format("simple") == PromptFormat.NATIVE_GGUF
        assert profile.get_prompt_format("focused") == PromptFormat.NATIVE_GGUF
        assert profile.get_prompt_format("coverage") == PromptFormat.NATIVE_GGUF

    def test_max_tokens(self, profile: ModelProfile) -> None:
        """Test max tokens."""
        assert profile.get_max_tokens("simple") == 3072
        assert profile.get_max_tokens("focused") == 3072
        assert profile.get_max_tokens("coverage") == 3072

    def test_retrieval_top_k_fields(self, profile: ModelProfile) -> None:
        """Test retrieval top k fields."""
        assert profile.retrieval_top_k_candidates > 0
        assert profile.retrieval_top_k_final > 0

    def test_requested_tuning_values(self, profile: ModelProfile) -> None:
        """Test requested tuning values."""
        assert profile.max_tokens == 3072
        assert profile.coverage_top_k == 18
        assert profile.timeout_seconds == 900
        assert profile.context_length == 24576
        assert profile.temperature == 0.2
        assert profile.rag_top_k == 10
        assert profile.rag_max_score == 0.90
        assert profile.rag_context_ratio == 0.65
        assert profile.rag_rerank_min_score == 0.10
        assert profile.retrieval_top_k_final == 12

    def test_stop_sequences_include_chatml(self, profile: ModelProfile) -> None:
        """Test stop sequences include chatml."""
        stops = profile.get_stop_sequences(reasoning_enabled=True)
        assert "<|im_end|>" in stops
        assert "<|im_start|>" in stops
        assert "<|endoftext|>" in stops

    def test_stop_sequences_no_reasoning_does_not_stop_on_think(
        self, profile: ModelProfile
    ) -> None:
        """Test stop sequences no reasoning does not stop on think."""
        no_reasoning = profile.get_stop_sequences(reasoning_enabled=False)
        assert "<think>" not in no_reasoning

    def test_prepare_messages_does_not_append_no_think(self, profile: ModelProfile) -> None:
        """Test prepare messages does not append no think."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is X?"},
        ]
        result = profile.prepare_messages(messages, "simple")
        assert result[-1]["content"] == messages[-1]["content"]


# ==============================================================================
# Qwen3.5 9B Profile (non-thinking template mode)
# ==============================================================================


class TestQwen359BProfile:
    """Class docstring."""
    @pytest.fixture
    def profile(self) -> ModelProfile:
        """Profile."""
        return QWEN3_5_9B_PROFILE

    def test_reasoning_disabled_by_profile(self, profile: ModelProfile) -> None:
        """Test reasoning disabled by profile."""
        assert profile.reasoning_mode == ReasoningMode.NEVER
        assert profile.no_think_token is None

    def test_sampling_defaults_for_non_thinking_mode(self, profile: ModelProfile) -> None:
        """Test sampling defaults for non thinking mode."""
        assert profile.temperature == 0.7
        assert profile.top_p == 0.8

    def test_template_kwargs_disable_thinking(self, profile: ModelProfile) -> None:
        """Test template kwargs disable thinking."""
        assert profile.chat_template_kwargs == {"enable_thinking": False}

    def test_prepare_messages_does_not_append_no_think_token(self, profile: ModelProfile) -> None:
        """Test prepare messages does not append no think token."""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Summarize the indexed evidence."},
        ]
        result = profile.prepare_messages(messages, "focused")
        assert result[-1]["content"] == messages[-1]["content"]


# ==============================================================================
# ==============================================================================
# Default Profile
# ==============================================================================


class TestDefaultProfile:
    """Class docstring."""
    def test_conservative_settings(self) -> None:
        """Test conservative settings."""
        assert DEFAULT_PROFILE.supports_think_blocks is True
        assert DEFAULT_PROFILE.reasoning_mode == ReasoningMode.FOCUSED_ONLY

    def test_retrieval_top_k_standard(self) -> None:
        """Test retrieval top k standard."""
        assert DEFAULT_PROFILE.retrieval_top_k_candidates == 25
        assert DEFAULT_PROFILE.retrieval_top_k_final == 12

    def test_no_think_token(self) -> None:
        # Default also supports /no_think (safe for ChatML models)
        """Test no think token."""
        assert DEFAULT_PROFILE.no_think_token == "/no_think"

    def test_stop_sequences_returned_when_no_reasoning(self) -> None:
        """Test stop sequences returned when no reasoning."""
        stops = DEFAULT_PROFILE.get_stop_sequences(reasoning_enabled=False)
        assert isinstance(stops, list)
        assert len(stops) > 0


# ==============================================================================
# ModelProfile Methods (generic)
# ==============================================================================


class TestModelProfileMethods:
    """Class docstring."""
    def test_get_max_tokens_unknown_type_returns_focused(self) -> None:
        """Test get max tokens unknown type returns focused."""
        profile = QWEN3_6_35B_A3B_PROFILE
        assert profile.get_max_tokens("unknown") == profile.max_tokens

    def test_get_prompt_format_unknown_type_returns_default(self) -> None:
        """Test get prompt format unknown type returns default."""
        profile = QWEN3_6_35B_A3B_PROFILE
        assert profile.get_prompt_format("unknown") == profile.prompt_format

    def test_get_timeout_seconds_scales_for_agent_mode(self) -> None:
        """Test get timeout seconds scales for agent mode."""
        profile = QWEN3_6_35B_A3B_PROFILE
        assert profile.get_timeout_seconds("focused") == profile.timeout_seconds
        assert profile.get_timeout_seconds("focused", agent_mode=True) == profile.timeout_seconds * 2

    def test_prepare_messages_no_mutation(self) -> None:
        """Test prepare messages no mutation."""
        messages = [{"role": "user", "content": "Test"}]
        _ = QWEN3_6_35B_A3B_PROFILE.prepare_messages(messages, "simple")
        assert messages[0]["content"] == "Test"  # Original not mutated

    def test_to_display_dict_contains_all_keys(self) -> None:
        """Test to display dict contains all keys."""
        display = QWEN3_6_35B_A3B_PROFILE.to_display_dict()
        expected_keys = {
            "name",
            "family",
            "supports_reasoning",
            "reasoning_mode",
            "max_tokens",
            "coverage_top_k",
            "min_tokens_coverage",
            "prompt_format",
            "coverage_prompt_format",
            "context_length",
            "temperature",
            "top_p",
            "rag_top_k",
            "retrieval_top_k_candidates",
            "retrieval_top_k_final",
            "rag_top_k_simple",
            "rag_top_k_focused",
            "rag_top_k_coverage",
            "rag_max_score",
            "rag_context_ratio",
            "rag_rerank_min_score",
            "timeout_seconds",
        }
        assert expected_keys == set(display.keys())

    def test_default_model_uses_qwen3_5_35b_profile(self) -> None:
        # Default config points at Qwen3.6 35B A3B
        """Test default model uses qwen3 5 35b profile."""
        from informity.config import _DEFAULT_LLM_MODEL_FILENAME

        profile = get_profile_for_filename(_DEFAULT_LLM_MODEL_FILENAME)
        assert profile is QWEN3_6_35B_A3B_PROFILE


# ==============================================================================
# get_retrieval_top_k (override point for adaptive tuning)
# ==============================================================================


class TestGetRetrievalTopK:
    """Class docstring."""
    def test_returns_profile_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns profile values."""
        _ = monkeypatch

        profile = get_profile()
        expected_focused = profile.rag_top_k_focused or profile.retrieval_top_k_final
        expected_coverage = profile.rag_top_k_coverage or profile.coverage_top_k
        assert get_retrieval_top_k("focused") == expected_focused
        assert get_retrieval_top_k("coverage") == expected_coverage


class TestProviderAwareProfileSelection:
    """Class docstring."""
    def test_ollama_unknown_model_uses_ollama_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ollama unknown model uses ollama default."""
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_provider", "ollama")
        monkeypatch.setattr(
            "informity.llm.model_adapter.settings.llm_model_id", "custom-unknown:latest"
        )
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_filename", "")
        profile = get_profile()
        assert profile is OLLAMA_DEFAULT_PROFILE
        assert profile.reasoning_mode == ReasoningMode.NEVER
        assert profile.no_think_token is None

    def test_ollama_known_model_id_maps_to_existing_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ollama known model id maps to existing profile."""
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_provider", "ollama")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_id", "qwen3:14b")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_filename", "")
        profile = get_profile()
        assert profile is QWEN3_14B_PROFILE

    def test_ollama_qwen36_35b_maps_to_35b_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ollama qwen36 35b maps to 35b profile."""
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_provider", "ollama")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_id", "qwen3.6:35b")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_filename", "")
        profile = get_profile()
        assert profile is QWEN3_6_35B_A3B_PROFILE

    def test_ollama_qwen35_9b_maps_to_9b_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ollama qwen35 9b maps to 9b profile."""
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_provider", "ollama")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_id", "qwen3.5:9b")
        monkeypatch.setattr("informity.llm.model_adapter.settings.llm_model_filename", "")
        profile = get_profile()
        assert profile is QWEN3_5_9B_PROFILE


class TestOllamaAliasInference:
    """Class docstring."""
    def test_infer_model_id_from_ollama_model_exact(self) -> None:
        """Test infer model id from ollama model exact."""
        assert infer_model_id_from_ollama_model("qwen3.6:35b") == "qwen3.6:35b"
        assert infer_model_id_from_ollama_model("qwen3:14b") == "qwen3:14b"
        assert infer_model_id_from_ollama_model("qwen3.5:9b") == "qwen3.5:9b"

    def test_infer_model_id_from_ollama_model_variant_tag(self) -> None:
        """Test infer model id from ollama model variant tag."""
        assert infer_model_id_from_ollama_model("qwen3.6:35b-q4_k_m") is None

    def test_infer_model_id_from_ollama_model_unknown(self) -> None:
        """Test infer model id from ollama model unknown."""
        assert infer_model_id_from_ollama_model("mistral:latest") is None
