# ==============================================================================
# Informity AI — Model Profiles
# Complete per-model configuration: prompt format, reasoning behavior, token
# limits, stop sequences, and post-processing rules. Each supported model has
# its own frozen ModelProfile instance — changes to one model never affect others.
#
# To add a new model:
#   1. Define a ModelProfile constant (e.g. MY_MODEL_PROFILE)
#   2. Add it to _PROFILE_REGISTRY (more specific patterns first)
#   3. Done — rag.py and engine.py use the profile automatically
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from informity.config import settings
from informity.llm.types import ChatRole, QueryType

# ==============================================================================
# Enums
# ==============================================================================

class ModelFamily(StrEnum):
    """Chat template family. Determines structural stop sequences."""
    CHATML  = 'chatml'      # Qwen, Phi (<|im_start|>)
    LLAMA   = 'llama'       # Meta Llama 2/3
    MISTRAL = 'mistral'     # Mistral / Mixtral / Codestral


class PromptFormat(StrEnum):
    """How to render the messages list into a prompt string."""
    NATIVE_GGUF = 'native_gguf'  # Use GGUF's embedded Jinja2 template
    CHATML      = 'chatml'       # Force standard ChatML format


class ReasoningMode(StrEnum):
    """When the model should use <think> reasoning blocks."""
    ALWAYS       = 'always'        # All queries except simple
    FOCUSED_ONLY = 'focused_only'  # Only focused queries (not coverage, not simple)
    NEVER        = 'never'         # Never reason


# ==============================================================================
# Common Stop Sequence Groups (reusable across profiles)
# ==============================================================================

_CHATML_STRUCTURAL = (
    '<|im_end|>',
    '<|im_start|>',
    '<|endoftext|>',
)

_CITATION = (
    '[Source',
    '[ Source',
    '(Source',
    '( Source',
    'Sources:',
    '\nSources',
)

# Qwen3-specific: Chinese accessibility prompts that may leak after answer
_QWEN_CHINESE_STOPS = (
    '无障碍模式',  # "Accessibility mode"
    '请告诉我',    # "Please tell me"
    '\n\n无',      # Start of Chinese paragraph
)

# Fallback phrase stops: prevent repetition of insufficient info message
# NOTE: These can fire inside <think> blocks when model reasons about insufficient context,
# causing premature generation cutoff on some profiles.
_FALLBACK_PHRASE_STOPS = (
    'The available documents do not contain enough information',
    '\nThe available documents',
    'do not contain enough information',
)

# 14B model follows Rule #5 ("no commentary") — no meta-commentary stop sequences


# ==============================================================================
# ModelProfile — single source of truth for model-specific behavior
# ==============================================================================

@dataclass(frozen=True)
class ModelProfile:
    """
    Complete per-model configuration. One frozen instance per supported model.
    All model-specific decisions (format, reasoning, tokens, stops, post-processing)
    are driven by profile fields — no scattered if/else chains in other modules.
    """

    # -- Identity --------------------------------------------------------------
    name:              str                # Human-readable name
    family:            ModelFamily        # Chat template family
    filename_patterns: tuple[str, ...]    # Lowercase substrings; ANY must match

    # -- Reasoning -------------------------------------------------------------
    supports_think_blocks:         bool            = False
    reasoning_mode:                ReasoningMode   = ReasoningMode.NEVER
    no_think_token:                str | None       = None   # Append to user message (e.g. Qwen3 /no_think)

    # -- Prompt format ---------------------------------------------------------
    prompt_format:          PromptFormat = PromptFormat.NATIVE_GGUF
    coverage_prompt_format: PromptFormat = PromptFormat.NATIVE_GGUF

    # -- Token limits ----------------------------------------------------------
    max_tokens:          int = 3072
    coverage_top_k:      int = 25    # Chunks for coverage queries
    min_tokens_coverage: int = 400   # Suppress EOS for first N tokens on coverage

    # -- Per-query-type top_k overrides ----------------------------------------
    # When set (> 0), these override rag_top_k for the specific query type.
    # Defaults of 0 mean "fall back to rag_top_k".
    # Suggested calibration: simple → 6, focused → 12, coverage → 24.
    rag_top_k_simple:   int = 0   # 0 = use rag_top_k
    rag_top_k_focused:  int = 0   # 0 = use rag_top_k
    rag_top_k_coverage: int = 0   # 0 = use coverage_top_k

    # -- Timeout configuration (model-specific) --------------------------------
    timeout_seconds: int = 450   # Wall-clock timeout for generation

    # -- Model-specific tuning (profile-controlled, read-only in UI) -----------
    context_length: int   = 16384   # Max context window (model architecture limit)
    generation_tokens_per_second: float = 12.0  # Runtime budget estimate baseline used by generation runtime.
    temperature:    float  = 0.1     # Sampling temperature (0 = deterministic)
    top_p:          float  = 1.0    # Nucleus sampling (1.0 = disabled)
    rag_top_k:      int   = 18      # Chunks to retrieve before filtering
    retrieval_top_k_candidates: int = 25  # Candidate pool before reranking
    retrieval_top_k_final: int = 12       # Final parent chunks after reranking

    # -- RAG retrieval tuning (model-specific optimal values) ------------------
    rag_max_score:            float = 0.95  # Max L2 distance for relevant chunk (lower = stricter)
    rag_context_ratio:        float = 0.75  # Share of prompt budget for context (rest for history)

    # -- Stop sequences --------------------------------------------------------
    stop_sequences:              tuple[str, ...] = ()
    stop_sequences_no_reasoning: tuple[str, ...] = ()  # Added when reasoning is off

    # -- Post-processing -------------------------------------------------------
    strip_meta_commentary: bool = True
    strip_citations:       bool = True
    dedupe_insufficient_context_after_stream: bool = False

    # -- Generation template control -------------------------------------------
    # Passed as chat_template_kwargs in the xllamacpp payload. Used for models
    # that control thinking via template variables (e.g. Qwen3.5 enable_thinking)
    # rather than user-message tokens (e.g. Qwen3 /no_think).
    chat_template_kwargs: dict = field(default_factory=dict)

    def get_stop_sequences(self, reasoning_enabled: bool) -> list[str]:
        """Return stop sequences, with extras appended when reasoning is off."""
        stops = list(self.stop_sequences)
        if not reasoning_enabled:
            stops.extend(self.stop_sequences_no_reasoning)
        return stops

    def get_max_tokens(self, query_type: QueryType) -> int:
        """Return single profile max_tokens (query-type agnostic)."""
        _ = query_type
        return self.max_tokens

    def get_timeout_seconds(self, query_type: QueryType) -> int:
        """Return single profile timeout_seconds (query-type agnostic)."""
        _ = query_type
        return self.timeout_seconds

    def get_reasoning_enabled(self, query_type: QueryType) -> bool:
        """Whether reasoning (<think> blocks) should be enabled for this query type."""
        if self.reasoning_mode == ReasoningMode.ALWAYS:
            return query_type != QueryType.SIMPLE
        if self.reasoning_mode == ReasoningMode.FOCUSED_ONLY:
            return query_type == QueryType.FOCUSED
        return False

    def get_prompt_format(self, query_type: QueryType) -> PromptFormat:
        """Return the prompt format for the given query type."""
        if query_type == QueryType.COVERAGE:
            return self.coverage_prompt_format
        return self.prompt_format

    def to_display_dict(self) -> dict:
        """Return profile values for the Settings UI (read-only display)."""
        _reasoning_labels = {
            ReasoningMode.ALWAYS:       'All queries',
            ReasoningMode.FOCUSED_ONLY: 'Focused queries only',
            ReasoningMode.NEVER:        'Off',
        }
        _format_labels = {
            PromptFormat.NATIVE_GGUF: 'Native (GGUF template)',
            PromptFormat.CHATML:      'ChatML',
        }
        return {
            'name':                    self.name,
            'family':                  str(self.family),
            'supports_reasoning':      self.supports_think_blocks,
            'reasoning_mode':          _reasoning_labels.get(self.reasoning_mode, str(self.reasoning_mode)),
            'max_tokens':              self.max_tokens,
            'coverage_top_k':          self.coverage_top_k,
            'min_tokens_coverage':     self.min_tokens_coverage,
            'prompt_format':           _format_labels.get(self.prompt_format, str(self.prompt_format)),
            'coverage_prompt_format':  _format_labels.get(self.coverage_prompt_format, str(self.coverage_prompt_format)),
            'context_length':          self.context_length,
            'temperature':             self.temperature,
            'top_p':                    self.top_p,
            'rag_top_k':               self.rag_top_k,
            'retrieval_top_k_candidates': self.retrieval_top_k_candidates,
            'retrieval_top_k_final':   self.retrieval_top_k_final,
            'rag_top_k_simple':        self.rag_top_k_simple or self.rag_top_k,
            'rag_top_k_focused':       self.rag_top_k_focused or self.rag_top_k,
            'rag_top_k_coverage':      self.rag_top_k_coverage or self.coverage_top_k,
            'rag_max_score':           self.rag_max_score,
            'rag_context_ratio':       self.rag_context_ratio,
            'timeout_seconds':         self.timeout_seconds,
        }

    def prepare_messages(
        self,
        messages:   list[dict[str, str]],
        query_type: QueryType,
    ) -> list[dict[str, str]]:
        """
        Apply model-specific message transformations (e.g. Qwen3 /no_think).
        Returns a potentially modified copy — never mutates the input.
        """
        reasoning_enabled = self.get_reasoning_enabled(query_type)
        messages = [m.copy() for m in messages]

        # Apply /no_think token for models that support it (e.g. Qwen3)
        if not reasoning_enabled and self.no_think_token:
            for m in reversed(messages):
                if m['role'] == ChatRole.USER:
                    m['content'] += f'\n{self.no_think_token}'
                    break

        return messages


# ==============================================================================
# Model Profiles — one frozen instance per supported model
# ==============================================================================

# -- Qwen3 14B Instruct --------------------------------------------------------
# Balanced profile for slower hardware.
# Keep template-level thinking disabled for reliability on lower-memory devices.
QWEN3_14B_PROFILE = ModelProfile(
    name              = 'Qwen3 14B',
    family            = ModelFamily.CHATML,
    filename_patterns = ('qwen3-14b', 'qwen-3-14b'),

    supports_think_blocks         = True,
    reasoning_mode                = ReasoningMode.NEVER,
    no_think_token                = None,

    prompt_format          = PromptFormat.NATIVE_GGUF,
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens         = 3072,
    coverage_top_k      = 15,
    min_tokens_coverage = 200,

    timeout_seconds = 450,

    context_length = 16384,
    generation_tokens_per_second = 9.0,
    temperature    = 0.2,
    top_p          = 0.9,
    rag_top_k      = 10,

    rag_max_score            = 0.92,
    rag_context_ratio        = 0.68,

    rag_top_k_simple   = 6,   # Simple queries need fewer candidates
    rag_top_k_focused  = 12,  # Focused queries benefit from slightly wider pool
    rag_top_k_coverage = 0,   # Use coverage_top_k (15) as configured

    stop_sequences  = _CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

    strip_meta_commentary = False,
    strip_citations       = True,
    chat_template_kwargs  = {'enable_thinking': False},
)


# -- Qwen3.5 9B Instruct ------------------------------------------------------
# Mid-size profile tuned for local Q4_K_M inference:
# - Uses focused-only reasoning to keep simple/coverage responsive
# - Uses native GGUF template (no forced ChatML fallback)
QWEN3_5_9B_PROFILE = ModelProfile(
    name              = 'Qwen3.5 9B',
    family            = ModelFamily.CHATML,
    filename_patterns = ('qwen3.5-9b', 'qwen-3.5-9b', 'qwen3-5-9b'),

    # Qwen3.5 uses enable_thinking template variable, not /no_think user token.
    # Thinking disabled via chat_template_kwargs; reasoning_mode=NEVER because
    # we cannot selectively enable reasoning per-query without token-level control.
    supports_think_blocks = True,
    reasoning_mode        = ReasoningMode.NEVER,
    no_think_token        = None,

    prompt_format          = PromptFormat.NATIVE_GGUF,
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens         = 3072,
    coverage_top_k      = 16,
    min_tokens_coverage = 200,

    timeout_seconds = 420,

    # Align with 14B default for predictable memory behavior on consumer hardware.
    context_length = 16384,
    generation_tokens_per_second = 12.0,
    temperature    = 0.7,   # Recommended for non-thinking mode (was 0.15 — too low)
    top_p          = 0.8,   # Recommended for non-thinking mode
    rag_top_k      = 10,

    rag_max_score             = 0.91,
    rag_context_ratio         = 0.66,

    rag_top_k_simple   = 6,
    rag_top_k_focused  = 12,
    rag_top_k_coverage = 0,   # Use coverage_top_k (16)

    stop_sequences  = _CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

    strip_meta_commentary = False,
    strip_citations       = True,
    dedupe_insufficient_context_after_stream = True,

    # Pass enable_thinking=False to the Qwen3.5 GGUF template so it prefills
    # <think>\n\n</think>\n\n (empty think block) instead of <think>\n (forced
    # thinking). Without this, the template always forces thinking mode which
    # breaks GBNF-grammar classification and produces empty answers.
    chat_template_kwargs = {'enable_thinking': False},
)


def _build_qwen_35b_a3b_profile(*, name: str, filename_patterns: tuple[str, ...]) -> ModelProfile:
    """Shared tuning for Qwen 35B A3B family variants (3.5/3.6)."""
    return ModelProfile(
        name=name,
        family=ModelFamily.CHATML,
        filename_patterns=filename_patterns,

        # Qwen 35B A3B GGUF variants use template-level thinking control.
        # Keep thinking disabled to avoid empty-stream failures when the model
        # consumes generation budget inside hidden reasoning.
        supports_think_blocks=True,
        reasoning_mode=ReasoningMode.NEVER,
        no_think_token=None,

        prompt_format=PromptFormat.NATIVE_GGUF,
        coverage_prompt_format=PromptFormat.NATIVE_GGUF,

        max_tokens=3072,
        coverage_top_k=18,
        min_tokens_coverage=200,

        timeout_seconds=900,

        context_length=24576,
        generation_tokens_per_second=5.0,
        temperature=0.2,
        top_p=0.9,
        rag_top_k=10,

        rag_max_score=0.90,
        rag_context_ratio=0.65,

        retrieval_top_k_final=12,
        rag_top_k_simple=6,
        rag_top_k_focused=12,
        rag_top_k_coverage=0,   # Use coverage_top_k (18)

        stop_sequences=_CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

        strip_meta_commentary=False,
        strip_citations=True,
        chat_template_kwargs={'enable_thinking': False},
    )


# -- Qwen3.6 35B A3B ----------------------------------------------------------
QWEN3_6_35B_A3B_PROFILE = _build_qwen_35b_a3b_profile(
    name='Qwen3.6 35B A3B',
    filename_patterns=('qwen3.6-35b-a3b', 'qwen-3.6-35b-a3b', 'qwen3-6-35b-a3b'),
)


# -- Default profile for unknown models (conservative ChatML) -----------------
DEFAULT_PROFILE = ModelProfile(
    name              = 'Unknown (ChatML default)',
    family            = ModelFamily.CHATML,
    filename_patterns = (),

    supports_think_blocks = True,
    reasoning_mode        = ReasoningMode.FOCUSED_ONLY,
    no_think_token            = '/no_think',  # Safe default; models that don't support it ignore it

    prompt_format          = PromptFormat.NATIVE_GGUF,
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens         = 3072,
    coverage_top_k      = 15,
    min_tokens_coverage = 100,

    timeout_seconds = 450,   # Conservative default for unknown models

    context_length = 8192,
    generation_tokens_per_second = 12.0,
    temperature    = 0.2,
    rag_top_k      = 12,

    # RAG tuning: Conservative defaults (matching current global settings)
    rag_max_score            = 0.95,
    rag_context_ratio        = 0.70,

    stop_sequences  = _CHATML_STRUCTURAL + _CITATION + _FALLBACK_PHRASE_STOPS,

    strip_meta_commentary = True,
    strip_citations       = True,
)


# ==============================================================================
# Profile Registry — ordered list, first match wins
# ==============================================================================

# Order matters: more specific patterns first.
_PROFILE_REGISTRY: list[ModelProfile] = [
    QWEN3_6_35B_A3B_PROFILE,       # Qwen3.6-35B-A3B(-UD)-Q4_K_M
    QWEN3_5_9B_PROFILE,            # Qwen3.5-9B-Q4_K_M (analysis RAG)
    QWEN3_14B_PROFILE,             # Qwen3-14B-Q5_K_M (analysis RAG profile)
]


def get_profile_for_filename(filename: str) -> ModelProfile:
    """Match a GGUF filename to a ModelProfile. First match wins (ANY pattern)."""
    name = filename.lower()
    for profile in _PROFILE_REGISTRY:
        if profile.filename_patterns and any(p in name for p in profile.filename_patterns):
            return profile
    return DEFAULT_PROFILE


def get_profile() -> ModelProfile:
    """Return the ModelProfile for the currently configured LLM model."""
    return get_profile_for_filename(settings.llm_model_filename)


def get_effective_context_length(profile: ModelProfile | None = None) -> int:
    """
    Resolve effective context length for runtime and budgeting.

    Uses the lower of:
    - model profile context length
    - configured llm_context_length (when > 0)
    """
    active_profile = profile or get_profile()
    profile_ctx = max(1, int(active_profile.context_length))
    configured_ctx = int(getattr(settings, 'llm_context_length', 0) or 0)
    if configured_ctx > 0:
        return max(1, min(profile_ctx, configured_ctx))
    return profile_ctx


def get_retrieval_top_k(query_type: QueryType) -> int:
    """Return model-profile-owned final retrieval top-k for the given query type."""
    profile = get_profile()
    # Prefer query-type-specific profile knobs when configured, then fall back to
    # retrieval_top_k_final as the stable default.
    if query_type == QueryType.COVERAGE:
        if int(profile.rag_top_k_coverage) > 0:
            return max(1, int(profile.rag_top_k_coverage))
        return max(1, int(profile.coverage_top_k))
    if query_type == QueryType.FOCUSED and int(profile.rag_top_k_focused) > 0:
        return max(1, int(profile.rag_top_k_focused))
    return max(1, int(profile.retrieval_top_k_final))


# ==============================================================================
# Model Discovery Utilities
# ==============================================================================

def discover_available_models() -> list[str]:
    """
    Discover all .gguf model files in the models directory.

    Returns:
        Sorted list of model filenames (e.g., ['Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf', ...])
    """
    models_dir = settings.models_dir
    if models_dir is None or not models_dir.exists():
        return []

    models: list[str] = []
    for path in models_dir.glob('*.gguf'):
        if path.is_file():
            models.append(path.name)

    return sorted(models)


def get_model_display_name(filename: str) -> str:
    """
    Get the display name for a model filename using its profile.

    Args:
        filename: Model filename (e.g., 'Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf')

    Returns:
        Display name from profile (e.g., 'Llama 3.1 8B') or filename stem if no profile match
    """
    profile = get_profile_for_filename(filename)
    if profile != DEFAULT_PROFILE:
        return profile.name
    # Fallback: use filename stem (without .gguf)
    return Path(filename).stem
