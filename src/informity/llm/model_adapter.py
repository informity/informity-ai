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

# ==============================================================================
# Enums
# ==============================================================================

class ModelFamily(StrEnum):
    """Chat template family. Determines structural stop sequences."""
    CHATML  = 'chatml'      # Qwen, DeepSeek, Phi (<|im_start|>)
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
# NOTE: These fire inside <think> blocks when model reasons about insufficient context,
# causing premature generation cutoff. Removed from Qwen3 30B profile (app compliance).
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
    max_tokens_simple:   int = 1024
    max_tokens_focused:  int = 2048
    max_tokens_coverage: int = 2048
    max_tokens_analysis: int | None = None
    # Research mode is the highest-depth response mode. Distinct values confirmed 2026-03-18:
    # max_tokens_research > max_tokens_analysis (more content), timeout_seconds_research > analysis
    # (allow longer generation), top_k_research >= top_k_analysis (wider evidence pool),
    # rag_context_ratio_research < analysis (allocate more context budget to documents).
    # Prompt: prompt_builder.py injects _RESEARCH_MODE_PROMPT_ADDENDUM when response_mode='research'.
    # Fallback: all four fields fall back to analysis/base values when None or 0.
    max_tokens_research: int | None = None
    coverage_top_k:      int = 25    # Chunks for coverage queries
    top_k_analysis:      int | None = None
    top_k_research:      int | None = None
    min_tokens_coverage: int = 400   # Suppress EOS for first N tokens on coverage

    # -- Per-query-type top_k overrides ----------------------------------------
    # When set (> 0), these override rag_top_k for the specific query type.
    # Defaults of 0 mean "fall back to rag_top_k".
    # Suggested calibration: simple → 6, focused → 12, coverage → 24.
    rag_top_k_simple:   int = 0   # 0 = use rag_top_k
    rag_top_k_focused:  int = 0   # 0 = use rag_top_k
    rag_top_k_coverage: int = 0   # 0 = use coverage_top_k

    # -- Timeout configuration (model-specific) --------------------------------
    timeout_seconds_simple:   int = 120   # Wall-clock timeout for simple queries
    timeout_seconds_focused:  int = 180   # Wall-clock timeout for focused queries
    timeout_seconds_coverage: int = 240   # Wall-clock timeout for coverage queries (larger models need more time)
    timeout_seconds_analysis: int | None = None
    timeout_seconds_research: int | None = None

    # -- Model-specific tuning (profile-controlled, read-only in UI) -----------
    context_length: int   = 16384   # Max context window (model architecture limit)
    generation_tokens_per_second: float = 12.0  # Runtime budget estimate baseline used by generation runtime.
    temperature:    float  = 0.1     # Sampling temperature (0 = deterministic)
    top_p:          float  = 1.0    # Nucleus sampling (1.0 = disabled; e.g. 0.95 for R1)
    rag_top_k:      int   = 18      # Chunks to retrieve before filtering

    # -- RAG retrieval tuning (model-specific optimal values) ------------------
    rag_max_score:            float = 0.95  # Max L2 distance for relevant chunk (lower = stricter)
    rag_context_ratio:        float = 0.75  # Share of prompt budget for context (rest for history)
    rag_context_ratio_analysis: float | None = None
    rag_context_ratio_research: float | None = None
    coverage_candidate_multiplier: int = 3  # Candidate pool scale for coverage retrieval.
    coverage_min_candidates: int = 50       # Minimum candidate pool size for coverage retrieval.

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

    # -- Public API ------------------------------------------------------------
    supported_modes: tuple[str, ...] = ('analysis',)

    def get_stop_sequences(self, reasoning_enabled: bool) -> list[str]:
        """Return stop sequences, with extras appended when reasoning is off."""
        stops = list(self.stop_sequences)
        if not reasoning_enabled:
            stops.extend(self.stop_sequences_no_reasoning)
        return stops

    def get_max_tokens(self, query_type: str) -> int:
        """Return max_tokens for 'simple', 'focused', or 'coverage' query type."""
        if query_type == 'simple':
            return self.max_tokens_simple
        if query_type == 'coverage':
            return self.max_tokens_coverage
        return self.max_tokens_focused

    def get_timeout_seconds(self, query_type: str) -> int:
        """Return timeout_seconds for 'simple', 'focused', or 'coverage' query type."""
        if query_type == 'simple':
            return self.timeout_seconds_simple
        if query_type == 'coverage':
            return self.timeout_seconds_coverage
        return self.timeout_seconds_focused

    def get_mode_max_tokens(self, query_type: str, response_mode: str) -> int:
        base = self.get_max_tokens(query_type)
        mode = (response_mode or 'analysis').strip().lower()
        if mode == 'research':
            if isinstance(self.max_tokens_research, int) and self.max_tokens_research > 0:
                return self.max_tokens_research
            if isinstance(self.max_tokens_analysis, int) and self.max_tokens_analysis > 0:
                return self.max_tokens_analysis
            return base
        if mode == 'analysis':
            if isinstance(self.max_tokens_analysis, int) and self.max_tokens_analysis > 0:
                return self.max_tokens_analysis
            return base
        return base

    def get_mode_timeout_seconds(self, query_type: str, response_mode: str) -> int:
        base = self.get_timeout_seconds(query_type)
        mode = (response_mode or 'analysis').strip().lower()
        if mode == 'research':
            if isinstance(self.timeout_seconds_research, int) and self.timeout_seconds_research > 0:
                return self.timeout_seconds_research
            if isinstance(self.timeout_seconds_analysis, int) and self.timeout_seconds_analysis > 0:
                return self.timeout_seconds_analysis
            return base
        if mode == 'analysis':
            if isinstance(self.timeout_seconds_analysis, int) and self.timeout_seconds_analysis > 0:
                return self.timeout_seconds_analysis
            return base
        return base

    def get_mode_top_k(self, response_mode: str, base_top_k: int) -> int:
        mode = (response_mode or 'analysis').strip().lower()
        if mode == 'research':
            if isinstance(self.top_k_research, int) and self.top_k_research > 0:
                return self.top_k_research
            if isinstance(self.top_k_analysis, int) and self.top_k_analysis > 0:
                return self.top_k_analysis
            return base_top_k
        if mode == 'analysis':
            if isinstance(self.top_k_analysis, int) and self.top_k_analysis > 0:
                return self.top_k_analysis
            return base_top_k
        return base_top_k

    def get_mode_context_ratio(self, response_mode: str) -> float:
        mode = (response_mode or 'analysis').strip().lower()
        if mode == 'research':
            if isinstance(self.rag_context_ratio_research, float) and self.rag_context_ratio_research > 0:
                return self.rag_context_ratio_research
            if isinstance(self.rag_context_ratio_analysis, float) and self.rag_context_ratio_analysis > 0:
                return self.rag_context_ratio_analysis
            return self.rag_context_ratio
        if mode == 'analysis':
            if isinstance(self.rag_context_ratio_analysis, float) and self.rag_context_ratio_analysis > 0:
                return self.rag_context_ratio_analysis
            return self.rag_context_ratio
        return self.rag_context_ratio

    def get_research_fallback_fields(self) -> list[str]:
        """Return research fields that will fall back to analysis/base values."""
        missing: list[str] = []
        if not isinstance(self.max_tokens_research, int) or self.max_tokens_research <= 0:
            missing.append('max_tokens_research')
        if not isinstance(self.timeout_seconds_research, int) or self.timeout_seconds_research <= 0:
            missing.append('timeout_seconds_research')
        if not isinstance(self.top_k_research, int) or self.top_k_research <= 0:
            missing.append('top_k_research')
        if not isinstance(self.rag_context_ratio_research, float) or self.rag_context_ratio_research <= 0:
            missing.append('rag_context_ratio_research')
        return missing

    def get_reasoning_enabled(self, query_type: str) -> bool:
        """Whether reasoning (<think> blocks) should be enabled for this query type."""
        if self.reasoning_mode == ReasoningMode.ALWAYS:
            return query_type != 'simple'
        if self.reasoning_mode == ReasoningMode.FOCUSED_ONLY:
            return query_type == 'focused'
        return False

    def get_prompt_format(self, query_type: str) -> PromptFormat:
        """Return the prompt format for the given query type."""
        if query_type == 'coverage':
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
            'supported_modes':         list(self.supported_modes),
            'supports_reasoning':      self.supports_think_blocks,
            'reasoning_mode':          _reasoning_labels.get(self.reasoning_mode, str(self.reasoning_mode)),
            'max_tokens_simple':       self.max_tokens_simple,
            'max_tokens_focused':      self.max_tokens_focused,
            'max_tokens_coverage':     self.max_tokens_coverage,
            'max_tokens_analysis':     self.max_tokens_analysis or self.max_tokens_coverage,
            'max_tokens_research':     (
                self.max_tokens_research
                or self.max_tokens_analysis
                or self.max_tokens_coverage
            ),
            'coverage_top_k':          self.coverage_top_k,
            'top_k_analysis':          self.top_k_analysis or self.rag_top_k,
            'top_k_research':          self.top_k_research or self.top_k_analysis or self.rag_top_k,
            'min_tokens_coverage':     self.min_tokens_coverage,
            'prompt_format':           _format_labels.get(self.prompt_format, str(self.prompt_format)),
            'coverage_prompt_format':  _format_labels.get(self.coverage_prompt_format, str(self.coverage_prompt_format)),
            'context_length':          self.context_length,
            'temperature':             self.temperature,
            'top_p':                    self.top_p,
            'rag_top_k':               self.rag_top_k,
            'rag_top_k_simple':        self.rag_top_k_simple or self.rag_top_k,
            'rag_top_k_focused':       self.rag_top_k_focused or self.rag_top_k,
            'rag_top_k_coverage':      self.rag_top_k_coverage or self.coverage_top_k,
            'rag_max_score':           self.rag_max_score,
            'rag_context_ratio':       self.rag_context_ratio,
            'rag_context_ratio_analysis': self.rag_context_ratio_analysis or self.rag_context_ratio,
            'rag_context_ratio_research': (
                self.rag_context_ratio_research
                or self.rag_context_ratio_analysis
                or self.rag_context_ratio
            ),
            'timeout_seconds_simple':   self.timeout_seconds_simple,
            'timeout_seconds_focused':  self.timeout_seconds_focused,
            'timeout_seconds_coverage': self.timeout_seconds_coverage,
            'timeout_seconds_analysis': self.timeout_seconds_analysis or self.timeout_seconds_coverage,
            'timeout_seconds_research': (
                self.timeout_seconds_research
                or self.timeout_seconds_analysis
                or self.timeout_seconds_coverage
            ),
        }

    def prepare_messages(
        self,
        messages:   list[dict[str, str]],
        query_type: str,
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
                if m['role'] == 'user':
                    m['content'] += f'\n{self.no_think_token}'
                    break

        return messages


# ==============================================================================
# Model Profiles — one frozen instance per supported model
# ==============================================================================

# -- Qwen3 14B Instruct --------------------------------------------------------
# Balanced profile for slower hardware: keep Qwen3 behavior with reduced token
# budgets/timeouts versus 30B to improve latency while preserving answer quality.
QWEN3_14B_PROFILE = ModelProfile(
    name              = 'Qwen3 14B',
    family            = ModelFamily.CHATML,
    filename_patterns = ('qwen3-14b', 'qwen-3-14b'),

    supports_think_blocks         = True,
    reasoning_mode                = ReasoningMode.FOCUSED_ONLY,
    no_think_token                = '/no_think',

    prompt_format          = PromptFormat.NATIVE_GGUF,
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens_simple   = 896,
    max_tokens_focused  = 1280,
    max_tokens_coverage = 1536,
    max_tokens_analysis = 3072,
    max_tokens_research = 5120,
    coverage_top_k      = 15,
    top_k_analysis      = 14,
    top_k_research      = 16,
    min_tokens_coverage = 200,

    timeout_seconds_simple   = 140,
    timeout_seconds_focused  = 240,
    timeout_seconds_coverage = 320,
    timeout_seconds_analysis = 450,
    timeout_seconds_research = 600,

    context_length = 16384,
    generation_tokens_per_second = 9.0,
    temperature    = 0.2,
    top_p          = 0.9,
    rag_top_k      = 10,

    rag_max_score            = 0.92,
    rag_context_ratio        = 0.70,
    rag_context_ratio_analysis = 0.68,
    rag_context_ratio_research = 0.62,

    rag_top_k_simple   = 6,   # Simple queries need fewer candidates
    rag_top_k_focused  = 12,  # Focused queries benefit from slightly wider pool
    rag_top_k_coverage = 0,   # Use coverage_top_k (15) as configured

    stop_sequences  = _CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

    strip_meta_commentary = False,
    strip_citations       = True,
    supported_modes       = ('analysis',),
)


# -- Qwen3.5 9B Instruct ------------------------------------------------------
# Mid-size profile tuned for local Q4_K_M inference:
# - Analysis mode only (research mode not supported — rejected with 422)
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

    max_tokens_simple   = 896,
    max_tokens_focused  = 1408,
    max_tokens_coverage = 1792,
    max_tokens_analysis = 3072,
    max_tokens_research = 6144,
    coverage_top_k      = 16,
    top_k_analysis      = 14,
    top_k_research      = 18,
    min_tokens_coverage = 200,

    timeout_seconds_simple   = 130,
    timeout_seconds_focused  = 220,
    timeout_seconds_coverage = 320,
    timeout_seconds_analysis = 420,
    timeout_seconds_research = 720,

    # Qwen3.5 supports long context; 24K is a practical local sweet spot for Q4_K_M.
    context_length = 24576,
    generation_tokens_per_second = 12.0,
    temperature    = 0.7,   # Recommended for non-thinking mode (was 0.15 — too low)
    top_p          = 0.8,   # Recommended for non-thinking mode
    rag_top_k      = 10,

    rag_max_score             = 0.91,
    rag_context_ratio         = 0.68,
    rag_context_ratio_analysis = 0.66,
    rag_context_ratio_research = 0.60,

    rag_top_k_simple   = 6,
    rag_top_k_focused  = 12,
    rag_top_k_coverage = 0,   # Use coverage_top_k (16)

    stop_sequences  = _CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

    strip_meta_commentary = False,
    strip_citations       = True,
    dedupe_insufficient_context_after_stream = True,
    supported_modes       = ('analysis',),

    # Pass enable_thinking=False to the Qwen3.5 GGUF template so it prefills
    # <think>\n\n</think>\n\n (empty think block) instead of <think>\n (forced
    # thinking). Without this, the template always forces thinking mode which
    # breaks GBNF-grammar classification and produces empty answers.
    chat_template_kwargs = {'enable_thinking': False},
)


# -- Qwen3 30B A3B Instruct ----------------------------------------------------
# High-capacity model (30B parameters). Significantly more capable than 14B,
# with better instruction-following, reasoning, and structured output.
#
# For RAG: Reasoning enabled for focused queries (FOCUSED_ONLY). Complex synthesis
# queries (e.g. "list accomplishments of X") need reasoning; ReasoningMode.NEVER
# causes /no_think to be appended but Qwen3 30B A3B still generates empty <think>
# blocks on such queries, leading to false refusals. FOCUSED_ONLY enables proper
# reasoning for focused queries while keeping simple/coverage fast.
#
# Native GGUF template works correctly (standard ChatML <|im_start|>).
# Q4_K_M quantization (~18GB, ~8-12 t/s on M3 Pro/Max). 32K native context;
# we use 24K for RAG (more headroom for complex queries and long context).
#
# Tuned for 30B Q4_K_M: higher token limits, stricter RAG thresholds (better
# model can be more selective), more context budget for answers.
QWEN3_30B_A3B_PROFILE = ModelProfile(
    name              = 'Qwen3 30B A3B',
    family            = ModelFamily.CHATML,
    filename_patterns = ('qwen3-30b-a3b', 'qwen3-30b', 'qwen-3-30b'),

    supports_think_blocks         = True,
    reasoning_mode                = ReasoningMode.FOCUSED_ONLY,  # Focused queries need reasoning; NEVER causes empty <think> → false refusal
    no_think_token                = '/no_think',  # Automatically appended to user message when reasoning disabled

    prompt_format          = PromptFormat.NATIVE_GGUF,  # Qwen3's native template works correctly
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens_simple   = 1024,    # Simple queries (greetings, clarifications) need short answers
    max_tokens_focused  = 1536,    # Focused queries: single-question answers
    max_tokens_coverage = 2048,    # Coverage queries: lists, comparisons, summaries
    max_tokens_analysis = 3072,
    max_tokens_research = 8192,
    coverage_top_k      = 18,       # Reduced from 25: 30B is slower, need to prevent timeout on coverage queries
    top_k_analysis      = 14,
    top_k_research      = 20,
                                    # 18 chunks is still comprehensive but more realistic for generation speed
                                    # Count queries now use SQL path (Fix #1), so coverage_top_k only affects list queries
    min_tokens_coverage = 200,      # Same as 14B: prevent premature stops

    timeout_seconds_simple   = 180,   # 30B is slower than 14B, allow generous headroom
    timeout_seconds_focused  = 280,   # Focused queries with 30B can be long on dense documents
    timeout_seconds_coverage = 420,   # Coverage queries may require large-table synthesis
    timeout_seconds_analysis = 450,
    timeout_seconds_research = 900,

    context_length = 24576,  # 24K for RAG; 30B native 32K, but 24K is ample and faster
    generation_tokens_per_second = 6.0,
    temperature    = 0.2,     # Low for factual extraction; same as 14B
    top_p          = 0.9,     # Slight nucleus sampling for variety
    rag_top_k      = 10,      # Focused retrieval: 10 chunks; reranker quality drops after rank 9

    # RAG tuning: 30B benefits from stricter threshold (0.90) - better model can be more selective
    rag_max_score            = 0.90,  # Stricter than 14B: better model can be more selective
    rag_context_ratio        = 0.65,  # More context budget for complex queries (24K context)
    rag_context_ratio_analysis = 0.65,
    rag_context_ratio_research = 0.60,

    rag_top_k_simple   = 6,
    rag_top_k_focused  = 12,
    rag_top_k_coverage = 0,   # Use coverage_top_k (18)

    # Do not include citation text stops for Qwen3: they can appear inside
    # <think> reasoning and prematurely terminate generation before final answer.
    stop_sequences  = _CHATML_STRUCTURAL + _QWEN_CHINESE_STOPS,

    strip_meta_commentary = False,  # 30B model follows Rule #5 — no need
    strip_citations       = True,   # Keep citation stripping as safety net
    dedupe_insufficient_context_after_stream = True,  # Fallback phrase stops are disabled for this profile
    supported_modes       = ('analysis', 'research'),
)


# -- DeepSeek R1 Distill (14B — quality analysis / diagnostics) ---------------
# Used for LLM-powered diagnostics analysis (diagnostics_llm_model_filename).
# Qwen-based; ChatML. Profile ensures correct prompt format and stop sequences.
# Also used when RAG model is 14B variant (filename matches r1-distill but not 32b).
DEEPSEEK_R1_DISTILL_PROFILE = ModelProfile(
    name              = 'DeepSeek R1 Distill',
    family            = ModelFamily.CHATML,
    filename_patterns = ('deepseek-r1', 'r1-distill'),

    supports_think_blocks = True,
    reasoning_mode       = ReasoningMode.FOCUSED_ONLY,
    no_think_token       = '/no_think',

    prompt_format          = PromptFormat.NATIVE_GGUF,
    coverage_prompt_format = PromptFormat.NATIVE_GGUF,

    max_tokens_simple   = 2048,
    max_tokens_focused  = 4096,   # Quality analysis can produce long JSON
    max_tokens_coverage = 4096,
    max_tokens_analysis = 4096,
    max_tokens_research = 4096,
    coverage_top_k      = 20,
    top_k_analysis      = 20,
    top_k_research      = 20,
    min_tokens_coverage = 200,

    timeout_seconds_simple   = 120,   # Diagnostics analysis: adequate for simple queries
    timeout_seconds_focused  = 180,   # Diagnostics analysis: longer timeout for focused queries
    timeout_seconds_coverage = 240,   # Diagnostics analysis: generous timeout for coverage queries
    timeout_seconds_analysis = 240,
    timeout_seconds_research = 240,

    context_length = 16384,
    generation_tokens_per_second = 12.0,
    temperature    = 0.1,
    top_p          = 0.9,
    rag_top_k      = 18,

    rag_max_score            = 0.90,
    rag_context_ratio        = 0.75,
    rag_context_ratio_analysis = 0.75,
    rag_context_ratio_research = 0.75,

    stop_sequences  = _CHATML_STRUCTURAL + _CITATION + _FALLBACK_PHRASE_STOPS,

    strip_meta_commentary = True,
    strip_citations       = True,
    supported_modes       = ('analysis',),
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

    max_tokens_simple   = 1024,
    max_tokens_focused  = 2048,
    max_tokens_coverage = 2048,
    max_tokens_analysis = 3072,
    max_tokens_research = 3072,
    coverage_top_k      = 15,
    top_k_analysis      = 14,
    top_k_research      = 14,
    min_tokens_coverage = 100,

    timeout_seconds_simple   = 120,   # Conservative default for unknown models
    timeout_seconds_focused  = 150,   # Conservative default for unknown models
    timeout_seconds_coverage = 180,   # Conservative default for unknown models
    timeout_seconds_analysis = 450,
    timeout_seconds_research = 450,

    context_length = 8192,
    generation_tokens_per_second = 12.0,
    temperature    = 0.2,
    rag_top_k      = 12,

    # RAG tuning: Conservative defaults (matching current global settings)
    rag_max_score            = 0.95,
    rag_context_ratio        = 0.75,
    rag_context_ratio_analysis = 0.70,
    rag_context_ratio_research = 0.70,

    stop_sequences  = _CHATML_STRUCTURAL + _CITATION + _FALLBACK_PHRASE_STOPS,

    strip_meta_commentary = True,
    strip_citations       = True,
    supported_modes       = ('analysis',),
)


# ==============================================================================
# Profile Registry — ordered list, first match wins
# ==============================================================================

# Order matters: more specific patterns first. R1 before Qwen3 30B for diagnostics.
_PROFILE_REGISTRY: list[ModelProfile] = [
    DEEPSEEK_R1_DISTILL_PROFILE,   # DeepSeek-R1-Distill-Qwen-14B (diagnostics)
    QWEN3_5_9B_PROFILE,            # Qwen3.5-9B-Q4_K_M (analysis RAG)
    QWEN3_14B_PROFILE,             # Qwen3-14B-Q5_K_M (balanced RAG profile)
    QWEN3_30B_A3B_PROFILE,         # Qwen3-30B-A3B-Q4_K_M (primary RAG)
]


def get_profile_for_filename(filename: str) -> ModelProfile:
    """Match a GGUF filename to a ModelProfile. First match wins (ANY pattern)."""
    name = filename.lower()
    for profile in _PROFILE_REGISTRY:
        if profile.filename_patterns and any(p in name for p in profile.filename_patterns):
            return profile
    return DEFAULT_PROFILE


def get_profile_tokens_per_second(profile_name: str) -> float:
    """
    Resolve deterministic throughput estimate from profile metadata by profile name.
    Falls back to default profile baseline if profile name is unknown.
    """
    normalized = str(profile_name or '').strip().casefold()
    for profile in _PROFILE_REGISTRY:
        if profile.name.casefold() == normalized:
            return max(1.0, float(profile.generation_tokens_per_second))
    if DEFAULT_PROFILE.name.casefold() == normalized:
        return max(1.0, float(DEFAULT_PROFILE.generation_tokens_per_second))
    profile = get_profile_for_filename(str(profile_name or ''))
    return max(1.0, float(profile.generation_tokens_per_second))


def get_profile() -> ModelProfile:
    """Return the ModelProfile for the currently configured LLM model."""
    return get_profile_for_filename(settings.llm_model_filename)


def get_retrieval_top_k(query_type: str, response_mode: str = 'analysis') -> int:
    """
    Return effective top-k for retrieval.

    Resolution order (highest priority first):
    1. Adaptive tuning (corpus-aware override, when enabled).
    2. Per-query-type profile override (rag_top_k_simple/focused/coverage when > 0).
    3. Mode-based profile value (top_k_analysis / top_k_research when set).
    4. Base profile value (rag_top_k for focused/simple; coverage_top_k for coverage).

    All top-k values must come from model profile — no config/env override path.
    """
    from informity.indexer.adaptive_tuning import get_effective_top_k

    profile = get_profile()

    # 1. Adaptive tuning (corpus-aware)
    adaptive = get_effective_top_k(query_type)
    if adaptive is not None:
        return profile.get_mode_top_k(response_mode, adaptive)

    # 2. Per-query-type override (when set > 0 — only for base query types, not mode)
    if query_type == 'simple' and profile.rag_top_k_simple > 0:
        return profile.rag_top_k_simple
    if query_type == 'focused' and profile.rag_top_k_focused > 0:
        return profile.rag_top_k_focused
    if query_type == 'coverage' and profile.rag_top_k_coverage > 0:
        return profile.rag_top_k_coverage

    # 3. Mode-adjusted base value
    mode_base = profile.coverage_top_k if query_type == 'coverage' else profile.rag_top_k
    return profile.get_mode_top_k(response_mode, mode_base)


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


def extract_model_name_from_chat_id(chat_id: str, available_models: list[str] | None = None) -> str | None:
    """
    Extract model name from a chat_id that follows the pattern: eval-{query_id}-{model_name}

    Tooling note:
        Kept for diagnostics/evaluation tooling that parses evaluation chat IDs.
        It may not be imported by primary runtime request paths.

    Args:
        chat_id: Chat ID string (e.g., 'eval-eval-1-doc-totals-Meta-Llama-3.1-8B-Instruct-Q5_K_M')
        available_models: Optional list of known model filenames for matching. If None, discovers models.

    Returns:
        Model filename (with .gguf) if found, or None if not found
    """
    from informity.config import DiagnosticsConstants

    if not chat_id.startswith(DiagnosticsConstants.EVAL_CHAT_ID_PREFIX):
        return None

    # Remove prefix
    prefix = DiagnosticsConstants.EVAL_CHAT_ID_PREFIX
    rest = chat_id[len(prefix):]

    # If available_models not provided, discover them
    if available_models is None:
        available_models = discover_available_models()

    # Try to match against known models (longest match first for specificity)
    # Sort by length descending to match longer model names first
    sorted_models = sorted(available_models, key=len, reverse=True)

    for model_filename in sorted_models:
        model_name = Path(model_filename).stem  # Remove .gguf
        # Check if chat_id ends with this model name
        if rest.endswith(model_name):
            return model_filename

    return None


def get_model_display_name_from_chat_id(chat_id: str, available_models: list[str] | None = None) -> str | None:
    """
    Get the display name for a model from a chat_id.

    Tooling note:
        Companion helper for diagnostics/evaluation tooling that derives model
        display labels from evaluation chat IDs.

    Args:
        chat_id: Chat ID string (e.g., 'eval-eval-1-doc-totals-Meta-Llama-3.1-8B-Instruct-Q5_K_M')
        available_models: Optional list of known model filenames for matching. If None, discovers models.

    Returns:
        Display name from profile (e.g., 'Llama 3.1 8B') or None if model not found
    """
    model_filename = extract_model_name_from_chat_id(chat_id, available_models)
    if model_filename:
        return get_model_display_name(model_filename)
    return None
