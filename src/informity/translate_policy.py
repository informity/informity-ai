# ==============================================================================
# Informity AI — Translation Policy
# Single source of truth for all translation constants and prompt fragments.
# Route handlers and workers must import from here; no inline magic values.
# ==============================================================================

"""Translation policy constants used by the translate workflow."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Provider / entity
# ---------------------------------------------------------------------------

TRANSLATE_PROVIDER = "translate.local"
TRANSLATE_ENTITY_TYPE = "file"

# Directory name under app_data_dir for translate uploads (mirrors upload.local)
TRANSLATE_STORAGE_DIRNAME = "storage/translate"

# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

# Input tokens per section call.
# Calibrated for observed throughput of 5 tok/s (Qwen3.6 35B A3B profile):
#   1000 input × 1.35 expansion = 1350 output ÷ 5 tok/s = 270s < 360s timeout ✓
# Smaller than the original 6K target but reliably fits within hardware capacity.
TRANSLATE_BATCH_TARGET_TOKENS = 1000

# First N source tokens used for glossary extraction.
TRANSLATE_GLOSSARY_INPUT_TOKENS = 2000

# Maximum terminology terms to extract.
TRANSLATE_GLOSSARY_TERM_COUNT = 25

# Max tokens the glossary LLM call may generate.
TRANSLATE_GLOSSARY_MAX_TOKENS = 512

# Temperature for glossary extraction (low = deterministic JSON).
TRANSLATE_GLOSSARY_TEMPERATURE = 0.1

# Temperature for translation calls (slightly higher for natural fluency).
TRANSLATE_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Timeouts and retry
# ---------------------------------------------------------------------------

TRANSLATE_SECTION_RETRY_MAX = 1  # retries per section on failure/timeout
TRANSLATE_SECTION_TIMEOUT_S = 360  # per-section generation wall-clock (seconds)
# = TRANSLATE_CALL_MAX_TOKENS / 5 tok/s = 1800/5 = 360s ✓
TRANSLATE_GLOSSARY_TIMEOUT_S = 150  # glossary timeout: 512 max tokens / 5 tok/s = 102s + buffer
TRANSLATE_JOB_STALL_S = 600  # no section has *started* for N seconds → stalled
# resets at section START so active sections never trigger it
TRANSLATE_JOB_MAX_RUNTIME_S = 7200  # hard ceiling: 2 hours

# On retry, cap section input to this many tokens (a tiny stub to confirm the pipeline works).
TRANSLATE_RETRY_TOKEN_CAP = 400

# ---------------------------------------------------------------------------
# Upload / lifecycle
# ---------------------------------------------------------------------------

TRANSLATE_CLEANUP_AGE_HOURS = 24  # sweep deletes translate.local files older than this
TRANSLATE_SOFT_PAGE_LIMIT = 50  # warn user; does not block
TRANSLATE_SOFT_SECTION_LIMIT = 25  # also warn when section count exceeds this.

# ---------------------------------------------------------------------------
# Timing estimate (for pre-flight estimate endpoint)
# ---------------------------------------------------------------------------

TRANSLATE_AVG_TOKENS_PER_PAGE = 650  # words ≈ 250/page × 1.3 token/word × ~2 pages/block
TRANSLATE_AVG_SECTION_SECONDS = 20  # measured: ~17s/section at 1K tokens on Qwen3.6 35B A3B

# ---------------------------------------------------------------------------
# Tone instructions injected into the translation system prompt
# Keys match the `tone` request field.
# ---------------------------------------------------------------------------

TONE_INSTRUCTIONS: dict[str, str] = {
    "natural": ("Use natural, fluent {language}. Prioritise readability over literal accuracy."),
    "literal": (
        "Translate into {language} as literally as possible. Preserve sentence structure "
        "and word order where grammatically permissible."
    ),
    "formal": (
        "Use formal, professional {language} register appropriate for business "
        "or academic contexts."
    ),
}

# Per-tone generation temperature.
# literal: near-zero for maximum determinism and fidelity.
# formal:  low variance for consistent register.
# natural: slightly higher to allow idiomatic phrasing.
TONE_TEMPERATURES: dict[str, float] = {
    "literal": 0.05,
    "formal": 0.1,
    "natural": 0.2,
}

# Approximate character count taken from the tail of the previous translated
# section and prepended as context for the next section's prompt.  Keeps the
# model aware of how the previous paragraph ended so it can maintain register,
# terminology, and narrative flow across section boundaries.
TRANSLATE_PREV_CONTEXT_CHARS = 300
