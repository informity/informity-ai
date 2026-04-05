# ==============================================================================
# Informity AI — Chat Mode Utilities
# ==============================================================================

_ALLOWED_CHAT_MODES = {'assistant', 'researcher'}
_DEFAULT_CHAT_MODE = 'researcher'


def normalize_chat_mode(chat_mode: str | None) -> str:
    """Normalize chat mode text for internal comparisons."""
    return str(chat_mode or '').strip().lower()


def resolve_chat_mode(chat_mode: str | None) -> str:
    """
    Resolve requested mode to a valid mode with researcher fallback.
    """
    normalized = normalize_chat_mode(chat_mode)
    if normalized in _ALLOWED_CHAT_MODES:
        return normalized
    return _DEFAULT_CHAT_MODE


def is_assistant_mode(chat_mode: str | None) -> bool:
    return resolve_chat_mode(chat_mode) == 'assistant'
