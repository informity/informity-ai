# ==============================================================================
# Informity AI — Chat Mode Utilities
# ==============================================================================

def normalize_chat_mode(chat_mode: str | None) -> str:
    """Normalize chat mode text for internal comparisons."""
    return str(chat_mode or '').strip().lower()
