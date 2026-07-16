"""Token counting helpers shared by the prompt builder and diagnostics."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """encoding."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate token count for the supplied text."""
    value = str(text or "")
    if not value:
        return 0
    try:
        return len(_encoding().encode(value))
    except (KeyError, LookupError, ValueError):
        return max(1, len(value) // 4)
