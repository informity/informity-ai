from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding('cl100k_base')


def count_tokens(text: str) -> int:
    value = str(text or '')
    if not value:
        return 0
    try:
        return len(_encoding().encode(value))
    except Exception:  # noqa: BLE001 - keep token counting non-fatal
        return max(1, len(value) // 4)
