# ==============================================================================
# Informity AI — Extraction Boundary Rules
# Shared helpers for preserving semantic block boundaries across extractors.
# ==============================================================================

"""Module for scanner extractors boundary rules."""

from __future__ import annotations

from collections.abc import Iterable

STRUCTURED_BLOCK_SEPARATOR = "\n\n"


def normalize_structured_block(text: str) -> str:
    """Normalize structured block."""
    return text.rstrip()


def join_structured_blocks(
    blocks: Iterable[str], *, separator: str = STRUCTURED_BLOCK_SEPARATOR
) -> str:
    """Join structured blocks."""
    normalized_blocks = [
        normalize_structured_block(block) for block in blocks if normalize_structured_block(block)
    ]
    return separator.join(normalized_blocks).strip()
