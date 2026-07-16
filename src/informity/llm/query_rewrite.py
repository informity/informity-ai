"""Module for llm query rewrite."""

from __future__ import annotations


def build_followup_retrieval_query(
    *,
    normalized_question: str,
    previous_user_question: str,
    max_chars_per_turn: int,
    max_query_chars: int,
) -> str:
    """Build followup retrieval query."""
    previous_user = str(previous_user_question or "")
    rewritten_query = (
        f"{normalized_question}\n\nFollow-up context:\n"
        f"- Previous user question: {previous_user[:max_chars_per_turn]}"
    )
    return rewritten_query[:max_query_chars]


__all__ = ["build_followup_retrieval_query"]
