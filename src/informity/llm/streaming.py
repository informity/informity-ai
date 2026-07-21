# ==============================================================================
# Informity AI — Streaming (v2)
# Minimal streaming function, no post-processing bandaids
# ==============================================================================

"""Module for llm streaming."""

from collections.abc import AsyncGenerator

from informity.llm.engine import llm_engine


async def stream_llm(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 1.0,
    timeout_seconds: float | None = None,
    stop_sequences: list[str] | None = None,
    chat_template_kwargs_override: dict[str, object] | None = None,
) -> AsyncGenerator[str | tuple[str, object]]:
    # Stream LLM response. Minimal post-processing.
    # Stop sequences prevent model-specific artifacts (Chinese prompts, reasoning leaks, etc.)
    """Stream llm."""
    async for token in llm_engine.generate_stream(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_seconds=timeout_seconds,
        stop=stop_sequences or [],
        chat_template_kwargs_override=chat_template_kwargs_override,
    ):
        yield token
