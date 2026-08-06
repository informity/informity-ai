# ==============================================================================
# Informity AI — Chat Message Translation Helper
# Shared translation prompt and LLM streaming helper for chat reply translation.
# ==============================================================================

"""Module for api chat translation."""

from __future__ import annotations

import asyncio

from informity.llm.engine import llm_engine
from informity.llm.model_adapter import get_profile
from informity.llm.types import StreamSignalTag
from informity.translate_languages import get_translate_language_model_name
from informity.translate_policy import (
    TONE_INSTRUCTIONS,
    TONE_TEMPERATURES,
    TRANSLATE_SECTION_TIMEOUT_S,
    TRANSLATE_TEMPERATURE,
)

# pylint: disable=line-too-long


def _normalize_tone(tone: str | None) -> str:
    """Internal helper for normalize tone."""
    resolved = str(tone or "").strip().lower()
    return resolved if resolved in TONE_INSTRUCTIONS else "natural"


def build_chat_translation_prompt(
    *,
    source_text: str,
    target_language: str,
    tone: str,
) -> tuple[list[dict[str, str]], float]:
    """Build chat translation prompt."""
    model_lang = get_translate_language_model_name(target_language)
    tone_key = _normalize_tone(tone)
    tone_instr = TONE_INSTRUCTIONS.get(tone_key, TONE_INSTRUCTIONS["natural"]).format(
        language=model_lang
    )
    system = (
        f"You are a professional translator. Translate the following text into "
        f"{model_lang}.\n"
        f"{tone_instr}\n"
        "Formatting rules:\n"
        "- Preserve all Markdown: headers (#, ##, ###), bold (**text**), italic (*text*), "
        "bullet lists (- item), numbered lists (1. item).\n"
        "- Preserve code blocks (```...```) and inline code (`...`) exactly — "
        "do not translate their contents.\n"
        "- Reproduce GFM tables with a header row, a separator row (|---|---|), and data rows. "
        "If a source table is ambiguous or malformed, render its content as a bulleted list. "
        "Never output a standalone separator line (---|---) without surrounding table rows.\n"
        "- Do not translate URLs, email addresses, file paths, variable names, proper nouns that are trademarks or brand names, "
        "and any text inside inline code or code blocks.\n"
        f"- If the source text is already written in {model_lang}, reproduce it unchanged.\n"
        "Output ONLY the translated text. No commentary, preamble, or explanations."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": source_text},
    ]
    profile = get_profile()
    if profile.no_think_token:
        messages[-1] = dict(messages[-1])
        messages[-1]["content"] += f"\n{profile.no_think_token}"
    temperature = TONE_TEMPERATURES.get(tone_key, TRANSLATE_TEMPERATURE)
    return messages, temperature


async def translate_chat_message_text(
    source_text: str,
    *,
    target_language: str,
    tone: str,
    cancel_event: asyncio.Event | None = None,
) -> tuple[str | None, str | None]:
    """Translate chat message text."""
    messages, temperature = build_chat_translation_prompt(
        source_text=source_text,
        target_language=target_language,
        tone=tone,
    )
    parts: list[str] = []
    finish_reason: str | None = None
    gen = llm_engine.generate_stream(
        messages,
        max_tokens=4096,
        temperature=temperature,
        timeout_seconds=float(TRANSLATE_SECTION_TIMEOUT_S),
    )
    try:
        async for item in gen:
            if cancel_event and cancel_event.is_set():
                await gen.aclose()
                return None, "cancelled"
            if isinstance(item, tuple):
                token, meta = item
                if token == StreamSignalTag.STREAM_SUMMARY:
                    continue
                if token == "__timeout__":
                    finish_reason = "timeout"
                else:
                    finish_reason = (
                        (meta or {}).get("finish_reason", "stop")
                        if isinstance(meta, dict)
                        else "stop"
                    )
                    if isinstance(token, str) and token:
                        parts.append(token)
                break
            if isinstance(item, str):
                parts.append(item)
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive wrapper mirrors translate route behavior
        raise RuntimeError(f"Chat translation stream error: {exc}") from exc

    translated_text = "".join(parts).strip() or None
    return translated_text, finish_reason
