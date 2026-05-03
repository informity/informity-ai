# ==============================================================================
# Informity AI — PromptCue Signal Adapter
# Generic prompt-shape signal extraction for app policy consumers.
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptcue import PromptCueQueryObject
try:
    from promptcue.patterns import (
        CONTINUATION_REQUEST_PATTERNS as _CONTINUATION_REQUEST_PATTERNS,
    )
    from promptcue.patterns import (
        DISCOURSE_PREFIX_PATTERN as _DISCOURSE_PREFIX_PATTERN,
    )
    from promptcue.patterns import (
        OUTPUT_FORMAT_PATTERNS as _OUTPUT_FORMAT_PATTERNS,
    )
    from promptcue.patterns import (
        REFERENTIAL_FOLLOWUP_PATTERN as _REFERENTIAL_FOLLOWUP_PATTERN,
    )
    from promptcue.patterns import (
        TOPIC_SHIFT_CUE_PATTERN as _TOPIC_SHIFT_CUE_PATTERN,
    )
except Exception:  # noqa: BLE001 - keep deterministic local fallback if promptcue internals move
    _TOPIC_SHIFT_CUE_PATTERN = re.compile(
        r"\bnew\s+topic\b"
        r"|"
        r"\bchange\s+(?:the\s+)?(?:topic|subject)\b"
        r"|"
        r"\bswitch(?:ing)?\s+(?:topic|topics|context)\b"
        r"|"
        r"\bdifferent\s+topic\b"
        r"|"
        r"\binstead\b"
        r"|"
        r"\bunrelated\b"
        r"|"
        r"\bnow\s+(?:about|switch(?:ing)?)\b",
        re.IGNORECASE,
    )
    _REFERENTIAL_FOLLOWUP_PATTERN = re.compile(
        r"\b("
        r"there|that|those|these|it|they|them|same|above|earlier|previous|prior|"
        r"as\s+discussed|as\s+mentioned|continue|follow[-\s]?up|again"
        r")\b",
        re.IGNORECASE,
    )
    _DISCOURSE_PREFIX_PATTERN = re.compile(
        r"^\s*(?:"
        r"ok(?:ay)?|alright|well|so|anyway|now|"
        r"(?:new|different)\s+(?:topic|subject)s?|"
        r"on\s+another\s+subject|switch(?:ing)?\s+(?:topic|topics|subject|subjects|context)|"
        r"(?:back|return(?:ing)?)\s+to"
        r")\b",
        re.IGNORECASE,
    )
    _CONTINUATION_REQUEST_PATTERNS = [
        re.compile(
            r"\b(continue|go\s+on|keep\s+going|the\s+rest|tell\s+me\s+more|next\s+part|next\s+section)\b",
            re.IGNORECASE,
        ),
    ]
    _OUTPUT_FORMAT_PATTERNS = {
        "bullets": re.compile(
            r"\b(as\s+bullet\s+points?|in\s+bullet\s+points?|bullet\s+list|as\s+bullets?)\b",
            re.IGNORECASE,
        ),
        "csv": re.compile(r"\b(csv\s+format|as\s+csv|output\s+csv)\b", re.IGNORECASE),
        "list": re.compile(r"\b(as\s+a\s+list|list\s+format)\b", re.IGNORECASE),
        "narrative": re.compile(r"\b(in\s+narrative\s+form|as\s+paragraphs?)\b", re.IGNORECASE),
        "table": re.compile(
            r"\b(markdown\s+table|as\s+a\s+table|in\s+table\s+form|in\s+columns?)\b", re.IGNORECASE
        ),
    }


@dataclass(frozen=True)
class PromptSignalSnapshot:
    has_discourse_prefix: bool = False
    has_topic_shift_cue: bool = False
    has_referential_followup: bool = False
    requests_continuation: bool = False
    is_continuation: bool = False
    requested_output_formats: tuple[str, ...] = field(default_factory=tuple)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _from_promptcue_query_object(pcue: PromptCueQueryObject | None) -> PromptSignalSnapshot | None:
    if pcue is None:
        return None
    prompt_signals = getattr(pcue, "prompt_signals", None)
    if prompt_signals is None:
        return PromptSignalSnapshot(is_continuation=bool(getattr(pcue, "is_continuation", False)))
    output_formats = tuple(
        sorted({
            str(item)
            for item in (getattr(prompt_signals, "requested_output_formats", None) or [])
            if str(item).strip()
        })
    )
    return PromptSignalSnapshot(
        has_discourse_prefix=bool(getattr(prompt_signals, "has_discourse_prefix", False)),
        has_topic_shift_cue=bool(getattr(prompt_signals, "has_topic_shift_cue", False)),
        has_referential_followup=bool(getattr(prompt_signals, "has_referential_followup", False)),
        requests_continuation=bool(getattr(prompt_signals, "requests_continuation", False)),
        is_continuation=bool(getattr(pcue, "is_continuation", False)),
        requested_output_formats=output_formats,
    )


def _fallback_signal_snapshot(text: str) -> PromptSignalSnapshot:
    requested_output_formats = tuple(
        sorted(
            key
            for key, pattern in _OUTPUT_FORMAT_PATTERNS.items()
            if pattern.search(text)
        )
    )
    return PromptSignalSnapshot(
        has_discourse_prefix=bool(_DISCOURSE_PREFIX_PATTERN.match(text)),
        has_topic_shift_cue=bool(_TOPIC_SHIFT_CUE_PATTERN.search(text)),
        has_referential_followup=bool(_REFERENTIAL_FOLLOWUP_PATTERN.search(text)),
        requests_continuation=any(pattern.search(text) for pattern in _CONTINUATION_REQUEST_PATTERNS),
        is_continuation=False,
        requested_output_formats=requested_output_formats,
    )


def extract_prompt_signals(
    text: str,
    *,
    pcue: PromptCueQueryObject | None = None,
) -> PromptSignalSnapshot:
    precomputed = _from_promptcue_query_object(pcue)
    if precomputed is not None:
        return precomputed

    normalized = _normalize_text(text)
    if not normalized:
        return PromptSignalSnapshot()

    return _fallback_signal_snapshot(normalized)


__all__ = ["PromptSignalSnapshot", "extract_prompt_signals"]
