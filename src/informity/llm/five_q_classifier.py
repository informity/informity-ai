from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from informity.config import settings
from informity.exceptions import LLMError
from informity.llm.engine import LLMEngine
from informity.llm.five_q_decision import FiveQDecision

log = structlog.get_logger(__name__)

_ALLOWED_SOURCES = {'index_metadata', 'document_content', 'chat_history', 'app_knowledge'}
_ALLOWED_SCOPES = {'targeted', 'broad', 'none'}
_ALLOWED_OPERATIONS = {'lookup', 'count_enumerate', 'summarize_synthesize', 'compare'}
_ROUTER_MAX_TOKENS = 220
_ROUTER_TEMPERATURE = 0.0

_SYSTEM_PROMPT = """You are a query classifier for a document chat app.
Answer exactly five questions using JSON only.

Return this object:
{
  "source": "index_metadata | document_content | chat_history | app_knowledge",
  "scope": "targeted | broad | none",
  "operation": "lookup | count_enumerate | summarize_synthesize | compare",
  "partitions": ["..."],
  "exhaustive": true|false,
  "confidence": 0.0-1.0
}

Rules:
- source=index_metadata for counts, lists, inventory, file/document metadata, or asking what is in the library.
- source=chat_history for recaps/summaries of the conversation.
- source=app_knowledge for questions about the app itself.
- source=document_content for questions that need reading documents.
- scope=none unless source=document_content.
- scope=targeted only when the user asks about one document or one explicitly named item.
- scope=broad for everything/all/across/overall/entire-set questions or multi-document synthesis.
- partitions are explicit grouping values such as years.
- operation lookup for a single fact, field, or attribute.
- operation summarize_synthesize for "what does ... say", "tell me everything", "summarize",
  "explain", "overview", or combined understanding across evidence.
- exhaustive=true only when the user explicitly asks for totals, every matching item, full
  inventory, or complete coverage across all matches.
- Do not set exhaustive=true just because the query is broad or summary-like.
- Broad synthesis questions like "tell me everything we know about X" should usually be
  exhaustive=false unless the user explicitly asks for every matching item or total coverage.
- personal finance / property / loan / mortgage / closing / escrow / statement questions are document_content when the answer should come from the user's documents.

Examples:
- "What kind of documents do you have indexed?" -> source=index_metadata, scope=none, operation=count_enumerate, partitions=[], exhaustive=false
- "How many PDFs do I have from 2024?" -> source=index_metadata, scope=none, operation=count_enumerate, partitions=["2024"], exhaustive=false
- "Summarize our last conversation." -> source=chat_history, scope=none, operation=summarize_synthesize, partitions=[], exhaustive=false
- "What does this app do?" -> source=app_knowledge, scope=none, operation=lookup, partitions=[], exhaustive=false
- "What is the interest rate on my mortgage?" -> source=document_content, scope=targeted, operation=lookup, partitions=[], exhaustive=false
- "What does the 2025 closing package say about escrow?" -> source=document_content, scope=targeted, operation=summarize_synthesize, partitions=["2025"], exhaustive=false
- "Tell me everything we know about my Escondido property." -> source=document_content, scope=broad, operation=summarize_synthesize, partitions=[], exhaustive=false
- "Compare the 2023 and 2025 closing documents." -> source=document_content, scope=broad, operation=compare, partitions=["2023","2025"], exhaustive=false
"""

_APP_HELP_PATTERN = re.compile(r'\b(what does this app do|how do i use|help me use|help with the app)\b', re.IGNORECASE)
_CHAT_HISTORY_PATTERN = re.compile(r'\b(what did we discuss|what have we been chatting about|recap (?:our|this) chat|summarize (?:our|this) chat)\b', re.IGNORECASE)
_INDEX_METADATA_PATTERN = re.compile(r'\b(how many|list all|what kinds|what type|inventory|enumerate|count)\b', re.IGNORECASE)
_DOCUMENT_CONTENT_PATTERN = re.compile(r'\b(mortgage|loan|property|document|file|agreement|statement|closing)\b', re.IGNORECASE)
_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')
_CLASSIFIER_MODEL_PRIORITY = (
    ('4b', re.compile(r'\b4b\b', re.IGNORECASE)),
    ('2b', re.compile(r'\b2b\b', re.IGNORECASE)),
)


@dataclass(frozen=True)
class ClassifierContext:
    chat_mode: str | None = None
    scope_kind: str | None = None
    has_prior_turns: bool = False
    prior_user_query: str | None = None


@dataclass(frozen=True)
class FiveQClassificationResult:
    decision: FiveQDecision
    raw_output: str
    model_name: str


def _discover_classifier_model_path() -> Path | None:
    preferred_dirs = [Path(settings.classifier_models_dir), Path(settings.models_dir)]
    for models_dir in preferred_dirs:
        if not models_dir.exists():
            continue
        candidates = sorted(path for path in models_dir.glob('*.gguf') if path.is_file())
        if not candidates:
            continue
        for _, pattern in _CLASSIFIER_MODEL_PRIORITY:
            for path in candidates:
                if pattern.search(path.name):
                    return path
        return candidates[0]
    return None


def _normalize_decision(data: dict[str, Any]) -> FiveQDecision:
    source = str(data.get('source') or '').strip()
    if source not in _ALLOWED_SOURCES:
        source = 'document_content'
    scope = str(data.get('scope') or 'none').strip()
    if scope not in _ALLOWED_SCOPES:
        scope = 'none'
    operation = str(data.get('operation') or 'lookup').strip()
    if operation not in _ALLOWED_OPERATIONS:
        operation = 'lookup'
    partitions_value = data.get('partitions') or []
    partitions = [str(item).strip() for item in partitions_value if str(item).strip()]
    exhaustive = bool(data.get('exhaustive', False))
    confidence = float(data.get('confidence') or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return FiveQDecision(
        source=source,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        operation=operation,  # type: ignore[arg-type]
        partitions=partitions,
        exhaustive=exhaustive,
        confidence=confidence,
    )


def _fallback_decision(query: str, context: ClassifierContext) -> FiveQDecision:
    text = str(query or '').strip()
    lowered = text.casefold()
    if _CHAT_HISTORY_PATTERN.search(text):
        return FiveQDecision(source='chat_history', scope='none', operation='summarize_synthesize', confidence=0.95)
    if _APP_HELP_PATTERN.search(text):
        return FiveQDecision(source='app_knowledge', scope='none', operation='lookup', confidence=0.9)
    if _INDEX_METADATA_PATTERN.search(text) and not _DOCUMENT_CONTENT_PATTERN.search(text):
        return FiveQDecision(source='index_metadata', scope='none', operation='count_enumerate', confidence=0.92)
    if context.chat_mode == 'assistant' and not context.has_prior_turns:
        return FiveQDecision(source='app_knowledge', scope='none', operation='lookup', confidence=0.72)
    if _YEAR_PATTERN.search(lowered) and 'compare' in lowered:
        return FiveQDecision(source='document_content', scope='broad', operation='compare', partitions=_YEAR_PATTERN.findall(lowered), confidence=0.8)
    if _YEAR_PATTERN.search(lowered) and ('all' in lowered or 'every' in lowered or 'across' in lowered):
        return FiveQDecision(source='document_content', scope='broad', operation='summarize_synthesize', partitions=_YEAR_PATTERN.findall(lowered), exhaustive=True, confidence=0.82)
    return FiveQDecision(source='document_content', scope='targeted', operation='lookup', confidence=0.75)


class FiveQClassifier:
    def __init__(self) -> None:
        self._model_path = _discover_classifier_model_path()
        self._model_filename = self._model_path.name if self._model_path is not None else None

    def classify(self, query: str, context: ClassifierContext) -> FiveQClassificationResult:
        text = str(query or '').strip()
        if not text:
            decision = FiveQDecision(source='app_knowledge', scope='none', operation='lookup', confidence=0.0)
            return FiveQClassificationResult(decision=decision, raw_output='', model_name='empty')

        if os.environ.get('PYTEST_CURRENT_TEST') or self._model_path is None:
            decision = _fallback_decision(text, context)
            return FiveQClassificationResult(decision=decision, raw_output='{"mode":"fallback"}', model_name='fallback')

        if context.prior_user_query:
            prior_user_query = str(context.prior_user_query or '').strip()
        else:
            prior_user_query = ''

        user_lines = [
            f'chat_mode: {context.chat_mode or "unknown"}',
            f'scope_kind: {context.scope_kind or "unknown"}',
            f'has_prior_turns: {str(bool(context.has_prior_turns)).lower()}',
            f'prior_user_query: {prior_user_query or "none"}',
            '',
            f'query: {text}',
        ]
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': '\n'.join(user_lines)},
        ]
        engine = LLMEngine(
            provider_name='local_gguf',
            model_filename=self._model_filename,
            model_dir=self._model_path.parent,
        )
        try:
            response = engine.chat_complete(
                messages=messages,
                max_tokens=_ROUTER_MAX_TOKENS,
                temperature=_ROUTER_TEMPERATURE,
                response_format={'type': 'json_object'},
            )
            raw_output = str((response.get('choices') or [{}])[0].get('message', {}).get('content') or '').strip()
            parsed = json.loads(raw_output or '{}')
            if not isinstance(parsed, dict):
                raise ValueError('five_q_classifier_expected_object')
            decision = _normalize_decision(parsed)
            return FiveQClassificationResult(decision=decision, raw_output=raw_output, model_name=self._model_filename)
        except (LLMError, json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning(
                'five_q_classifier_fallback',
                error=str(exc),
                model_filename=self._model_filename,
            )
            decision = _fallback_decision(text, context)
            return FiveQClassificationResult(decision=decision, raw_output='', model_name=self._model_filename)
        finally:
            engine.unload()


__all__ = ['ClassifierContext', 'FiveQClassifier', 'FiveQClassificationResult']
