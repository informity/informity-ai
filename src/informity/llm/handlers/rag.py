# ==============================================================================
# Informity AI — RAG Query Handler
# Handles focused and coverage queries using vector search → rerank → LLM
# ==============================================================================

"""Module for llm handlers rag."""

# pylint: disable=unused-argument

import asyncio
import re
import time
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.error_messages import to_client_error_message
from informity.api.schemas import ChatSourceReference
from informity.config import settings
from informity.db.models import ChatMessage
from informity.db.sqlite import get_chunk_count
from informity.llm.metrics_payload import build_metrics_payload
from informity.llm.model_adapter import get_profile, get_retrieval_top_k
from informity.llm.nlp_heuristics import BY_PER_YEAR_PATTERN
from informity.llm.prompt_builder import BuildMessagesRequest, build_messages, resolve_history_limit
from informity.llm.query_classifier import QueryClassification
from informity.llm.query_patterns import (
    build_acronym_entity_listing_pattern,
    build_exhaustive_entity_inventory_scope_pattern,
    build_global_entity_listing_pattern,
    build_person_entity_listing_pattern,
)
from informity.llm.query_rewrite import build_followup_retrieval_query
from informity.llm.rag_patterns import (
    SUMMARY_BLOCK_TYPE_EXCLUDE,
    evaluate_substantive_evidence,
    has_comparison_cue,
    has_explicit_title_reference,
    has_extraction_cue,
    has_referential_followup_language,
    has_topic_overlap_with_previous_user,
    has_topic_shift_cue,
    is_summary_style_request,
    normalize_query_text,
    should_prefer_title_alignment,
)
from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.rag_runtime import generation_stream as _generation_stream
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.llm.retrieval import retrieve_chunks
from informity.llm.streaming import stream_llm
from informity.llm.types import (
    OutputFormat,
    QuerySubtype,
    QueryType,
    StreamSignalTag,
)
from informity.llm.user_messages import (
    EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE,
    INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE,
)

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)

_INSUFFICIENT_CONTEXT_RESPONSE = INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE
_CHUNK_PREVIEW_MAX_LENGTH = 200
_DETERMINISTIC_EXTRACTION_HEADING = "### Deterministic Numeric Extraction\n\n"
_STRICT_CONTRACT_MAX_TEMPERATURE = 0.2
_STRICT_CONTRACT_MAX_TOP_P = 0.8
_COVERAGE_ENTITY_LISTING_TOP_K_BOOST = 8
_COVERAGE_ENTITY_LISTING_TOP_K_MAX = 60
_FILE_DISCOVERY_RETRIEVAL_TOP_K = 200
_FILE_DISCOVERY_DISPLAY_LIMIT = 20
_FILE_DISCOVERY_LEAD_IN = "Here are the files related to your query:"
_GLOBAL_ENTITY_ENUMERATION_PATTERN = build_global_entity_listing_pattern()
_ENTITY_INVENTORY_SCOPE_PATTERN = build_exhaustive_entity_inventory_scope_pattern()
_PERSON_INVENTORY_PATTERN = build_person_entity_listing_pattern()
_ACRONYM_INVENTORY_PATTERN = build_acronym_entity_listing_pattern()
_ENTITY_INVENTORY_MAX_ROWS = 300
_ENTITY_INVENTORY_SOURCE_LIMIT = 12
_ENTITY_INVENTORY_DEFAULT_PREVIEW = "Indexed term evidence match."
_WEAK_SUMMARY_SUBSTANTIVE_RATIO_THRESHOLD = 0.55
_WEAK_SUMMARY_DOMINANT_FILE_RATIO_THRESHOLD = 0.6
_SUMMARY_TITLE_MAX_TEMPERATURE = 0.3
_SUMMARY_TITLE_MAX_TOP_P = 0.9
_WEAK_COMPARISON_DISTINCT_FILE_THRESHOLD = 2


def _dominant_file_ratio(chunks: list[dict]) -> float:
    """Internal helper for dominant file ratio."""
    if not chunks:
        return 0.0
    counts: dict[int, int] = {}
    for chunk in chunks:
        file_id = chunk.get("file_id")
        if not isinstance(file_id, int):
            continue
        counts[file_id] = counts.get(file_id, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / max(1, len(chunks))


def _distinct_file_count(chunks: list[dict]) -> int:
    """Internal helper for distinct file count."""
    file_ids: set[int] = set()
    for chunk in chunks:
        file_id = chunk.get("file_id")
        if isinstance(file_id, int):
            file_ids.add(file_id)
    return len(file_ids)


def _chunk_dedupe_key(chunk: dict) -> tuple[object, ...]:
    """Internal helper for chunk dedupe key."""
    chunk_id = chunk.get("chunk_id")
    if isinstance(chunk_id, int):
        return ("chunk_id", chunk_id)

    file_id = chunk.get("file_id")
    filename = str(chunk.get("filename") or "").casefold()
    chunk_text = str(chunk.get("chunk_text") or "").casefold()[:120]
    if isinstance(file_id, int):
        return ("file_id", file_id, filename, chunk_text)
    return ("fallback", filename, chunk_text)


def _collapse_duplicate_insufficient_context_message(
    answer: str,
    *,
    phrase: str = _INSUFFICIENT_CONTEXT_RESPONSE,
) -> tuple[str, bool]:
    """Internal helper for collapse duplicate insufficient context message."""
    if not answer:
        return answer, False
    escaped_phrase = re.escape(phrase)
    pattern = re.compile(rf"(?:{escaped_phrase}\s*){{2,}}")
    collapsed = pattern.sub(f"{phrase}\n", answer)
    if collapsed == answer:
        return answer, False
    return collapsed, True


def _resolve_sampling_params(
    *,
    profile_temperature: float,
    profile_top_p: float,
    format_requirements: list[str] | None = None,
) -> tuple[float, float]:
    # Keep runtime simple and contract-compliant:
    # when strict format contracts are present, lower sampling variance so
    # heading/section structure is more reproducible across reruns.
    """Internal helper for resolve sampling params."""
    requirements_joined = " ".join(
        str(item or "").casefold() for item in (format_requirements or [])
    )
    strict_contract = (
        "requested order" in requirements_joined
        or "include heading:" in requirements_joined
        or "one subsection per year" in requirements_joined
    )
    if strict_contract:
        return (
            min(profile_temperature, _STRICT_CONTRACT_MAX_TEMPERATURE),
            min(profile_top_p, _STRICT_CONTRACT_MAX_TOP_P),
        )
    return profile_temperature, profile_top_p


def _apply_output_format_preferences(
    *,
    output_format: OutputFormat | None,
    format_requirements: list[str],
    output_constraints: dict[str, int],
) -> None:
    """Internal helper for apply output format preferences."""
    if output_format == OutputFormat.TABLE:
        format_requirements.append("Output as a markdown table.")
        return
    if output_format == OutputFormat.BULLETS:
        format_requirements.append("Output using bullet points.")
        return
    if output_format == OutputFormat.CSV:
        format_requirements.append("Output as CSV with a single header row.")
        return
    if output_format == OutputFormat.LIST:
        format_requirements.append("Output as a concise list.")
        return
    if output_format == OutputFormat.NARRATIVE:
        format_requirements.append("Output as narrative paragraphs, not a table.")
        output_constraints.pop("exact_top_level_bullets", None)


def _apply_negation_preferences(
    *,
    is_negation_query: bool,
    format_requirements: list[str],
) -> None:
    """Internal helper for apply negation preferences."""
    if not is_negation_query:
        return
    format_requirements.append(
        "If exact negation cannot be guaranteed from retrieved evidence and metadata filters, "
        "state that limitation explicitly and avoid definitive exclusion claims."
    )


def _truncate_preview(text: str, max_length: int = _CHUNK_PREVIEW_MAX_LENGTH) -> str:
    """Internal helper for truncate preview."""
    return text[:max_length]


def _should_prepend_deterministic_extraction_heading(
    *,
    question: str,
    classification: QueryClassification,
) -> bool:
    """Internal helper for should prepend deterministic extraction heading."""
    if classification.intent != QueryType.COVERAGE:
        return False
    if classification.subtype != QuerySubtype.AGGREGATE_BY_PERIOD:
        return False
    if not has_extraction_cue(question):
        return False
    return BY_PER_YEAR_PATTERN.search(question)


def _build_history_aware_retrieval_query_with_classification(
    *,
    question: str,
    history: list[ChatMessage] | None,
    classification: QueryClassification | None,
) -> tuple[str, bool]:
    """Internal helper for build history aware retrieval query with classification."""
    normalized_question = normalize_query_text(question)
    if not normalized_question:
        return "", False
    if classification is not None:
        if bool(classification.focus_resolved):
            rewritten_query = normalize_query_text(classification.focus_rewritten_query or "")
            if rewritten_query and bool(classification.focus_query_rewritten):
                return rewritten_query, True
            return normalized_question, False
        if bool(classification.is_scope_reset):
            return normalized_question, False
        if classification.filename_filter is not None:
            return normalized_question, False
        if classification.intent not in {QueryType.FOCUSED, QueryType.COVERAGE}:
            return normalized_question, False
    if not bool(settings.rag_query_rewrite_enabled):
        return normalized_question, False
    if not history:
        return normalized_question, False
    if has_topic_shift_cue(normalized_question):
        return normalized_question, False
    has_referential_language = has_referential_followup_language(normalized_question)
    has_topical_overlap = has_topic_overlap_with_previous_user(
        question=normalized_question,
        history=history,
    )
    if not has_referential_language and not has_topical_overlap:
        return normalized_question, False

    previous_user = ""
    history_limit = max(0, int(settings.rag_query_rewrite_max_history_messages))
    if history_limit == 0:
        return normalized_question, False
    max_chars_per_turn = max(1, int(settings.rag_query_rewrite_max_chars_per_turn))
    max_query_chars = max(64, int(settings.rag_query_rewrite_max_query_chars))

    for message in reversed(history[-history_limit:]):
        content = normalize_query_text(message.content or "")
        if not content:
            continue
        if not previous_user and message.role == "user":
            previous_user = content
        if previous_user:
            break

    if not previous_user:
        return normalized_question, False

    rewritten_query = build_followup_retrieval_query(
        normalized_question=normalized_question,
        previous_user_question=previous_user,
        max_chars_per_turn=max_chars_per_turn,
        max_query_chars=max_query_chars,
    )
    return rewritten_query, True


def _resolve_minimal_query_type(classification: QueryClassification) -> QueryType:
    """Internal helper for resolve minimal query type."""
    if classification.intent == QueryType.COVERAGE:
        return QueryType.COVERAGE
    return QueryType.FOCUSED


def _should_boost_coverage_top_k(question: str, classification: QueryClassification) -> bool:
    """Internal helper for should boost coverage top k."""
    if classification.intent != QueryType.COVERAGE:
        return False
    lowered = str(question or "").casefold()
    if not lowered:
        return False
    return bool(
        _GLOBAL_ENTITY_ENUMERATION_PATTERN.search(lowered)
        and _ENTITY_INVENTORY_SCOPE_PATTERN.search(lowered)
    )


def _resolve_agent_retrieval_queries(
    *,
    retrieval_query: str,
    classification: QueryClassification,
) -> list[str]:
    """Internal helper for resolve agent retrieval queries."""
    normalized_queries = [
        normalize_query_text(query).strip().strip(" .?!,;:")
        for query in (classification.agent_subqueries or [])
        if normalize_query_text(query).strip().strip(" .?!,;:")
    ]
    if normalized_queries:
        seen_queries: set[str] = set()
        resolved_queries: list[str] = []
        for query in normalized_queries:
            key = query.casefold()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            resolved_queries.append(query)
            if len(resolved_queries) >= int(settings.agent_multi_query_max_subqueries):
                break
        return resolved_queries
    fallback_query = normalize_query_text(retrieval_query).strip().strip(" .?!,;:")
    return [fallback_query] if fallback_query else []


def _resolve_minimal_answerability_settings(query_type: QueryType) -> tuple[float, int]:
    """Internal helper for resolve minimal answerability settings."""
    if query_type == QueryType.COVERAGE:
        return (
            float(settings.rag_minimal_answerability_threshold_coverage),
            int(settings.rag_minimal_min_chunks_coverage),
        )
    return (
        float(settings.rag_minimal_answerability_threshold_focused),
        int(settings.rag_minimal_min_chunks_focused),
    )


def _resolve_exhaustive_inventory_term_type(
    question: str,
    classification: QueryClassification,
) -> str | None:
    """Internal helper for resolve exhaustive inventory term type."""
    if classification.intent != QueryType.COVERAGE:
        return None
    lowered = str(question or "").casefold()
    if not lowered:
        return None
    if not _ENTITY_INVENTORY_SCOPE_PATTERN.search(lowered):
        return None
    if _PERSON_INVENTORY_PATTERN.search(lowered):
        return "person_name"
    if _ACRONYM_INVENTORY_PATTERN.search(lowered):
        return "acronym"
    return None


async def _fetch_term_inventory_rows(
    *,
    db: aiosqlite.Connection,
    term_type: str,
    limit: int = _ENTITY_INVENTORY_MAX_ROWS,
) -> list[dict[str, object]]:
    """Internal helper for fetch term inventory rows."""
    cursor = await db.execute(
        """
        SELECT
            te.canonical_term AS canonical_term,
            te.confidence AS confidence,
            COUNT(DISTINCT tei.file_id) AS file_count
        FROM term_entries te
        JOIN term_dictionary_state tds ON tds.singleton_id = 1
        LEFT JOIN term_evidence tei ON tei.term_id = te.term_id
        WHERE te.dict_version = tds.current_version
          AND te.status = 'active'
          AND te.type = ?
        GROUP BY te.term_id
        ORDER BY te.canonical_term COLLATE NOCASE ASC
        LIMIT ?
        """,
        (term_type, max(1, int(limit))),
    )
    rows = await cursor.fetchall()
    return [
        {
            "canonical_term": str(row["canonical_term"] or "").strip(),
            "confidence": float(row["confidence"] or 0.0),
            "file_count": int(row["file_count"] or 0),
        }
        for row in rows
        if str(row["canonical_term"] or "").strip()
    ]


async def _fetch_term_inventory_sources(
    *,
    db: aiosqlite.Connection,
    term_type: str,
    limit: int = _ENTITY_INVENTORY_SOURCE_LIMIT,
) -> list[ChatSourceReference]:
    """Internal helper for fetch term inventory sources."""
    cursor = await db.execute(
        """
        SELECT
            f.id AS file_id,
            f.filename AS filename,
            f.path AS path,
            MAX(COALESCE(tei.evidence_snippet, '')) AS chunk_preview,
            MAX(te.confidence) AS relevance_score
        FROM term_entries te
        JOIN term_dictionary_state tds ON tds.singleton_id = 1
        JOIN term_evidence tei ON tei.term_id = te.term_id
        JOIN files f ON f.id = tei.file_id
        WHERE te.dict_version = tds.current_version
          AND te.status = 'active'
          AND te.type = ?
        GROUP BY f.id, f.filename, f.path
        ORDER BY f.filename COLLATE NOCASE ASC
        LIMIT ?
        """,
        (term_type, max(1, int(limit))),
    )
    rows = await cursor.fetchall()
    sources: list[ChatSourceReference] = []
    for row in rows:
        path = str(row["path"] or "").strip()
        filename = str(row["filename"] or "").strip() or (
            path.rsplit("/", maxsplit=1)[-1] if path else "source"
        )
        if not path:
            continue
        chunk_preview = str(row["chunk_preview"] or "").strip() or _ENTITY_INVENTORY_DEFAULT_PREVIEW
        relevance_score = float(row["relevance_score"] or 0.0)
        sources.append(
            ChatSourceReference(
                filename=filename,
                path=path,
                chunk_preview=chunk_preview,
                relevance_score=relevance_score,
                file_id=int(row["file_id"]) if row["file_id"] is not None else None,
            )
        )
    return sources


def _format_term_inventory_answer(*, term_type: str, rows: list[dict[str, object]]) -> str:
    """Internal helper for format term inventory answer."""
    if term_type == "person_name":
        title = "people names"
    elif term_type == "acronym":
        title = "acronyms"
    else:
        title = "entities"
    if not rows:
        return (
            f"No {title} were found in the current indexed term dictionary. "
            "Reindex your documents and try again."
        )

    lines = [
        f"From indexed term dictionary entries, here are the {title} found across the corpus:",
        "",
    ]
    for row in rows:
        name = str(row.get("canonical_term") or "").strip()
        if not name:
            continue
        file_count = int(row.get("file_count") or 0)
        if file_count > 0:
            lines.append(f"- **{name}** ({file_count} file{'s' if file_count != 1 else ''})")
        else:
            lines.append(f"- **{name}**")
    return "\n".join(lines)


def _should_use_deterministic_file_discovery_response(
    classification: QueryClassification,
) -> bool:
    """Internal helper for should use deterministic file discovery response."""
    decision = getattr(classification, "shadow_classifier_decision", None)
    if not isinstance(decision, dict):
        return False
    return (
        str(decision.get("source") or "") == "document_content"
        and str(decision.get("operation") or "") == "count_enumerate"
    )


def _format_file_discovery_answer(chunks: list[dict]) -> str:
    """Internal helper for format file discovery answer."""
    return _build_file_discovery_response(question="", chunks=chunks)[0]


_FILE_DISCOVERY_PREFIX_RE = re.compile(
    r"^(?:"
    r"show me|find|list|display|get|give me|tell me|which|what"
    r")\s+",
    re.IGNORECASE,
)
_FILE_DISCOVERY_LEADING_FILLER_RE = re.compile(
    r"^(?:all|every|any|the|a|an|my|our|your|this|that|these|those)\s+",
    re.IGNORECASE,
)
_FILE_DISCOVERY_TRAILING_FILLER_RE = re.compile(
    r"\s+(?:all|every|any|the|a|an|my|our|your|this|that|these|those)$",
    re.IGNORECASE,
)
_FILE_DISCOVERY_CORPUS_NOUNS_RE = re.compile(
    r"^(?:files?|documents?|records?|items?)\s+",
    re.IGNORECASE,
)
_FILE_DISCOVERY_TRAILING_CORPUS_NOUNS_RE = re.compile(
    r"\s+(?:files?|documents?|records?|items?)$",
    re.IGNORECASE,
)
_FILE_DISCOVERY_SEMANTIC_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:related(?:\s+to)?|about|mention(?:ing)?|mentions?|involving|"
        r"involves?|containing|contains?|regarding|pertaining|connected(?:\s+to)?|"
        r"relevant(?:\s+to)?)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:related(?:\s+to)?|about|mention(?:ing)?|mentions?|involving|"
        r"involves?|containing|contains?|regarding|pertaining|connected(?:\s+to)?|"
        r"relevant(?:\s+to)?)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:related(?:\s+to)?|about|mention(?:ing)?|mentions?|involving|"
        r"involves?|containing|contains?|regarding|pertaining|connected(?:\s+to)?|"
        r"relevant(?:\s+to)?)$",
        re.IGNORECASE,
    ),
)


def _extract_file_discovery_search_term(question: str) -> str:
    """Internal helper for extract file discovery search term."""
    candidate = normalize_query_text(question).strip().strip(" .?!,;:")
    if not candidate:
        return ""

    previous = None
    while candidate and candidate != previous:
        previous = candidate
        candidate = _FILE_DISCOVERY_PREFIX_RE.sub("", candidate)
        candidate = _FILE_DISCOVERY_LEADING_FILLER_RE.sub("", candidate)
        candidate = _FILE_DISCOVERY_CORPUS_NOUNS_RE.sub("", candidate)
        candidate = _FILE_DISCOVERY_TRAILING_CORPUS_NOUNS_RE.sub("", candidate)
        candidate = _FILE_DISCOVERY_TRAILING_FILLER_RE.sub("", candidate)
        for pattern in _FILE_DISCOVERY_SEMANTIC_PHRASE_PATTERNS:
            candidate = pattern.sub(" ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()

    return candidate or normalize_query_text(question).strip().strip(" .?!,;:")


def _build_file_discovery_response(
    *,
    question: str,
    chunks: list[dict],
) -> tuple[str, dict[str, object]]:
    """Internal helper for build file discovery response."""
    seen_files: set[int | str] = set()
    filenames: list[str] = []
    for chunk in chunks:
        file_id = chunk.get("file_id")
        filename = str(chunk.get("filename") or "").strip()
        if not filename:
            continue
        key: int | str = file_id if isinstance(file_id, int) else filename.casefold()
        if key in seen_files:
            continue
        seen_files.add(key)
        filenames.append(filename)

    shown_filenames = filenames[:_FILE_DISCOVERY_DISPLAY_LIMIT]
    if not shown_filenames:
        return (
            "I could not identify any relevant files in the retrieved context.",
            {
                "is_file_discovery": True,
                "search_term": _extract_file_discovery_search_term(question),
                "shown_count": 0,
                "total_count": 0,
            },
        )
    answer_lines = [_FILE_DISCOVERY_LEAD_IN, ""]
    answer_lines.extend(f"- **{filename}**" for filename in shown_filenames)

    file_discovery = {
        "is_file_discovery": True,
        "search_term": _extract_file_discovery_search_term(question),
        "shown_count": len(shown_filenames),
        "total_count": len(filenames),
    }
    return "\n".join(answer_lines), file_discovery


def _evaluate_minimal_answerability(
    chunks: list[dict],
    *,
    query_type: QueryType,
) -> tuple[bool, float, float, int]:
    """Internal helper for evaluate minimal answerability."""
    threshold, min_chunks = _resolve_minimal_answerability_settings(query_type)
    normalized_scores = [
        _retrieval_validation._normalize_relevance_score(chunk.get("score", 0.0))
        for chunk in chunks[:3]
    ]
    mean_top3_score = sum(normalized_scores) / max(1, len(normalized_scores))
    chunk_count = len(chunks)
    passed = (chunk_count >= min_chunks) and (mean_top3_score >= threshold)
    return passed, mean_top3_score, threshold, min_chunks


class RAGHandler:
    """
    Handler for focused and coverage queries.

    Uses vector search → rerank → LLM pipeline.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match focused and coverage queries."""
        return classification.intent in {QueryType.FOCUSED, QueryType.COVERAGE}

    async def _handle_minimal_mode(
        self,
        question: str,
        classification: QueryClassification,
        history: list[ChatMessage] | None,
        db: aiosqlite.Connection,
        trace: object | None,
        timing_context: dict[str, object] | None = None,
        file_ids: list[int] | None = None,
        specialization_id: str | None = None,
        agent_mode: bool = False,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """Internal helper for handle minimal mode."""
        profile = get_profile()
        effective_query_type = _resolve_minimal_query_type(classification)
        inventory_term_type = _resolve_exhaustive_inventory_term_type(question, classification)
        if inventory_term_type is not None:
            inventory_start = time.perf_counter()
            inventory_rows = await _fetch_term_inventory_rows(
                db=db,
                term_type=inventory_term_type,
            )
            inventory_sources = await _fetch_term_inventory_sources(
                db=db,
                term_type=inventory_term_type,
            )
            inventory_elapsed_ms = (time.perf_counter() - inventory_start) * 1000.0
            if trace is not None:
                trace.record(
                    "deterministic_inventory",
                    {
                        "enabled": True,
                        "term_type": inventory_term_type,
                        "entry_count": len(inventory_rows),
                        "source_count": len(inventory_sources),
                        "duration_ms": round(inventory_elapsed_ms, 1),
                    },
                )
            yield (
                StreamSignalTag.METRICS,
                build_metrics_payload(
                    query_type=effective_query_type,
                    raw_chunks_count=0,
                    retrieval_duration_ms=round(inventory_elapsed_ms, 1),
                    prompt_build_ms="N/A",
                    llm_submit_ms="N/A",
                    llm_queue_wait_ms="N/A",
                    llm_decode_first_token_ms="N/A",
                    generation_skipped=True,
                    minimal_mode=True,
                    deterministic_inventory=True,
                    deterministic_inventory_term_type=inventory_term_type,
                ),
            )
            yield _format_term_inventory_answer(
                term_type=inventory_term_type,
                rows=inventory_rows,
            )
            yield inventory_sources
            return

        effective_top_k = get_retrieval_top_k(effective_query_type)
        if _should_boost_coverage_top_k(question, classification):
            effective_top_k = min(
                _COVERAGE_ENTITY_LISTING_TOP_K_MAX,
                effective_top_k + _COVERAGE_ENTITY_LISTING_TOP_K_BOOST,
            )
        if _should_use_deterministic_file_discovery_response(classification):
            effective_top_k = max(effective_top_k, _FILE_DISCOVERY_RETRIEVAL_TOP_K)
        max_tokens = profile.get_max_tokens(effective_query_type)
        timeout_seconds = profile.get_timeout_seconds(
            effective_query_type,
            agent_mode=agent_mode,
        )
        reasoning_enabled = profile.get_reasoning_enabled(
            effective_query_type,
            reasoning_enabled_override=agent_mode if profile.supports_think_blocks else None,
        )
        chat_template_kwargs_override = (
            {"enable_thinking": True}
            if agent_mode and profile.supports_think_blocks
            else None
        )
        format_requirements: list[str] = []
        output_constraints: dict[str, int] = {}

        if trace is not None:
            trace.record(
                "intent",
                {
                    "model_profile": profile.name,
                    "intent": classification.intent,
                    "query_type": effective_query_type,
                    "coverage_mode": effective_query_type == QueryType.COVERAGE,
                    "minimal_mode": True,
                    "top_k": effective_top_k,
                    "rag_max_score": getattr(profile, "rag_max_score", None),
                    "continuation_behavior": "fresh_retrieval",
                },
            )

        retrieval_timing: dict[str, float] = {}
        retrieval_query, query_rewritten = _build_history_aware_retrieval_query_with_classification(
            question=(classification.retrieval_content_query or question),
            history=history,
            classification=classification,
        )
        if classification.focus_resolved:
            explicit_title_reference = bool(classification.focus_explicit_title_reference)
            referential_title_anchor = classification.focus_referential_title_anchor
            title_alignment_query = classification.focus_title_alignment_query or question
            prefer_title_alignment = bool(classification.focus_prefer_title_alignment)
            strict_title_alignment = bool(classification.focus_strict_title_alignment)
            disable_term_expansion_for_focused_title = bool(
                classification.focus_disable_term_expansion
            )
        else:
            explicit_title_reference = has_explicit_title_reference(question)
            referential_title_anchor = None
            title_alignment_query = question
            prefer_title_alignment = should_prefer_title_alignment(
                question=question,
                classification=classification,
            )
            strict_title_alignment = bool(
                explicit_title_reference and not has_comparison_cue(question)
            )
            disable_term_expansion_for_focused_title = False
        summary_style_request = is_summary_style_request(question, classification)
        effective_block_type_exclude = list(
            getattr(classification, "block_type_exclude", None) or []
        )
        if summary_style_request and not classification.block_type_filter:
            for excluded_type in SUMMARY_BLOCK_TYPE_EXCLUDE:
                if excluded_type not in effective_block_type_exclude:
                    effective_block_type_exclude.append(excluded_type)
        if trace is not None:
            trace.record(
                "retrieval.query_rewrite",
                {
                    "query_rewrite": query_rewritten,
                    "original_query": question,
                    "retrieval_content_query": classification.retrieval_content_query,
                    "retrieval_content_confidence": round(
                        float(classification.retrieval_content_confidence or 0.0), 4
                    ),
                    "retrieval_content_reasons": list(
                        classification.retrieval_content_reasons or []
                    ),
                    "rewritten_query": retrieval_query if query_rewritten else None,
                    "summary_style_request": summary_style_request,
                    "prefer_title_alignment": prefer_title_alignment,
                    "strict_title_alignment": strict_title_alignment,
                    "explicit_title_reference": explicit_title_reference,
                    "referential_title_anchor": referential_title_anchor,
                    "focus_query_rewritten": classification.focus_query_rewritten,
                    "term_expansion_enabled": not disable_term_expansion_for_focused_title,
                },
            )
        comparison_style_request = has_comparison_cue(question)
        base_retrieval_kwargs: dict[str, object] = {
            "max_score": None,
            "year_filter": classification.year_filter,
            "category_filter": classification.category_filter,
            "extension_filter": classification.file_type_filter,
            "filename_filter": classification.filename_filter,
            "filename_exclude": classification.filename_exclude,
            "block_type_filter": classification.block_type_filter,
            "block_type_exclude": effective_block_type_exclude or None,
            "section_filter": classification.section_filter,
            "file_ids_filter": file_ids,
            "exclude_upload_sources": not bool(file_ids),
            "prefer_substantive_sections": summary_style_request,
            "prefer_title_alignment": prefer_title_alignment,
            "title_alignment_query": title_alignment_query if prefer_title_alignment else None,
            "strict_title_alignment": strict_title_alignment,
            "enable_term_expansion": not disable_term_expansion_for_focused_title,
            "prefer_within_file_diversity": summary_style_request or comparison_style_request,
            "query_type": effective_query_type,
            "db": db,
        }

        async def _retrieve_for_query(
            query_text: str,
            *,
            top_k: int,
            trace_obj: object | None,
        ) -> tuple[list[dict], dict[str, float]]:
            """Internal helper for retrieve for query."""
            timing_output: dict[str, float] = {}
            retrieved_chunks = await retrieve_chunks(
                query=query_text,
                top_k=top_k,
                trace=trace_obj,
                timing_output=timing_output,
                **base_retrieval_kwargs,
            )
            return retrieved_chunks, timing_output

        agent_planning_start = time.perf_counter() if agent_mode else None
        if agent_mode:
            agent_retrieval_queries = _resolve_agent_retrieval_queries(
                classification=classification,
                retrieval_query=retrieval_query,
            )
        else:
            agent_retrieval_queries = [retrieval_query]
        agent_plan_enabled = bool(agent_mode)
        agent_plan_step_2_description = (
            "Retrieving evidence from subqueries"
            if len(agent_retrieval_queries) > 1
            else "Retrieving evidence"
        )

        def _build_agent_event(
            kind: str,
            title: str,
            message: str,
            *,
            status: str = "running",
            tool_name: str | None = None,
            subquery_index: int | None = None,
            subquery_total: int | None = None,
            result_count: int | None = None,
            query: str | None = None,
        ) -> tuple[StreamSignalTag, dict[str, object]]:
            return (
                StreamSignalTag.AGENT_EVENT,
                {
                    "kind": kind,
                    "status": status,
                    "title": title,
                    "message": message,
                    "tool_name": tool_name,
                    "subquery_index": subquery_index,
                    "subquery_total": subquery_total,
                    "result_count": result_count,
                    "query": query,
                },
            )

        chunks: list[dict] = []
        if agent_plan_enabled:
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 1,
                    "description": "Analyzing the request",
                    "status": "running",
                },
            )
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 1,
                    "description": "Analyzing the request",
                    "status": "done",
                },
            )
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 2,
                    "description": agent_plan_step_2_description,
                    "status": "running",
                },
            )
        agent_planning_elapsed_ms = (
            (time.perf_counter() - agent_planning_start) * 1000.0
            if agent_planning_start is not None
            else None
        )
        if len(agent_retrieval_queries) > 1:
            per_query_top_k = max(
                4,
                min(
                    effective_top_k,
                    (effective_top_k // len(agent_retrieval_queries)) + 2,
                ),
            )
            seen_chunk_keys: set[tuple[object, ...]] = set()
            retrieval_start = time.perf_counter()
            if agent_plan_enabled:
                yield _build_agent_event(
                    "tool_call",
                    "Retrieving evidence",
                    "",
                    status="running",
                    tool_name="search_vectors",
                )
            # Agent subqueries are independent retrieval slices, so run them in parallel
            # to reduce time-to-first-answer without changing ranking or dedupe behavior.
            subquery_results = await asyncio.gather(
                *(
                    _retrieve_for_query(
                        agent_query,
                        top_k=per_query_top_k,
                        trace_obj=trace,
                    )
                    for agent_query in agent_retrieval_queries
                )
            )
            for subquery_index, (
                agent_query,
                (subquery_chunks, subquery_timing),
            ) in enumerate(zip(agent_retrieval_queries, subquery_results, strict=True), start=1):
                for timing_key, timing_value in subquery_timing.items():
                    if isinstance(timing_value, (int, float)):
                        numeric_timing = float(timing_value)
                        current_timing = retrieval_timing.get(timing_key)
                        if timing_key.endswith("_at_s"):
                            retrieval_timing[timing_key] = (
                                max(float(current_timing), numeric_timing)
                                if isinstance(current_timing, (int, float))
                                else numeric_timing
                            )
                        else:
                            retrieval_timing[timing_key] = (
                                float(current_timing)
                                if isinstance(current_timing, (int, float))
                                else 0.0
                            ) + numeric_timing
                for chunk in subquery_chunks:
                    dedupe_key = _chunk_dedupe_key(chunk)
                    if dedupe_key in seen_chunk_keys:
                        continue
                    seen_chunk_keys.add(dedupe_key)
                    chunks.append(chunk)
                    if len(chunks) >= effective_top_k:
                        break
                if trace is not None:
                    trace.record(
                        "agent_multi_query_retrieval_pass",
                        {
                            "pass_index": subquery_index,
                            "query": agent_query,
                            "returned_chunks": len(subquery_chunks),
                            "merged_chunks": len(chunks),
                        },
                    )
                if len(chunks) >= effective_top_k:
                    break
            if trace is not None:
                trace.record(
                    "agent_multi_query_retrieval",
                    {
                        "enabled": True,
                        "subquery_count": len(agent_retrieval_queries),
                        "returned_chunks": len(chunks),
                        "per_query_top_k": per_query_top_k,
                    },
                )
            if agent_plan_enabled:
                yield _build_agent_event(
                    "observation",
                    "Retrieving evidence",
                    "",
                    status="done",
                    tool_name="search_vectors",
                )
        else:
            retrieval_start = time.perf_counter()
            if agent_plan_enabled:
                yield _build_agent_event(
                    "tool_call",
                    "Retrieving evidence",
                    "",
                    status="running",
                    tool_name="search_vectors",
                    query=retrieval_query,
                )
            chunks, retrieval_timing = await _retrieve_for_query(
                retrieval_query,
                top_k=effective_top_k,
                trace_obj=trace,
            )
            if agent_plan_enabled:
                yield _build_agent_event(
                    "observation",
                    "Retrieving evidence",
                    "",
                    status="done",
                    tool_name="search_vectors",
                    query=retrieval_query,
                )
        if agent_plan_enabled:
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 2,
                    "description": agent_plan_step_2_description,
                    "status": "done",
                },
            )
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 3,
                    "description": "Generating answer",
                    "status": "running",
                },
            )
        if summary_style_request and explicit_title_reference:
            evidence_profile = evaluate_substantive_evidence(chunks)
            dominant_ratio = _dominant_file_ratio(chunks)
            weak_summary_evidence = (
                float(evidence_profile.get("substantive_ratio") or 0.0)
                < _WEAK_SUMMARY_SUBSTANTIVE_RATIO_THRESHOLD
                and dominant_ratio >= _WEAK_SUMMARY_DOMINANT_FILE_RATIO_THRESHOLD
            )
            if weak_summary_evidence:
                retry_timing: dict[str, float] = {}
                retry_chunks = await retrieve_chunks(
                    query=question,
                    top_k=effective_top_k,
                    max_score=None,
                    year_filter=classification.year_filter,
                    category_filter=classification.category_filter,
                    extension_filter=classification.file_type_filter,
                    filename_filter=classification.filename_filter,
                    filename_exclude=classification.filename_exclude,
                    block_type_filter=classification.block_type_filter,
                    block_type_exclude=effective_block_type_exclude or None,
                    section_filter=classification.section_filter,
                    file_ids_filter=file_ids,
                    exclude_upload_sources=not bool(file_ids),
                    prefer_substantive_sections=True,
                    prefer_title_alignment=True,
                    title_alignment_query=question,
                    strict_title_alignment=True,
                    enable_term_expansion=False,
                    prefer_within_file_diversity=True,
                    query_type=effective_query_type,
                    db=db,
                    trace=None,
                    timing_output=retry_timing,
                )
                retry_evidence_profile = evaluate_substantive_evidence(retry_chunks)
                if retry_chunks and float(
                    retry_evidence_profile.get("substantive_ratio") or 0.0
                ) >= float(evidence_profile.get("substantive_ratio") or 0.0):
                    chunks = retry_chunks
                    retrieval_timing = retry_timing
                if trace is not None:
                    trace.record(
                        "retrieval.summary_retry",
                        {
                            "triggered": True,
                            "weak_summary_evidence": weak_summary_evidence,
                            "initial_substantive_ratio": round(
                                float(evidence_profile.get("substantive_ratio") or 0.0), 4
                            ),
                            "retry_substantive_ratio": round(
                                float(retry_evidence_profile.get("substantive_ratio") or 0.0), 4
                            ),
                            "initial_dominant_file_ratio": round(dominant_ratio, 4),
                            "retry_used": bool(chunks is retry_chunks and retry_chunks),
                        },
                    )
        if (
            comparison_style_request
            and _distinct_file_count(chunks) < _WEAK_COMPARISON_DISTINCT_FILE_THRESHOLD
        ):
            initial_distinct_file_count = _distinct_file_count(chunks)
            comparison_retry_timing: dict[str, float] = {}
            comparison_retry_chunks = await retrieve_chunks(
                query=question,
                top_k=min(max(effective_top_k + 4, effective_top_k), 16),
                max_score=None,
                year_filter=classification.year_filter,
                category_filter=classification.category_filter,
                extension_filter=classification.file_type_filter,
                filename_filter=classification.filename_filter,
                filename_exclude=classification.filename_exclude,
                block_type_filter=classification.block_type_filter,
                block_type_exclude=effective_block_type_exclude or None,
                section_filter=classification.section_filter,
                file_ids_filter=file_ids,
                exclude_upload_sources=not bool(file_ids),
                prefer_substantive_sections=False,
                prefer_title_alignment=False,
                title_alignment_query=None,
                strict_title_alignment=False,
                enable_term_expansion=True,
                prefer_within_file_diversity=True,
                query_type=effective_query_type,
                db=db,
                trace=None,
                timing_output=comparison_retry_timing,
            )
            comparison_retry_distinct_file_count = _distinct_file_count(comparison_retry_chunks)
            if comparison_retry_distinct_file_count > initial_distinct_file_count:
                chunks = comparison_retry_chunks
                retrieval_timing = comparison_retry_timing
            if trace is not None:
                trace.record(
                    "retrieval.comparison_retry",
                    {
                        "triggered": True,
                        "initial_distinct_file_count": initial_distinct_file_count,
                        "retry_distinct_file_count": comparison_retry_distinct_file_count,
                        "retry_used": bool(
                            chunks is comparison_retry_chunks and comparison_retry_chunks
                        ),
                    },
        )
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000
        agent_retrieval_elapsed_ms = retrieval_elapsed_ms if agent_plan_enabled else None
        stream_summary: _generation_stream.StreamExecutionSummary | None = None

        if timing_context is not None:
            for timing_key in (
                "embed_complete_at_s",
                "search_complete_at_s",
                "retrieval_complete_at_s",
            ):
                timing_value = retrieval_timing.get(timing_key)
                if isinstance(timing_value, (int, float)):
                    current_timing = timing_context.get(timing_key)
                    timing_context[timing_key] = (
                        max(float(current_timing), float(timing_value))
                        if isinstance(current_timing, (int, float))
                        else float(timing_value)
                    )

        answerability_passed, answerability_score, answerability_threshold, min_chunks = (
            _evaluate_minimal_answerability(
                chunks,
                query_type=effective_query_type,
            )
        )

        if trace is not None:
            trace.record(
                "answerability_decision",
                {
                    "passed": answerability_passed,
                    "score": round(answerability_score, 4),
                    "threshold": answerability_threshold,
                    "chunk_count": len(chunks),
                    "min_chunks": min_chunks,
                    "mode": "minimal",
                },
            )
        if agent_plan_enabled and trace is not None:
            trace.record(
                "agent_planning",
                {
                    "enabled": True,
                    "duration_ms": round(agent_planning_elapsed_ms, 1)
                    if agent_planning_elapsed_ms is not None
                    else None,
                    "subquery_count": len(agent_retrieval_queries),
                    "multi_query": len(agent_retrieval_queries) > 1,
                },
            )
            trace.record(
                "agent_retrieval",
                {
                    "enabled": True,
                    "duration_ms": round(agent_retrieval_elapsed_ms, 1)
                    if agent_retrieval_elapsed_ms is not None
                    else None,
                    "subquery_count": len(agent_retrieval_queries),
                    "returned_chunks": len(chunks),
                    "multi_query": len(agent_retrieval_queries) > 1,
                },
            )

        if not answerability_passed:
            index_empty = False
            if len(chunks) == 0:
                total_chunks = await get_chunk_count(db)
                index_empty = total_chunks == 0
            if agent_plan_enabled:
                yield (
                    StreamSignalTag.PLAN_STEP,
                    {
                        "step_id": 3,
                        "description": "Generating answer",
                        "status": "empty",
                    },
                )
            yield (
                StreamSignalTag.METRICS,
                {
                    "query_type": effective_query_type,
                    "raw_chunks_count": len(chunks),
                    "retrieval_duration_ms": round(retrieval_elapsed_ms, 1),
                    "agent_mode": agent_plan_enabled,
                    "agent_planning_duration_ms": round(agent_planning_elapsed_ms, 1)
                    if agent_planning_elapsed_ms is not None
                    else None,
                    "agent_retrieval_duration_ms": round(agent_retrieval_elapsed_ms, 1)
                    if agent_retrieval_elapsed_ms is not None
                    else None,
                    "prompt_build_ms": "N/A",
                    "llm_submit_ms": round(stream_summary.submit_ms, 1)
                    if stream_summary is not None and stream_summary.submit_ms is not None
                    else "N/A",
                    "llm_queue_wait_ms": round(stream_summary.queue_wait_ms, 1)
                    if stream_summary is not None and stream_summary.queue_wait_ms is not None
                    else "N/A",
                    "llm_decode_first_token_ms": "N/A",
                    "answerability_passed": False,
                    "answerability_score": round(answerability_score, 4),
                    "answerability_threshold": answerability_threshold,
                    "answerability_min_chunks": min_chunks,
                    "generation_skipped": True,
                    "minimal_mode": True,
                    "index_empty": index_empty,
                },
            )
            if index_empty:
                yield EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE
            else:
                yield _INSUFFICIENT_CONTEXT_RESPONSE
            yield []
            return

        if _should_use_deterministic_file_discovery_response(classification):
            deterministic_start = time.perf_counter()
            answer_text, file_discovery = _build_file_discovery_response(
                question=question,
                chunks=chunks,
            )
            deterministic_elapsed_ms = (time.perf_counter() - deterministic_start) * 1000.0
            if trace is not None:
                trace.record(
                    "deterministic_file_discovery",
                    {
                        "enabled": True,
                        "file_count": file_discovery["total_count"],
                        "shown_count": file_discovery["shown_count"],
                        "search_term": file_discovery["search_term"],
                        "duration_ms": round(deterministic_elapsed_ms, 1),
                        "agent_mode": agent_plan_enabled,
                        "agent_planning_duration_ms": round(agent_planning_elapsed_ms, 1)
                        if agent_planning_elapsed_ms is not None
                        else None,
                        "agent_retrieval_duration_ms": round(agent_retrieval_elapsed_ms, 1)
                        if agent_retrieval_elapsed_ms is not None
                        else None,
                    },
                )
            yield (
                StreamSignalTag.METRICS,
                build_metrics_payload(
                    query_type=effective_query_type,
                    raw_chunks_count=len(chunks),
                    retrieval_duration_ms=round(retrieval_elapsed_ms, 1),
                    agent_mode=agent_plan_enabled,
                    agent_planning_duration_ms=round(agent_planning_elapsed_ms, 1)
                    if agent_planning_elapsed_ms is not None
                    else None,
                    agent_retrieval_duration_ms=round(agent_retrieval_elapsed_ms, 1)
                    if agent_retrieval_elapsed_ms is not None
                    else None,
                    prompt_build_ms="N/A",
                    llm_submit_ms="N/A",
                    llm_queue_wait_ms="N/A",
                    llm_decode_first_token_ms="N/A",
                    generation_skipped=True,
                    minimal_mode=True,
                    deterministic_file_discovery=True,
                ),
            )
            yield (StreamSignalTag.FILE_DISCOVERY, file_discovery)
            if agent_plan_enabled:
                yield (
                    StreamSignalTag.PLAN_STEP,
                    {
                        "step_id": 3,
                        "description": "Synthesizing a final answer from the combined evidence",
                        "status": "done",
                    },
                )
            yield answer_text
            sources = _generation_closeout.build_source_references(
                chunks=chunks,
                answer_text=answer_text,
                truncate_preview_fn=_truncate_preview,
                normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
            )
            _generation_closeout.record_sources_trace(trace=trace, sources=sources)
            yield sources
            return

        (
            format_requirements,
            output_constraints,
            max_tokens,
            reasoning_enabled,
            chunks,
            _,
        ) = _generation_runtime._apply_strict_format_prompt_controls(
            question=question,
            chunks=chunks,
            output_constraints={},
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            derive_format_requirements_fn=_structured_numeric._derive_format_requirements,
            action_hints=classification.action_hints,
            applied_degradations=[],
        )
        _apply_output_format_preferences(
            output_format=classification.output_format,
            format_requirements=format_requirements,
            output_constraints=output_constraints,
        )
        _apply_negation_preferences(
            is_negation_query=classification.is_negation_query,
            format_requirements=format_requirements,
        )
        stop_sequences = profile.get_stop_sequences(reasoning_enabled)
        generation_temperature, generation_top_p = _resolve_sampling_params(
            profile_temperature=profile.temperature,
            profile_top_p=profile.top_p,
            format_requirements=format_requirements,
        )
        if summary_style_request and explicit_title_reference:
            generation_temperature = min(generation_temperature, _SUMMARY_TITLE_MAX_TEMPERATURE)
            generation_top_p = min(generation_top_p, _SUMMARY_TITLE_MAX_TOP_P)

        prompt_build_start = time.perf_counter()
        if timing_context is not None:
            timing_context.setdefault("prompt_build_started_at_s", time.time())
        messages = build_messages(
            BuildMessagesRequest(
                question=question,
                context_chunks=chunks,
                history=history,
                output_constraints=output_constraints,
                format_requirements=format_requirements,
                model_profile=profile,
                chat_mode="researcher",
                specialization_id=specialization_id,
                agent_mode=agent_mode,
                agent_synthesis_focus=classification.agent_synthesis_focus,
            )
        )
        messages = profile.prepare_messages(
            messages,
            effective_query_type,
            reasoning_enabled=reasoning_enabled,
        )
        prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000
        if timing_context is not None:
            timing_context["prompt_build_complete_at_s"] = time.time()

        if trace is not None:
            trace.record(
                "prompt",
                {
                    "messages_count": len(messages),
                    "context_chunks": len(chunks),
                    "history_messages": len(history) if history else 0,
                    "effective_history_limit": resolve_history_limit("researcher"),
                    "chat_mode": "researcher",
                    "reasoning_enabled": reasoning_enabled,
                    "output_constraints": output_constraints,
                    "format_requirements": format_requirements,
                    "minimal_mode": True,
                },
            )

        llm_start = time.perf_counter()
        first_token_ms: float | None = None
        token_count = 0
        answer_parts: list[str] = []
        stream_summary: _generation_stream.StreamExecutionSummary | None = None
        if _should_prepend_deterministic_extraction_heading(
            question=question,
            classification=classification,
        ):
            answer_parts.append(_DETERMINISTIC_EXTRACTION_HEADING)
            yield _DETERMINISTIC_EXTRACTION_HEADING
        if timing_context is not None:
            timing_context.setdefault("generation_started_at_s", time.time())
        async for item in _generation_stream.stream_generation_with_budget(
            messages=messages,
            max_tokens=max_tokens,
            temperature=generation_temperature,
            top_p=generation_top_p,
            timeout_seconds=timeout_seconds,
            stop_sequences=stop_sequences,
            dedupe_insufficient_context_after_stream=bool(
                profile.dedupe_insufficient_context_after_stream
            ),
            insufficient_context_response=_INSUFFICIENT_CONTEXT_RESPONSE,
            applied_degradations=[],
            output_contract_plan=None,
            collapse_duplicate_message_fn=_collapse_duplicate_insufficient_context_message,
            chat_template_kwargs_override=chat_template_kwargs_override,
            stream_llm_fn=stream_llm,
            timing_context=timing_context,
        ):
            if isinstance(item, tuple):
                if item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
                    stream_summary = item[1]
                    continue
                if item[0] == StreamSignalTag.BUDGET_CHECKPOINT:
                    # Preserve current minimal RAG behavior: do not emit budget checkpoints.
                    continue
                yield item
                continue
            token_count += 1
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - llm_start) * 1000
            answer_parts.append(item)
            yield item
        llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
        if timing_context is not None:
            timing_context["generation_complete_at_s"] = time.time()
        if stream_summary is not None:
            token_count = stream_summary.token_count
            first_token_ms = stream_summary.first_token_ms
            llm_elapsed_ms = stream_summary.total_elapsed_ms
            if timing_context is not None:
                if stream_summary.submit_at_s is not None:
                    timing_context["payload_sent_to_model_runtime_at_s"] = stream_summary.submit_at_s
                if stream_summary.first_token_at_s is not None:
                    timing_context["runtime_first_token_at_s"] = stream_summary.first_token_at_s
                if stream_summary.runtime_metrics is not None:
                    timing_context["runtime_metrics"] = stream_summary.runtime_metrics
        answer_text = "".join(answer_parts)

        if trace is not None:
            trace.record(
                "llm",
                {
                    "token_count": token_count,
                    "max_tokens": max_tokens,
                    "first_token_ms": round(first_token_ms, 1)
                    if first_token_ms is not None
                    else None,
                    "total_elapsed_ms": round(llm_elapsed_ms, 1),
                    "model_profile": profile.name,
                    "minimal_mode": True,
                    "agent_mode": agent_plan_enabled,
                    "agent_generation_duration_ms": round(llm_elapsed_ms, 1)
                    if agent_plan_enabled
                    else None,
                    "agent_generation_first_token_ms": round(first_token_ms, 1)
                    if first_token_ms is not None and agent_plan_enabled
                    else None,
                    "llm_submit_ms": round(stream_summary.submit_ms, 1)
                    if stream_summary is not None and stream_summary.submit_ms is not None
                    else None,
                    "llm_queue_wait_ms": round(stream_summary.queue_wait_ms, 1)
                    if stream_summary is not None and stream_summary.queue_wait_ms is not None
                    else None,
                },
            )
        if agent_plan_enabled and trace is not None:
            trace.record(
                "agent_generation",
                {
                    "enabled": True,
                    "duration_ms": round(llm_elapsed_ms, 1),
                    "first_token_ms": round(first_token_ms, 1)
                    if first_token_ms is not None
                    else None,
                    "token_count": token_count,
                    "model_profile": profile.name,
                },
            )

        yield (
            StreamSignalTag.METRICS,
            build_metrics_payload(
                query_type=effective_query_type,
                raw_chunks_count=len(chunks),
                retrieval_duration_ms=round(retrieval_elapsed_ms, 1),
                agent_mode=agent_plan_enabled,
                agent_planning_duration_ms=round(agent_planning_elapsed_ms, 1)
                if agent_planning_elapsed_ms is not None
                else None,
                agent_retrieval_duration_ms=round(agent_retrieval_elapsed_ms, 1)
                if agent_retrieval_elapsed_ms is not None
                else None,
                agent_generation_duration_ms=round(llm_elapsed_ms, 1)
                if agent_plan_enabled
                else None,
                agent_generation_first_token_ms=round(first_token_ms, 1)
                if first_token_ms is not None and agent_plan_enabled
                else None,
                first_token_latency_ms=round(first_token_ms, 1)
                if first_token_ms is not None
                else None,
                stream_duration_ms=round(llm_elapsed_ms, 1),
                embed_ms=retrieval_timing.get("embed_ms"),
                vector_search_ms=retrieval_timing.get("vector_search_ms"),
                rerank_ms=retrieval_timing.get("rerank_ms"),
                prompt_build_ms=round(prompt_build_ms, 1),
                llm_submit_ms=round(stream_summary.submit_ms, 1)
                if stream_summary is not None and stream_summary.submit_ms is not None
                else "N/A",
                llm_queue_wait_ms=round(stream_summary.queue_wait_ms, 1)
                if stream_summary is not None and stream_summary.queue_wait_ms is not None
                else "N/A",
                llm_decode_first_token_ms=(
                    round(first_token_ms, 1) if first_token_ms is not None else "N/A"
                ),
                answerability_passed=True,
                answerability_score=round(answerability_score, 4),
                answerability_threshold=answerability_threshold,
                answerability_min_chunks=min_chunks,
                generation_skipped=False,
                minimal_mode=True,
            ),
        )

        sources = _generation_closeout.build_source_references(
            chunks=chunks,
            answer_text=answer_text,
            truncate_preview_fn=_truncate_preview,
            normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
        )
        _generation_closeout.record_sources_trace(trace=trace, sources=sources)
        if agent_plan_enabled:
            yield (
                StreamSignalTag.PLAN_STEP,
                {
                    "step_id": 3,
                    "description": "Generating answer",
                    "status": "done",
                },
            )
        yield sources

    async def handle(
        self,
        question: str,
        classification: QueryClassification,
        history: list[ChatMessage] | None,
        db: aiosqlite.Connection,
        trace: object | None,
        diagnostics_context: dict[str, object] | None = None,
        chat_id: str | None = None,
        file_ids: list[int] | None = None,
        chat_mode: str | None = None,
        specialization_id: str | None = None,
        agent_mode: bool = False,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle RAG query using the single minimal runtime path.
        """
        timing_context: dict[str, object] | None = None
        if isinstance(diagnostics_context, dict):
            timing_candidate = diagnostics_context.get("timing")
            if isinstance(timing_candidate, dict):
                timing_context = timing_candidate
        try:
            if diagnostics_context is not None and trace is not None:
                trace.record("diagnostics_context", diagnostics_context)
            async for item in self._handle_minimal_mode(
                question=question,
                classification=classification,
                history=history,
                db=db,
                trace=trace,
                timing_context=timing_context,
                file_ids=file_ids,
                specialization_id=specialization_id,
                agent_mode=agent_mode,
            ):
                yield item
        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error("rag_handler_failed", error=str(exc), exc_info=True)
            yield to_client_error_message(exc)
            yield []
