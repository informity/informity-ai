# ==============================================================================
# Informity AI — Term Dictionary Builder
# Post-index deterministic extraction and versioned dictionary build.
# ==============================================================================

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import aiosqlite
import structlog
from nameparser import HumanName

from informity.config import settings
from informity.db.sqlite import (
    delete_term_dictionary_version,
    finalize_term_dictionary_build_run,
    get_latest_term_dictionary_build_run,
    get_term_dictionary_current_version,
    get_term_dictionary_source_rows,
    insert_term_alias,
    insert_term_entry,
    insert_term_evidence,
    set_term_dictionary_current_version,
    start_term_dictionary_build_run,
    update_term_dictionary_build_run_progress,
)
from informity.indexer.term_dictionary_quality import evaluate_term_dictionary_quality
from informity.llm.term_dictionary import normalize_term_text, term_dictionary_enabled

log = structlog.get_logger(__name__)
_TERM_DICTIONARY_BUILD_GUARD_EXCEPTIONS = (
    aiosqlite.Error,
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    LookupError,
)

_TERM_DEFINITION_PATTERN = re.compile(
    r'\b([A-Za-z][A-Za-z0-9/&-]*(?:\s+[A-Za-z][A-Za-z0-9/&-]*){1,6})\s+\(([A-Z][A-Z0-9]{1,9})\)'
)
_ACRONYM_DEFINITION_PATTERN = re.compile(
    r'\b([A-Z][A-Z0-9]{1,9})\s+\(([A-Za-z][A-Za-z0-9/&-]*(?:\s+[A-Za-z][A-Za-z0-9/&-]*){1,6})\)'
)
_SINGLE_LETTER_ACRONYM_SEQUENCE_PATTERN = re.compile(
    r'\b([A-Z](?:\s+[A-Z0-9]){1,9})\s*(?=\()'
)
_SINGLE_LETTER_ACRONYM_IN_PARENS_PATTERN = re.compile(
    r'\(([A-Z](?:\s+[A-Z0-9]){1,9})\)'
)
_OCR_HYPHENATED_LINEBREAK_PATTERN = re.compile(r'([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})')
_OCR_SPACED_HYPHEN_PATTERN = re.compile(r'([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})')
_WORD_TOKEN_PATTERN = re.compile(r'[a-z0-9]+')
_PERSON_NAME_PATTERN = re.compile(r'\b([A-Z][a-z]{1,29}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,29})\b')
_PERSON_STRONG_LEFT_CUE_PATTERN = re.compile(
    r'\b(mr|mrs|ms|dr|prof|professor|president|governor|senator|representative|director|ceo|cfo)\.?\s+$',
    re.IGNORECASE,
)
_PERSON_ACTION_CONTEXT_PATTERN = re.compile(
    r'\b(said|says|met|emailed|called|wrote|signed|approved|reviewed|presented|introduced|joined)\b',
    re.IGNORECASE,
)
_PERSON_FIELD_LABEL_CUE_PATTERN = re.compile(
    r'\b(employee|owner|recipient|borrower|seller|buyer|insured|contact)\s+name[:\s]*$',
    re.IGNORECASE,
)
_PERSON_ROLE_LEFT_CUE_PATTERN = re.compile(
    r'\b(employee|manager|director|officer|owner|president|ceo|cfo|treasurer|supervisor)\s+$',
    re.IGNORECASE,
)
_PERSON_CAMEL_SPLIT_PATTERN = re.compile(r'([a-z])([A-Z])')
_BOILERPLATE_PHRASES: tuple[str, ...] = (
    'see instructions',
    'see instruction',
    'see note above',
    'refer to instructions',
    'refer to note above',
    'for more information',
)
_PERSON_NAME_STOPWORDS: set[str] = {
    'about',
    'across',
    'after',
    'all',
    'analysis',
    'and',
    'any',
    'are',
    'before',
    'between',
    'chapter',
    'data',
    'details',
    'document',
    'documents',
    'during',
    'each',
    'evidence',
    'figure',
    'figures',
    'file',
    'files',
    'for',
    'from',
    'guide',
    'how',
    'in',
    'index',
    'into',
    'list',
    'month',
    'months',
    'more',
    'note',
    'notes',
    'on',
    'or',
    'overview',
    'page',
    'pages',
    'part',
    'policy',
    'process',
    'project',
    'report',
    'section',
    'sections',
    'summary',
    'table',
    'tables',
    'the',
    'this',
    'through',
    'to',
    'topic',
    'update',
    'week',
    'weeks',
    'with',
    'year',
    'years',
}
_ORGANIZATION_SUFFIXES: set[str] = {
    'inc',
    'llc',
    'ltd',
    'corp',
    'co',
    'company',
    'group',
    'committee',
    'agency',
    'department',
    'office',
    'bank',
    'university',
}
_PERSON_DISALLOWED_TAIL_TOKENS: set[str] = {
    'bill',
    'adjustment',
    'analysis',
    'certificate',
    'coverage',
    'detail',
    'details',
    'figure',
    'figures',
    'history',
    'information',
    'line',
    'lines',
    'dept',
    'department',
    'policy',
    'premium',
    'record',
    'records',
    'report',
    'section',
    'sections',
    'settlement',
    'summary',
    'table',
    'tables',
    'tax',
    'first',
    'second',
    'third',
    'name',
    'no',
    'number',
}
_PERSON_DISALLOWED_LEAD_TOKENS: set[str] = {
    'account',
    'additional',
    'annual',
    'attachment',
    'business',
    'city',
    'control',
    'effective',
    'federal',
    'general',
    'health',
    'homeowners',
    'information',
    'internal',
    'itemized',
    'local',
    'mailing',
    'mortgage',
    'payment',
    'property',
    'production',
    'retirement',
    'review',
    'schedule',
    'social',
    'state',
    'statement',
    'student',
    'taxpayer',
    'title',
    'treasury',
}
_MAX_EVIDENCE_PER_CANDIDATE = 4


@dataclass(frozen=True, slots=True)
class _EntityPolicy:
    min_confidence: float
    min_tokens: int
    require_observed_alias: bool = True
    allow_self_alias: bool = False


_ENTITY_POLICIES: dict[str, _EntityPolicy] = {
    'acronym': _EntityPolicy(
        min_confidence=0.65,
        min_tokens=2,
        require_observed_alias=True,
        allow_self_alias=False,
    ),
    'person_name': _EntityPolicy(
        min_confidence=0.75,
        min_tokens=2,
        require_observed_alias=False,
        allow_self_alias=True,
    ),
    'organization': _EntityPolicy(
        min_confidence=0.75,
        min_tokens=2,
        require_observed_alias=False,
        allow_self_alias=True,
    ),
    'location': _EntityPolicy(
        min_confidence=0.75,
        min_tokens=2,
        require_observed_alias=False,
        allow_self_alias=True,
    ),
    'numeric_id': _EntityPolicy(
        min_confidence=0.80,
        min_tokens=1,
        require_observed_alias=True,
        allow_self_alias=False,
    ),
}


@dataclass(slots=True)
class _Candidate:
    canonical: str
    normalized_canonical: str
    term_type: str
    confidence: float
    aliases: dict[str, tuple[str, float]] = field(default_factory=dict)  # normalized_alias -> (raw_alias, confidence)
    evidence: list[tuple[int | None, int | None, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class _ExtractionContext:
    raw_text: str
    processed_text: str
    file_id: int | None
    chunk_id: int | None


def _builder_enabled() -> bool:
    if not term_dictionary_enabled():
        return False
    return bool(settings.term_dictionary_build_enabled)


def _batch_size() -> int:
    try:
        value = int(settings.term_dictionary_build_batch_size)
    except (TypeError, ValueError):
        value = 500
    return max(50, min(2000, value))


def _entity_toggle_enabled(setting_name: str, *, default: bool) -> bool:
    raw = getattr(settings, setting_name, default)
    return bool(raw)


def _enabled_entity_types() -> set[str]:
    toggles = {
        'acronym': _entity_toggle_enabled('entity_extract_acronym', default=True),
        'person_name': _entity_toggle_enabled('entity_extract_person_name', default=False),
        'organization': _entity_toggle_enabled('entity_extract_organization', default=False),
        'location': _entity_toggle_enabled('entity_extract_location', default=False),
        'numeric_id': _entity_toggle_enabled('entity_extract_numeric_id', default=False),
    }
    return {entity_type for entity_type, enabled in toggles.items() if enabled}


def _trim_snippet(text: str, limit: int = 220) -> str:
    snippet = ' '.join(str(text or '').split())
    return snippet[:limit]


def _preprocess_text_for_matching(text: str) -> str:
    value = str(text or '')
    if not value:
        return ''
    value = _OCR_HYPHENATED_LINEBREAK_PATTERN.sub(r'\1\2', value)
    value = _OCR_SPACED_HYPHEN_PATTERN.sub(r'\1-\2', value)
    value = _SINGLE_LETTER_ACRONYM_SEQUENCE_PATTERN.sub(
        lambda m: m.group(1).replace(' ', ''),
        value,
    )
    value = _SINGLE_LETTER_ACRONYM_IN_PARENS_PATTERN.sub(
        lambda m: f"({m.group(1).replace(' ', '')})",
        value,
    )
    return value


def _looks_like_person_name(long_term: str, acronym: str) -> bool:
    tokens = [token for token in re.split(r'\s+', long_term.strip()) if token]
    if len(tokens) != 2:
        return False
    if not all(token.isalpha() for token in tokens):
        return False
    if not all(token[0:1].isupper() and token[1:].islower() for token in tokens):
        return False
    initials = ''.join(token[0] for token in tokens).upper()
    return initials == acronym.strip().upper()


def _canonical_has_minimum_shape(canonical: str, *, term_type: str) -> bool:
    policy = _ENTITY_POLICIES.get(term_type, _ENTITY_POLICIES['acronym'])
    normalized = normalize_term_text(canonical)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _BOILERPLATE_PHRASES):
        return False
    tokens = [token for token in _WORD_TOKEN_PATTERN.findall(normalized) if token]
    if len(tokens) < policy.min_tokens:
        return False
    if len(set(tokens)) < min(2, len(tokens)):
        return False
    return not all(len(token) <= 2 for token in tokens)


def _score_candidate_confidence(*, long_term: str, acronym: str, extraction_method: str) -> float:
    score = 0.95
    initials = ''.join(token[0] for token in re.findall(r'[A-Za-z]+', long_term)).upper()
    if not initials or not acronym.upper().startswith(initials[: min(len(initials), len(acronym))]):
        score -= 0.15
    if any(ch.isdigit() for ch in long_term):
        score -= 0.10
    if extraction_method.endswith('ocr_normalized'):
        score -= 0.08
    if any(phrase in normalize_term_text(long_term) for phrase in _BOILERPLATE_PHRASES):
        score = min(score, 0.35)
    return max(0.0, min(1.0, score))


def _is_valid_person_name_candidate(candidate: str) -> bool:
    tokens = [token for token in candidate.split() if token]
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if len(tokens) == 3:
        middle = tokens[1].rstrip('.')
        if not (len(middle) == 1 and middle.isalpha() and middle.isupper()):
            return False
    if any(len(token) < 2 for token in tokens):
        return False
    first = tokens[0]
    last = tokens[-1]
    if not (first[0:1].isupper() and first[1:].islower()):
        return False
    if not (last[0:1].isupper() and last[1:].islower()):
        return False

    lower_tokens = [token.casefold().rstrip('.') for token in tokens]
    if any(token in _PERSON_NAME_STOPWORDS for token in lower_tokens):
        return False
    if lower_tokens[0] in _PERSON_DISALLOWED_LEAD_TOKENS:
        return False
    if lower_tokens[-1] in _ORGANIZATION_SUFFIXES:
        return False
    if lower_tokens[-1] in _PERSON_DISALLOWED_TAIL_TOKENS:
        return False
    return not any(any(char.isdigit() for char in token) for token in tokens)


def _score_person_name_confidence(*, match_text: str, raw_text: str, mention_count: int, start_index: int) -> float:
    score = 0.35
    if mention_count > 1:
        score += min(0.25, 0.12 * (mention_count - 1))

    prefix_window = raw_text[max(0, start_index - 36):start_index]
    if _PERSON_STRONG_LEFT_CUE_PATTERN.search(prefix_window):
        score += 0.45
    if _PERSON_FIELD_LABEL_CUE_PATTERN.search(prefix_window):
        score += 0.40
    if _PERSON_ROLE_LEFT_CUE_PATTERN.search(prefix_window):
        score += 0.18
    around_window = raw_text[max(0, start_index - 40): min(len(raw_text), start_index + 56)]
    if _PERSON_ACTION_CONTEXT_PATTERN.search(around_window):
        score += 0.10

    normalized = normalize_term_text(match_text)
    if normalized in {'john doe', 'jane doe'}:
        score -= 0.35

    return max(0.0, min(1.0, score))


def _canonicalize_person_name(name: str) -> str:
    tokens = [token for token in name.split() if token]
    if len(tokens) < 2:
        return name.strip()
    fallback_first = tokens[0].rstrip('.')
    fallback_last = tokens[-1].rstrip('.')
    fallback = f'{fallback_first.capitalize()} {fallback_last.capitalize()}'

    parsed = HumanName(name)
    first = str(parsed.first or '').strip().rstrip('.')
    last = str(parsed.last or '').strip().rstrip('.')
    if not first or not last:
        return fallback
    if not (first[0:1].isalpha() and last[0:1].isalpha()):
        return fallback
    return f'{first.capitalize()} {last.capitalize()}'


def _add_candidate(
    candidates: dict[tuple[str, str], _Candidate],
    *,
    canonical: str,
    alias: str,
    term_type: str,
    confidence: float,
    file_id: int | None,
    chunk_id: int | None,
    evidence_snippet: str,
    extraction_method: str,
    allow_self_alias: bool = False,
) -> None:
    canonical_norm = normalize_term_text(canonical)
    alias_norm = normalize_term_text(alias)
    if not canonical_norm or not alias_norm:
        return
    if canonical_norm == alias_norm and not allow_self_alias:
        return
    if not _canonical_has_minimum_shape(canonical, term_type=term_type):
        return

    key = (term_type, canonical_norm)
    existing = candidates.get(key)
    if existing is None:
        existing = _Candidate(
            canonical=canonical.strip(),
            normalized_canonical=canonical_norm,
            term_type=term_type,
            confidence=confidence,
        )
        candidates[key] = existing

    if confidence > existing.confidence:
        existing.confidence = confidence
    if alias_norm not in existing.aliases or confidence > existing.aliases[alias_norm][1]:
        existing.aliases[alias_norm] = (alias.strip(), confidence)

    if len(existing.evidence) < _MAX_EVIDENCE_PER_CANDIDATE:
        existing.evidence.append((file_id, chunk_id, _trim_snippet(evidence_snippet), extraction_method))


def _extract_acronym_candidates(ctx: _ExtractionContext, out: dict[tuple[str, str], _Candidate]) -> None:
    def _extract_from_text(text: str, *, normalized_pass: bool) -> None:
        long_acronym_method = (
            'definition_pair_long_acronym_ocr_normalized'
            if normalized_pass
            else 'definition_pair_long_acronym'
        )
        acronym_long_method = (
            'definition_pair_acronym_long_ocr_normalized'
            if normalized_pass
            else 'definition_pair_acronym_long'
        )

        for match in _TERM_DEFINITION_PATTERN.finditer(text):
            long_term = match.group(1).strip()
            acronym = match.group(2).strip()
            if len(acronym) < 2:
                continue
            if _looks_like_person_name(long_term, acronym):
                continue
            _add_candidate(
                out,
                canonical=long_term,
                alias=acronym,
                term_type='acronym',
                confidence=_score_candidate_confidence(
                    long_term=long_term,
                    acronym=acronym,
                    extraction_method=long_acronym_method,
                ),
                file_id=ctx.file_id,
                chunk_id=ctx.chunk_id,
                evidence_snippet=match.group(0),
                extraction_method=long_acronym_method,
            )

        for match in _ACRONYM_DEFINITION_PATTERN.finditer(text):
            acronym = match.group(1).strip()
            long_term = match.group(2).strip()
            if len(acronym) < 2:
                continue
            if _looks_like_person_name(long_term, acronym):
                continue
            _add_candidate(
                out,
                canonical=long_term,
                alias=acronym,
                term_type='acronym',
                confidence=_score_candidate_confidence(
                    long_term=long_term,
                    acronym=acronym,
                    extraction_method=acronym_long_method,
                ),
                file_id=ctx.file_id,
                chunk_id=ctx.chunk_id,
                evidence_snippet=match.group(0),
                extraction_method=acronym_long_method,
            )

    _extract_from_text(ctx.raw_text, normalized_pass=False)
    if ctx.processed_text and ctx.processed_text != ctx.raw_text:
        _extract_from_text(ctx.processed_text, normalized_pass=True)


def _extract_person_name_candidates(ctx: _ExtractionContext, out: dict[tuple[str, str], _Candidate]) -> None:
    scan_text = _PERSON_CAMEL_SPLIT_PATTERN.sub(r'\1 \2', ctx.raw_text)
    mention_counts: dict[str, int] = {}
    accepted: list[tuple[str, str, int, str]] = []
    matches = list(_PERSON_NAME_PATTERN.finditer(scan_text))
    if not matches:
        return

    for match in matches:
        raw_name = match.group(1).strip()
        if not _is_valid_person_name_candidate(raw_name):
            continue
        canonical = _canonicalize_person_name(raw_name)
        canonical_norm = normalize_term_text(canonical)
        if not canonical_norm:
            continue
        mention_counts[canonical_norm] = mention_counts.get(canonical_norm, 0) + 1
        accepted.append((raw_name, canonical, match.start(1), match.group(0)))

    for raw_name, canonical, start_index, evidence_snippet in accepted:
        canonical_norm = normalize_term_text(canonical)
        if canonical_norm not in mention_counts:
            continue
        confidence = _score_person_name_confidence(
            match_text=raw_name,
            raw_text=scan_text,
            mention_count=mention_counts[canonical_norm],
            start_index=start_index,
        )
        if confidence < _ENTITY_POLICIES['person_name'].min_confidence:
            continue
        _add_candidate(
            out,
            canonical=canonical,
            alias=raw_name,
            term_type='person_name',
            confidence=confidence,
            file_id=ctx.file_id,
            chunk_id=ctx.chunk_id,
            evidence_snippet=evidence_snippet,
            extraction_method='person_name_span_v1',
            allow_self_alias=_ENTITY_POLICIES['person_name'].allow_self_alias,
        )


_EXTRACTOR_REGISTRY: dict[str, Callable[[_ExtractionContext, dict[tuple[str, str], _Candidate]], None]] = {
    'acronym': _extract_acronym_candidates,
    'person_name': _extract_person_name_candidates,
}


def _extract_candidates_from_chunk(
    *,
    content: str,
    file_id: int | None,
    chunk_id: int | None,
    out: dict[tuple[str, str], _Candidate],
    enabled_entity_types: set[str] | None = None,
) -> None:
    effective_enabled = enabled_entity_types or _enabled_entity_types()
    if not effective_enabled:
        return

    ctx = _create_extraction_context(
        content=content,
        file_id=file_id,
        chunk_id=chunk_id,
    )
    if ctx is None:
        return

    _run_registered_extractors(
        ctx=ctx,
        out=out,
        enabled_entity_types=effective_enabled,
    )


def _candidate_passes_filter(candidate: _Candidate) -> bool:
    policy = _ENTITY_POLICIES.get(candidate.term_type, _ENTITY_POLICIES['acronym'])
    has_observed_alias = len(candidate.aliases) >= 1
    if policy.require_observed_alias and not has_observed_alias:
        return False
    return candidate.confidence >= policy.min_confidence


def _create_extraction_context(
    *,
    content: str,
    file_id: int | None,
    chunk_id: int | None,
) -> _ExtractionContext | None:
    raw_text = str(content or '')
    if not raw_text.strip():
        return None
    return _ExtractionContext(
        raw_text=raw_text,
        processed_text=_preprocess_text_for_matching(raw_text),
        file_id=file_id,
        chunk_id=chunk_id,
    )


def _run_registered_extractors(
    *,
    ctx: _ExtractionContext,
    out: dict[tuple[str, str], _Candidate],
    enabled_entity_types: set[str],
) -> None:
    for entity_type in sorted(enabled_entity_types):
        extractor = _EXTRACTOR_REGISTRY.get(entity_type)
        if extractor is None:
            continue
        extractor(ctx, out)


def _collect_candidates_from_rows(
    *,
    rows: list[aiosqlite.Row],
    candidates: dict[tuple[str, str], _Candidate],
    enabled_entity_types: set[str],
) -> int:
    processed_chunks = 0
    for row in rows:
        ctx = _create_extraction_context(
            content=row['content'],
            file_id=row['file_id'],
            chunk_id=row['chunk_id'],
        )
        if ctx is None:
            continue
        processed_chunks += 1
        _run_registered_extractors(
            ctx=ctx,
            out=candidates,
            enabled_entity_types=enabled_entity_types,
        )
    return processed_chunks


def _filter_candidates(candidates: dict[tuple[str, str], _Candidate]) -> list[_Candidate]:
    return [candidate for candidate in candidates.values() if _candidate_passes_filter(candidate)]


async def _persist_filtered_candidates(
    *,
    db: aiosqlite.Connection,
    target_version: int,
    filtered_candidates: list[_Candidate],
) -> tuple[int, int]:
    terms_inserted = 0
    aliases_inserted = 0
    await delete_term_dictionary_version(db, dict_version=target_version)
    for candidate in filtered_candidates:
        term_id = await insert_term_entry(
            db,
            canonical_term=candidate.canonical,
            normalized_term=candidate.normalized_canonical,
            term_type=candidate.term_type,
            confidence=candidate.confidence,
            status='active',
            dict_version=target_version,
        )
        terms_inserted += 1

        for normalized_alias, (alias, alias_confidence) in candidate.aliases.items():
            await insert_term_alias(
                db,
                term_id=term_id,
                alias=alias,
                normalized_alias=normalized_alias,
                alias_type='observed',
                confidence=alias_confidence,
            )
            aliases_inserted += 1

        if candidate.normalized_canonical not in candidate.aliases:
            await insert_term_alias(
                db,
                term_id=term_id,
                alias=candidate.canonical,
                normalized_alias=candidate.normalized_canonical,
                alias_type='canonical',
                confidence=candidate.confidence,
            )
            aliases_inserted += 1

        for file_id, chunk_id, snippet, method in candidate.evidence:
            await insert_term_evidence(
                db,
                term_id=term_id,
                file_id=file_id,
                chunk_id=chunk_id,
                evidence_snippet=snippet,
                extraction_method=method,
            )
    return terms_inserted, aliases_inserted


async def rebuild_term_dictionary(
    db: aiosqlite.Connection,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
    if not _builder_enabled():
        return {
            'status': 'skipped',
            'reason': 'disabled',
        }

    current_version = await get_term_dictionary_current_version(db)
    target_version = current_version + 1
    run_identifier = run_id or f'term-dict-{uuid.uuid4().hex[:12]}'
    processed_chunks = 0
    aliases_inserted = 0
    terms_inserted = 0
    last_chunk_id = 0
    candidates: dict[tuple[str, str], _Candidate] = {}
    enabled_entity_types = _enabled_entity_types()

    # Keep history for diagnostics; do not overwrite previous runs.
    await start_term_dictionary_build_run(
        db,
        run_id=run_identifier,
        target_version=target_version,
    )
    log.info(
        'term_dictionary_build_started',
        run_id=run_identifier,
        target_version=target_version,
        enabled_entity_types=sorted(enabled_entity_types),
    )

    try:
        while True:
            rows = await get_term_dictionary_source_rows(
                db,
                after_chunk_id=last_chunk_id,
                limit=_batch_size(),
            )
            if not rows:
                break
            last_chunk_id = int(rows[-1]['chunk_id'])
            processed_chunks += _collect_candidates_from_rows(
                rows=rows,
                candidates=candidates,
                enabled_entity_types=enabled_entity_types,
            )
            await update_term_dictionary_build_run_progress(
                db,
                run_id=run_identifier,
                last_processed_chunk_id=last_chunk_id,
                processed_chunks=processed_chunks,
            )

        filtered_candidates = _filter_candidates(candidates)
        quality_gate = evaluate_term_dictionary_quality(
            total_candidates=len(candidates),
            kept_candidates=len(filtered_candidates),
            noise_rate_threshold=float(settings.term_dictionary_quality_noise_rate_threshold),
            min_candidates_for_gate=int(settings.term_dictionary_quality_min_candidates_for_gate),
            gate_enabled=bool(settings.term_dictionary_quality_gate_enabled),
            candidate_term_types=[candidate.term_type for candidate in candidates.values()],
            kept_term_types=[candidate.term_type for candidate in filtered_candidates],
        )
        if not quality_gate.passed:
            raise RuntimeError(f'term_dictionary_quality_gate_failed:{quality_gate.reason}')

        terms_inserted, aliases_inserted = await _persist_filtered_candidates(
            db=db,
            target_version=target_version,
            filtered_candidates=filtered_candidates,
        )

        await set_term_dictionary_current_version(db, target_version)
        await finalize_term_dictionary_build_run(
            db,
            run_id=run_identifier,
            status='completed',
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
        )
        log.info(
            'term_dictionary_build_completed',
            run_id=run_identifier,
            target_version=target_version,
            processed_chunks=processed_chunks,
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
            quality_metrics={
                'noise_rate': quality_gate.metrics.noise_rate,
                'keep_rate': quality_gate.metrics.keep_rate,
                'candidate_type_counts': quality_gate.metrics.candidate_type_counts,
                'kept_type_counts': quality_gate.metrics.kept_type_counts,
            },
        )
        return {
            'status': 'completed',
            'run_id': run_identifier,
            'target_version': target_version,
            'processed_chunks': processed_chunks,
            'terms_inserted': terms_inserted,
            'aliases_inserted': aliases_inserted,
            'quality_metrics': {
                'noise_rate': quality_gate.metrics.noise_rate,
                'keep_rate': quality_gate.metrics.keep_rate,
                'candidate_type_counts': quality_gate.metrics.candidate_type_counts,
                'kept_type_counts': quality_gate.metrics.kept_type_counts,
            },
        }
    except _TERM_DICTIONARY_BUILD_GUARD_EXCEPTIONS as exc:
        await finalize_term_dictionary_build_run(
            db,
            run_id=run_identifier,
            status='failed',
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
            error_summary=str(exc),
        )
        log.warning('term_dictionary_build_failed', run_id=run_identifier, error=str(exc))
        return {
            'status': 'failed',
            'run_id': run_identifier,
            'target_version': target_version,
            'processed_chunks': processed_chunks,
            'terms_inserted': terms_inserted,
            'aliases_inserted': aliases_inserted,
            'error': str(exc),
        }


async def get_term_dictionary_build_status(db: aiosqlite.Connection) -> dict[str, object]:
    latest = await get_latest_term_dictionary_build_run(db)
    current_version = await get_term_dictionary_current_version(db)
    return {
        'enabled': _builder_enabled(),
        'current_version': current_version,
        'latest_run': latest,
    }
