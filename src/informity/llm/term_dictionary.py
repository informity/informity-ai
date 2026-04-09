# ==============================================================================
# Informity AI — Term Dictionary Runtime Expansion
# Deterministic query-time term matching and bounded expansion.
# ==============================================================================

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import aiosqlite

from informity.config import settings
from informity.db.sqlite import get_active_term_alias_rows, get_term_dictionary_current_version

_NON_WORD_PATTERN = re.compile(r'[^a-z0-9\s]')
_WS_PATTERN = re.compile(r'\s+')

_DEFAULT_MAX_EMBED_EXPANSIONS = 6
_DEFAULT_MAX_FTS_EXPANSIONS = 10
_DEFAULT_MAX_FUZZY_EXPANSIONS = 2
_DEFAULT_MAX_FUZZY_PER_CANONICAL = 1
_PERSON_SCOPE_HINTS: tuple[str, ...] = (
    'who',
    'person',
    'people',
    'name',
    'names',
)
_PERSON_INTENT_HINTS: tuple[str, ...] = (
    'mention',
    'mentions',
    'mentioned',
    'list',
    'identify',
    'identified',
)


def _clamp_int(value: int, *, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def term_dictionary_enabled() -> bool:
    return bool(settings.term_dictionary_enabled)


def term_dictionary_routing_enabled() -> bool:
    return bool(settings.term_dictionary_routing_enabled)


def _high_confidence_threshold() -> float:
    try:
        return max(0.0, min(1.0, float(settings.term_dictionary_high_confidence)))
    except (TypeError, ValueError):
        return 0.85


def _medium_confidence_threshold() -> float:
    try:
        return max(0.0, min(1.0, float(settings.term_dictionary_medium_confidence)))
    except (TypeError, ValueError):
        return 0.65


def normalize_term_text(text: str) -> str:
    lowered = str(text or '').strip().casefold()
    lowered = _NON_WORD_PATTERN.sub(' ', lowered)
    lowered = _WS_PATTERN.sub(' ', lowered)
    return lowered.strip()


def _tokenize_normalized(text: str) -> list[str]:
    normalized = normalize_term_text(text)
    return [part for part in normalized.split(' ') if part]


def _bounded_ocr_normalize_token(token: str) -> str:
    # Conservative OCR normalization for longer alnum tokens.
    if len(token) < 5:
        return token
    if not any(ch.isdigit() for ch in token):
        return token
    return token.replace('0', 'o').replace('1', 'l')


def _edit_distance(a: str, b: str, max_distance: int = 1) -> int:
    # Bounded Levenshtein distance with early exit.
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if not a or not b:
        return max(len(a), len(b))

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        min_row = curr[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr.append(value)
            if value < min_row:
                min_row = value
        if min_row > max_distance:
            return max_distance + 1
        prev = curr
    return prev[-1]


def _confidence_tier(confidence: float) -> str:
    high = _high_confidence_threshold()
    medium = _medium_confidence_threshold()
    if confidence >= high:
        return 'high'
    if confidence >= medium:
        return 'medium'
    return 'low'


def _allow_person_name_expansion_for_query(query: str) -> bool:
    normalized = normalize_term_text(query)
    if not normalized:
        return False
    tokens = set(normalized.split())
    has_scope_hint = any(hint in tokens for hint in _PERSON_SCOPE_HINTS)
    has_intent_hint = any(hint in tokens for hint in _PERSON_INTENT_HINTS)
    return has_scope_hint and has_intent_hint


@dataclass(slots=True)
class TermMatch:
    alias: str
    canonical: str
    match_type: str
    tier: str


@dataclass(slots=True)
class TermExpansion:
    dictionary_version: int = 0
    embedding_query: str = ''
    fts_query: str = ''
    matches: list[TermMatch] = field(default_factory=list)
    embedding_terms: list[str] = field(default_factory=list)
    fts_terms: list[str] = field(default_factory=list)
    fuzzy_cap_reached: bool = False


@dataclass(slots=True)
class RoutingExpansion:
    dictionary_version: int = 0
    expanded_query: str = ''
    canonical_terms: list[str] = field(default_factory=list)


async def expand_query_for_retrieval(
    *,
    db: aiosqlite.Connection | None,
    query: str,
    allow_person_name_expansion: bool | None = None,
) -> TermExpansion:
    raw_query = str(query or '').strip()
    default = TermExpansion(
        dictionary_version=0,
        embedding_query=raw_query,
        fts_query=raw_query,
    )
    if not raw_query or db is None or not isinstance(db, aiosqlite.Connection):
        return default
    if not term_dictionary_enabled():
        return default

    try:
        dictionary_version = await get_term_dictionary_current_version(db)
        if dictionary_version <= 0:
            return default
        alias_rows = await get_active_term_alias_rows(db)
    except (aiosqlite.Error, sqlite3.Error, TypeError, ValueError):
        return default
    if not alias_rows:
        return default

    normalized_query = normalize_term_text(raw_query)
    query_tokens = _tokenize_normalized(normalized_query)
    if not query_tokens:
        return default
    person_name_expansion_allowed = (
        bool(allow_person_name_expansion)
        if allow_person_name_expansion is not None
        else _allow_person_name_expansion_for_query(raw_query)
    )

    matched_aliases: set[str] = set()
    embedding_terms: list[str] = []
    fts_terms: list[str] = []
    matches: list[TermMatch] = []

    max_embed = _clamp_int(
        settings.term_dictionary_max_embed_expansions,
        minimum=0,
        maximum=32,
        fallback=_DEFAULT_MAX_EMBED_EXPANSIONS,
    )
    max_fts = _clamp_int(
        settings.term_dictionary_max_fts_expansions,
        minimum=0,
        maximum=64,
        fallback=_DEFAULT_MAX_FTS_EXPANSIONS,
    )
    max_fuzzy = _clamp_int(
        settings.term_dictionary_max_fuzzy_expansions,
        minimum=0,
        maximum=8,
        fallback=_DEFAULT_MAX_FUZZY_EXPANSIONS,
    )
    max_fuzzy_per_canonical = _clamp_int(
        settings.term_dictionary_max_fuzzy_per_canonical,
        minimum=0,
        maximum=4,
        fallback=_DEFAULT_MAX_FUZZY_PER_CANONICAL,
    )
    fuzzy_count = 0
    fuzzy_per_canonical: dict[str, int] = {}
    fuzzy_cap_reached = False

    # 1) Phrase matching first (longest alias first from SQL ORDER BY).
    for row in alias_rows:
        alias_norm = str(row.get('normalized_alias') or '').strip()
        if not alias_norm or ' ' not in alias_norm:
            continue
        if str(row.get('term_type') or '').strip() == 'person_name' and not person_name_expansion_allowed:
            continue
        if alias_norm in matched_aliases:
            continue
        if f' {alias_norm} ' not in f' {normalized_query} ':
            continue

        canonical = str(row.get('canonical_term') or '').strip()
        if not canonical:
            continue
        tier = _confidence_tier(float(row.get('term_confidence') or 0.0))
        if tier == 'low':
            continue
        matched_aliases.add(alias_norm)
        matches.append(TermMatch(alias=alias_norm, canonical=canonical, match_type='phrase', tier=tier))
        if tier == 'high' and len(embedding_terms) < max_embed and canonical not in embedding_terms:
            embedding_terms.append(canonical)
        if len(fts_terms) < max_fts and canonical not in fts_terms:
            fts_terms.append(canonical)

    # 2) Exact token matching.
    query_token_set = set(query_tokens)
    for row in alias_rows:
        alias_norm = str(row.get('normalized_alias') or '').strip()
        if not alias_norm or ' ' in alias_norm:
            continue
        if str(row.get('term_type') or '').strip() == 'person_name' and not person_name_expansion_allowed:
            continue
        if alias_norm in matched_aliases:
            continue
        if alias_norm not in query_token_set:
            continue
        canonical = str(row.get('canonical_term') or '').strip()
        if not canonical:
            continue
        tier = _confidence_tier(float(row.get('term_confidence') or 0.0))
        if tier == 'low':
            continue
        matched_aliases.add(alias_norm)
        matches.append(TermMatch(alias=alias_norm, canonical=canonical, match_type='exact', tier=tier))
        if tier == 'high' and len(embedding_terms) < max_embed and canonical not in embedding_terms:
            embedding_terms.append(canonical)
        if len(fts_terms) < max_fts and canonical not in fts_terms:
            fts_terms.append(canonical)

    # 3) Guarded fuzzy matching (high confidence only).
    if max_fuzzy > 0:
        ocr_tokens = {_bounded_ocr_normalize_token(token) for token in query_tokens}
        for row in alias_rows:
            alias_norm = str(row.get('normalized_alias') or '').strip()
            if not alias_norm or ' ' in alias_norm or alias_norm in matched_aliases:
                continue
            if str(row.get('term_type') or '').strip() == 'person_name' and not person_name_expansion_allowed:
                continue
            if _confidence_tier(float(row.get('term_confidence') or 0.0)) != 'high':
                continue
            # Avoid fuzzy on very short aliases (acronym collision risk).
            if len(alias_norm) < 5:
                continue
            canonical = str(row.get('canonical_term') or '').strip()
            if not canonical:
                continue
            if fuzzy_count >= max_fuzzy:
                fuzzy_cap_reached = True
                break
            canonical_key = normalize_term_text(canonical)
            if fuzzy_per_canonical.get(canonical_key, 0) >= max_fuzzy_per_canonical:
                continue

            found = False
            for token in ocr_tokens:
                if len(token) < 5:
                    continue
                if _edit_distance(token, alias_norm, max_distance=1) <= 1:
                    found = True
                    break
            if not found:
                continue

            matched_aliases.add(alias_norm)
            fuzzy_count += 1
            fuzzy_per_canonical[canonical_key] = fuzzy_per_canonical.get(canonical_key, 0) + 1
            matches.append(TermMatch(alias=alias_norm, canonical=canonical, match_type='fuzzy', tier='high'))
            if len(embedding_terms) < max_embed and canonical not in embedding_terms:
                embedding_terms.append(canonical)
            if len(fts_terms) < max_fts and canonical not in fts_terms:
                fts_terms.append(canonical)

    embedding_query = raw_query
    if embedding_terms:
        embedding_query = f'{raw_query} {" ".join(embedding_terms)}'.strip()
    fts_query = raw_query
    if fts_terms:
        fts_query = f'{raw_query} {" ".join(fts_terms)}'.strip()

    return TermExpansion(
        dictionary_version=dictionary_version,
        embedding_query=embedding_query,
        fts_query=fts_query,
        matches=matches,
        embedding_terms=embedding_terms,
        fts_terms=fts_terms,
        fuzzy_cap_reached=fuzzy_cap_reached,
    )


def _get_active_term_alias_rows_sync() -> tuple[int, list[dict]]:
    db_path = str(settings.db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        current_row = conn.execute(
            'SELECT current_version FROM term_dictionary_state WHERE singleton_id = 1'
        ).fetchone()
        version = int(current_row['current_version']) if current_row and current_row['current_version'] is not None else 0
        if version <= 0:
            return 0, []
        rows = conn.execute(
            '''
            SELECT
                ta.normalized_alias,
                te.canonical_term,
                te.type AS term_type,
                te.confidence AS term_confidence
            FROM term_aliases ta
            JOIN term_entries te ON te.term_id = ta.term_id
            WHERE te.dict_version = ?
              AND te.status = 'active'
            ORDER BY LENGTH(ta.normalized_alias) DESC, ta.normalized_alias ASC
            ''',
            (version,),
        ).fetchall()
    finally:
        conn.close()
    return version, [
        {
            'normalized_alias': str(row['normalized_alias'] or '').strip(),
            'canonical_term': str(row['canonical_term'] or '').strip(),
            'term_type': str(row['term_type'] or '').strip(),
            'term_confidence': float(row['term_confidence'] or 0.0),
        }
        for row in rows
    ]


def expand_query_for_routing(query: str) -> RoutingExpansion:
    raw_query = str(query or '').strip()
    result = RoutingExpansion(dictionary_version=0, expanded_query=raw_query)
    if not raw_query or not term_dictionary_enabled() or not term_dictionary_routing_enabled():
        return result

    try:
        version, rows = _get_active_term_alias_rows_sync()
    except (sqlite3.Error, TypeError, ValueError):
        return result
    if version <= 0 or not rows:
        return result

    normalized_query = normalize_term_text(raw_query)
    if not normalized_query:
        return result
    query_tokens = set(_tokenize_normalized(normalized_query))
    person_name_expansion_allowed = _allow_person_name_expansion_for_query(raw_query)
    canonical_terms: list[str] = []
    max_terms = _clamp_int(
        settings.term_dictionary_max_routing_expansions,
        minimum=0,
        maximum=16,
        fallback=4,
    )
    for row in rows:
        alias_norm = row['normalized_alias']
        if not alias_norm:
            continue
        if str(row.get('term_type') or '').strip() == 'person_name' and not person_name_expansion_allowed:
            continue
        if _confidence_tier(float(row['term_confidence'] or 0.0)) != 'high':
            continue
        canonical = row['canonical_term']
        if not canonical or canonical in canonical_terms:
            continue
        matched = False
        if ' ' in alias_norm:
            matched = f' {alias_norm} ' in f' {normalized_query} '
        else:
            matched = alias_norm in query_tokens
        if not matched:
            continue
        canonical_terms.append(canonical)
        if len(canonical_terms) >= max_terms:
            break

    if not canonical_terms:
        return RoutingExpansion(dictionary_version=version, expanded_query=raw_query, canonical_terms=[])

    expanded_query = f'{raw_query} {" ".join(canonical_terms)}'.strip()
    return RoutingExpansion(
        dictionary_version=version,
        expanded_query=expanded_query,
        canonical_terms=canonical_terms,
    )
