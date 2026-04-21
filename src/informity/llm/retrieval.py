# ==============================================================================
# Informity AI — Retrieval (v2)
# One retrieval path: embed → vector search with WHERE → rerank → top-k
# ==============================================================================

import asyncio
import re
import time

import aiosqlite
import structlog

from informity.config import settings
from informity.db.sqlite import get_chunks_by_parent_ids
from informity.db.vectors import vector_store
from informity.indexer.embedder import embedder
from informity.indexer.reranker import reranker
from informity.llm.metadata_filters import (
    MetadataFilter,
    build_where_clause_and_params,
)
from informity.llm.model_adapter import get_profile
from informity.llm.term_dictionary import TermExpansion, expand_query_for_retrieval
from informity.llm.types import BlockType, FilterOperator, QueryType
from informity.upload_policy import UPLOAD_ENTITY_TYPE, UPLOAD_PROVIDER
from informity.utils.file_utils import normalize_extension

log = structlog.get_logger(__name__)
_COVERAGE_DIVERSITY_PRIMARY_FILE_CAP = 1
_COVERAGE_DIVERSITY_SECONDARY_FILE_CAP = 2
_STRUCTURAL_SECTION_PATTERNS = (
    re.compile(r'\b(table\s+of\s+contents?|contents?)\b', re.IGNORECASE),
    re.compile(r'\b(index|appendix|appendices|glossary)\b', re.IGNORECASE),
    re.compile(r'\b(references?|bibliography|citations?)\b', re.IGNORECASE),
    re.compile(r'\b(copyright|license|legal(?:\s+notice)?)\b', re.IGNORECASE),
    re.compile(r'\b(acknowledg(?:e)?ments?)\b', re.IGNORECASE),
    re.compile(r'\b(changelog|revision\s+history|release\s+notes?)\b', re.IGNORECASE),
    re.compile(r'\b(title\s+page|cover\s+page|front\s+matter)\b', re.IGNORECASE),
)
_SECTION_STRUCTURAL_PENALTY = 0.15
_BLOCK_TYPE_STRUCTURAL_PENALTY = 0.08
_TEXT_STRUCTURAL_PENALTY = 0.35
_SHORT_TEXT_PENALTY = 0.04
_SHORT_TEXT_CHAR_THRESHOLD = 180
_SHORT_TEXT_LINE_THRESHOLD = 3
_SUBSTANTIVE_FILTER_MIN_CANDIDATES = 8
_TITLE_ALIGNMENT_BONUS_PER_MATCH = 0.05
_TITLE_ALIGNMENT_BONUS_MAX = 0.20
_TITLE_ALIGNMENT_STRICT_NO_MATCH_PENALTY = 0.30
_TITLE_ALIGNMENT_STRICT_BONUS_PER_MATCH = 0.12
_TITLE_ALIGNMENT_STRICT_BONUS_MAX = 0.50
_TITLE_ALIGNMENT_STOPWORDS = {
    'a', 'an', 'and', 'as', 'at', 'attachment', 'by', 'compare', 'description', 'describe', 'document',
    'entry', 'file', 'for', 'from', 'give', 'in', 'is', 'it', 'item', 'later', 'material', 'note', 'of',
    'on', 'or', 'paper', 'record', 'source', 'text', 'the', 'to', 'vs', 'versus', 'what', 'with',
}
_STRUCTURAL_TEXT_PATTERNS = (
    re.compile(r'\*\*\*\s*start\s+of\s+the\s+project\s+gutenberg\s+ebook', re.IGNORECASE),
    re.compile(r'\bproject\s+gutenberg\s+ebook\b', re.IGNORECASE),
    re.compile(r'\bthis\s+ebook\s+is\s+for\s+the\s+use\s+of\s+anyone\b', re.IGNORECASE),
    re.compile(r'\bother\s+information\s+and\s+formats?\b', re.IGNORECASE),
    re.compile(r'\bcredits?:\b', re.IGNORECASE),
    re.compile(r'\blanguage:\s*[A-Za-z]', re.IGNORECASE),
)


def _coerce_reranker_score(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_within_file_location_key(chunk: dict) -> str:
    section_path = str(chunk.get('section_path') or '').strip().casefold()
    if section_path:
        return f'section:{section_path}'
    start_page = chunk.get('start_page')
    end_page = chunk.get('end_page')
    if isinstance(start_page, int) and isinstance(end_page, int):
        return f'pages:{start_page}-{end_page}'
    page_number = chunk.get('page_number')
    if isinstance(page_number, int):
        return f'page:{page_number}'
    return f"chunk:{chunk.get('chunk_id')}"


def _looks_structural_section(section_path: object) -> bool:
    value = str(section_path or '').strip()
    if not value:
        return False
    return any(pattern.search(value) for pattern in _STRUCTURAL_SECTION_PATTERNS)


def _apply_substantive_section_bias(
    *,
    chunks: list[dict],
    prefer_substantive_sections: bool,
) -> list[dict]:
    if not prefer_substantive_sections or len(chunks) <= 1:
        return chunks

    rescored: list[dict] = []
    for chunk in chunks:
        score = _coerce_reranker_score(chunk.get('score')) or 0.0
        penalty = 0.0

        block_type = str(chunk.get('block_type') or '').strip().casefold()
        if block_type in {'table', 'form'}:
            penalty += _BLOCK_TYPE_STRUCTURAL_PENALTY

        if _looks_structural_section(chunk.get('section_path')):
            penalty += _SECTION_STRUCTURAL_PENALTY

        text = str(chunk.get('chunk_text') or '').strip()
        if text and any(pattern.search(text[:1200]) for pattern in _STRUCTURAL_TEXT_PATTERNS):
            penalty += _TEXT_STRUCTURAL_PENALTY
        if text and len(re.findall(r'\bchapter\s+[ivxlcdm0-9]+\b', text[:1200], re.IGNORECASE)) >= 4:
            penalty += _TEXT_STRUCTURAL_PENALTY
        if text:
            line_count = text.count('\n') + 1
            if len(text) < _SHORT_TEXT_CHAR_THRESHOLD and line_count <= _SHORT_TEXT_LINE_THRESHOLD:
                penalty += _SHORT_TEXT_PENALTY

        rescored.append({**chunk, 'score': score - penalty})

    rescored.sort(key=lambda item: _coerce_reranker_score(item.get('score')) or 0.0, reverse=True)
    return rescored


def _is_structural_text_snippet(text: str) -> bool:
    snippet = str(text or '')[:1200]
    if not snippet:
        return False
    if any(pattern.search(snippet) for pattern in _STRUCTURAL_TEXT_PATTERNS):
        return True
    if len(re.findall(r'\bchapter\s+[ivxlcdm0-9]+\b', snippet, re.IGNORECASE)) >= 4:
        return True
    return False


def _filter_structural_chunks_when_possible(
    *,
    chunks: list[dict],
    prefer_substantive_sections: bool,
    top_k: int,
) -> list[dict]:
    if not prefer_substantive_sections or len(chunks) <= 1:
        return chunks
    non_structural = [
        chunk for chunk in chunks
        if not _is_structural_text_snippet(str(chunk.get('chunk_text') or ''))
    ]
    minimum_keep = max(top_k, _SUBSTANTIVE_FILTER_MIN_CANDIDATES)
    if len(non_structural) >= minimum_keep:
        return non_structural
    return chunks


def _tokenize_title_alignment_terms(text: str) -> set[str]:
    lowered = str(text or '').strip().lower()
    if not lowered:
        return set()
    raw_terms = set(re.findall(r"[a-z0-9][a-z0-9'_-]{2,}", lowered))
    terms: set[str] = set()
    for term in raw_terms:
        if term in _TITLE_ALIGNMENT_STOPWORDS:
            continue
        terms.add(term)
        if term.endswith('s') and len(term) > 4:
            terms.add(term[:-1])
    return terms


def _apply_title_alignment_bias(
    *,
    chunks: list[dict],
    query: str | None,
    prefer_title_alignment: bool,
    strict_title_alignment: bool = False,
) -> list[dict]:
    if not prefer_title_alignment or len(chunks) <= 1:
        return chunks
    query_terms = _tokenize_title_alignment_terms(query or '')
    if not query_terms:
        return chunks

    chunk_overlaps: list[tuple[dict, int]] = []
    has_overlap_match = False
    for chunk in chunks:
        filename = str(chunk.get('filename') or '')
        filename_terms = _tokenize_title_alignment_terms(filename)
        overlap_count = len(query_terms & filename_terms)
        if overlap_count > 0:
            has_overlap_match = True
        chunk_overlaps.append((chunk, overlap_count))

    rescored: list[dict] = []
    for chunk, overlap_count in chunk_overlaps:
        score = _coerce_reranker_score(chunk.get('score')) or 0.0
        if overlap_count <= 0:
            penalty = _TITLE_ALIGNMENT_STRICT_NO_MATCH_PENALTY if strict_title_alignment and has_overlap_match else 0.0
            rescored.append({**chunk, 'score': score - penalty})
            continue
        if strict_title_alignment:
            bonus = min(_TITLE_ALIGNMENT_STRICT_BONUS_MAX, overlap_count * _TITLE_ALIGNMENT_STRICT_BONUS_PER_MATCH)
        else:
            bonus = min(_TITLE_ALIGNMENT_BONUS_MAX, overlap_count * _TITLE_ALIGNMENT_BONUS_PER_MATCH)
        rescored.append({**chunk, 'score': score + bonus})

    rescored.sort(key=lambda item: _coerce_reranker_score(item.get('score')) or 0.0, reverse=True)
    return rescored


def _apply_strict_title_file_focus(
    *,
    chunks: list[dict],
    query: str | None,
    strict_title_alignment: bool,
) -> list[dict]:
    if not strict_title_alignment or len(chunks) <= 1:
        return chunks
    query_terms = _tokenize_title_alignment_terms(query or '')
    if not query_terms:
        return chunks

    overlap_by_file: dict[int, int] = {}
    for chunk in chunks:
        file_id = chunk.get('file_id')
        try:
            normalized_file_id = int(file_id)
        except (TypeError, ValueError):
            continue
        filename_terms = _tokenize_title_alignment_terms(str(chunk.get('filename') or ''))
        overlap = len(query_terms & filename_terms)
        if overlap <= 0:
            continue
        overlap_by_file[normalized_file_id] = max(overlap_by_file.get(normalized_file_id, 0), overlap)

    if not overlap_by_file:
        return chunks
    best_overlap = max(overlap_by_file.values())
    focused_file_ids = {file_id for file_id, overlap in overlap_by_file.items() if overlap == best_overlap}
    focused_chunks: list[dict] = []
    for chunk in chunks:
        try:
            normalized_file_id = int(chunk.get('file_id'))
        except (TypeError, ValueError):
            continue
        if normalized_file_id in focused_file_ids:
            focused_chunks.append(chunk)
    return focused_chunks or chunks


def _select_top_children(
    *,
    reranked_children: list[dict],
    top_k: int,
    query_type: QueryType,
    prefer_within_file_diversity: bool = False,
) -> list[dict]:
    if top_k <= 0:
        return []
    if query_type != QueryType.COVERAGE:
        if not prefer_within_file_diversity:
            return reranked_children[:top_k]
        selected: list[dict] = []
        seen_chunk_ids: set[int] = set()
        seen_file_locations: set[tuple[int, str]] = set()
        for chunk in reranked_children:
            try:
                chunk_id = int(chunk.get('chunk_id'))
            except (TypeError, ValueError):
                continue
            if chunk_id in seen_chunk_ids:
                continue
            file_id_raw = chunk.get('file_id')
            try:
                file_id = int(file_id_raw) if file_id_raw is not None else -1
            except (TypeError, ValueError):
                file_id = -1
            file_location = (file_id, _resolve_within_file_location_key(chunk))
            if file_location in seen_file_locations:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk_id)
            seen_file_locations.add(file_location)
            if len(selected) >= top_k:
                return selected
        if len(selected) < top_k:
            for chunk in reranked_children:
                try:
                    chunk_id = int(chunk.get('chunk_id'))
                except (TypeError, ValueError):
                    continue
                if chunk_id in seen_chunk_ids:
                    continue
                selected.append(chunk)
                seen_chunk_ids.add(chunk_id)
                if len(selected) >= top_k:
                    break
        return selected

    selected: list[dict] = []
    seen_chunk_ids: set[int] = set()
    file_counts: dict[int, int] = {}
    seen_file_locations: set[tuple[int, str]] = set()

    def _try_add(
        chunk: dict,
        *,
        per_file_cap: int | None,
        enforce_within_file_diversity: bool,
    ) -> None:
        if len(selected) >= top_k:
            return
        try:
            chunk_id = int(chunk.get('chunk_id'))
        except (TypeError, ValueError):
            return
        if chunk_id in seen_chunk_ids:
            return

        file_id_raw = chunk.get('file_id')
        file_id: int | None = None
        try:
            if file_id_raw is not None:
                file_id = int(file_id_raw)
        except (TypeError, ValueError):
            file_id = None

        if (
            per_file_cap is not None
            and file_id is not None
            and file_counts.get(file_id, 0) >= per_file_cap
        ):
            return
        if enforce_within_file_diversity and file_id is not None:
            location_key = _resolve_within_file_location_key(chunk)
            file_location = (file_id, location_key)
            if file_location in seen_file_locations:
                return

        selected.append(chunk)
        seen_chunk_ids.add(chunk_id)
        if file_id is not None:
            file_counts[file_id] = file_counts.get(file_id, 0) + 1
            if enforce_within_file_diversity:
                seen_file_locations.add((file_id, _resolve_within_file_location_key(chunk)))

    for per_file_cap, enforce_within_file_diversity in (
        (_COVERAGE_DIVERSITY_PRIMARY_FILE_CAP, True),
        (_COVERAGE_DIVERSITY_SECONDARY_FILE_CAP, True),
        (_COVERAGE_DIVERSITY_SECONDARY_FILE_CAP, False),
        (None, False),
    ):
        for chunk in reranked_children:
            _try_add(
                chunk,
                per_file_cap=per_file_cap,
                enforce_within_file_diversity=enforce_within_file_diversity,
            )
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    return selected[:top_k]


async def retrieve_chunks(
    query: str,
    top_k: int,
    max_score: float | None = None,
    year_filter: int | None = None,
    category_filter: str | None = None,
    extension_filter: str | None = None,
    filename_filter: str | None = None,
    filename_exclude: list[str] | None = None,
    block_type_filter: str | None = None,
    block_type_exclude: list[str] | None = None,
    section_filter: str | None = None,
    file_ids_filter: list[int] | None = None,
    exclude_upload_sources: bool = False,
    prefer_substantive_sections: bool = False,
    prefer_title_alignment: bool = False,
    title_alignment_query: str | None = None,
    strict_title_alignment: bool = False,
    enable_term_expansion: bool = True,
    prefer_within_file_diversity: bool = False,
    query_type: QueryType = QueryType.FOCUSED,
    db: aiosqlite.Connection | None = None,
    trace: object | None = None,
    timing_output: dict | None = None,
) -> list[dict]:
    """
    Unified retrieval path: embed query → vector search (optional WHERE) → rerank → top-k.

    If trace is provided (TraceWriter protocol), records 'retrieval' and 'rerank' steps.
    """
    # 1. Embed query (CPU-bound, run in thread pool to avoid blocking event loop)
    if enable_term_expansion:
        term_expansion = await expand_query_for_retrieval(db=db, query=query)
    else:
        term_expansion = TermExpansion(
            dictionary_version=0,
            embedding_query=query,
            fts_query=query,
        )
    query_for_embedding = term_expansion.embedding_query or query
    query_for_fts = term_expansion.fts_query or query

    if term_expansion.embedding_terms:
        log.debug(
            'term_dictionary_query_expanded',
            dictionary_version=term_expansion.dictionary_version,
            embedding_terms_count=len(term_expansion.embedding_terms),
            fts_terms_count=len(term_expansion.fts_terms),
            matches_count=len(term_expansion.matches),
            fuzzy_cap_reached=term_expansion.fuzzy_cap_reached,
        )

    embed_start = time.perf_counter()
    query_vector = await asyncio.to_thread(embedder.embed_query, query_for_embedding)
    embed_elapsed_ms = (time.perf_counter() - embed_start) * 1000

    start = time.perf_counter()

    # 2. Build WHERE clause from filters using unified filter system
    filters: list[MetadataFilter] = []
    if year_filter:
        filters.append(MetadataFilter(field='year', operator=FilterOperator.EQ, value=year_filter))
    if category_filter:
        # Sanitize category filter (only alphanumeric, underscore, hyphen)
        safe_category = ''.join(c for c in category_filter if c.isalnum() or c in '_-')
        if safe_category:
            filters.append(MetadataFilter(field='category', operator=FilterOperator.EQ, value=safe_category))
    if extension_filter:
        # Sanitize extension filter (ensure it starts with dot)
        safe_extension = normalize_extension(extension_filter)
        filters.append(MetadataFilter(field='extension', operator=FilterOperator.EQ, value=safe_extension))
    if filename_filter:
        normalized_filename_filter = filename_filter.strip()
        if normalized_filename_filter:
            filters.append(
                MetadataFilter(
                    field='filename',
                    operator=FilterOperator.LIKE,
                    value=f'%{normalized_filename_filter}%',
                )
            )
    if filename_exclude:
        for excluded_name in filename_exclude:
            normalized_excluded_name = str(excluded_name or '').strip()
            if not normalized_excluded_name:
                continue
            filters.append(
                MetadataFilter(
                    field='filename',
                    operator=FilterOperator.NE,
                    value=normalized_excluded_name,
                )
            )
    if file_ids_filter:
        normalized_file_ids = sorted({int(file_id) for file_id in file_ids_filter if int(file_id) > 0})
        if normalized_file_ids:
            filters.append(
                MetadataFilter(
                    field='file_id',
                    operator=FilterOperator.IN,
                    value=normalized_file_ids,
                )
            )
    safe_block_type_filter = (
        block_type_filter
        if block_type_filter in {BlockType.TABLE, BlockType.FORM, BlockType.NARRATIVE}
        else None
    )
    if safe_block_type_filter:
        filters.append(
            MetadataFilter(
                field='block_type',
                operator=FilterOperator.EQ,
                value=safe_block_type_filter,
            )
        )
    safe_block_type_exclude = [
        block_type
        for block_type in (block_type_exclude or [])
        if block_type in {BlockType.TABLE, BlockType.FORM, BlockType.NARRATIVE}
    ]
    for excluded_block_type in safe_block_type_exclude:
        filters.append(
            MetadataFilter(
                field='block_type',
                operator=FilterOperator.NE,
                value=excluded_block_type,
            )
        )

    active_filters = list(filters)
    # Vector search operates on vec_chunks columns only. Structural chunk-level
    # filters (for example block_type) are enforced after fetching chunks from
    # the chunks table, where those fields are authoritative.
    vector_filters = [f for f in active_filters if f.field != 'block_type']
    where_clause, where_params = build_where_clause_and_params(vector_filters)
    if exclude_upload_sources:
        upload_exclusion_clause = 'file_id NOT IN (SELECT id FROM files WHERE source_provider = ? AND entity_type = ?)'
        upload_exclusion_params: list[int | str] = [UPLOAD_PROVIDER, UPLOAD_ENTITY_TYPE]
        if where_clause:
            where_clause = f'({where_clause}) AND {upload_exclusion_clause}'
            where_params = [*where_params, *upload_exclusion_params]
        else:
            where_clause = upload_exclusion_clause
            where_params = upload_exclusion_params
    applied_filters_for_trace = [
        {
            'field': metadata_filter.field,
            'operator': metadata_filter.operator,
            'value': metadata_filter.value,
        }
        for metadata_filter in active_filters
    ]
    safe_section_filter = section_filter.strip().casefold() if isinstance(section_filter, str) and section_filter.strip() else None

    fts5_augmented_count: int = 0  # Net-new candidates added by FTS5 augmentation (focused only)
    profile = get_profile()
    # 3. Vector retrieval (single path, no coverage-specific fallback branch)
    search_k = max(top_k * 2, int(getattr(profile, 'retrieval_top_k_candidates', 25)))
    results = await asyncio.to_thread(
        vector_store.search_similar,
        query_vector,
        search_k,
        where_clause,
        where_params,
    )
    if max_score is not None:
        raw_before_score_filter = len(results)
        results = [r for r in results if r.get('score', float('inf')) <= max_score]
        if raw_before_score_filter > 0 and not results:
            log.info(
                'l2_threshold_eliminated_all_candidates',
                query_type=query_type,
                max_score=max_score,
                raw_candidates=raw_before_score_filter,
            )
    search_elapsed_ms = (time.perf_counter() - start) * 1000

    # FTS5 candidate augmentation — add exact-match pool candidates before reranking.
    # Candidate-only: FTS5 contributes chunk IDs to the pool, reranker remains sole scorer.
    if settings.fts5_candidate_limit > 0:
        existing_ids = {r['chunk_id'] for r in results}
        fts5_candidates = await asyncio.to_thread(
            vector_store.fts5_augment_candidates,
            query_for_fts,
            settings.fts5_candidate_limit,
            existing_ids,
            where_clause,
            where_params if where_params else None,
        )
        if fts5_candidates:
            fts5_augmented_count = len(fts5_candidates)
            results = results + fts5_candidates
            log.debug(
                'fts5_candidates_augmented',
                count=fts5_augmented_count,
                total_pool=len(results),
            )

    if not results:
        log.info(
            'retrieval_completed',
            query_type=query_type,
            mode='vector',
            query_length=len(query),
            raw_candidates=0,
            children_reranked=0,
            children_returned=0,
            parents_returned=0,
            embed_duration_ms=round(embed_elapsed_ms, 1),
            search_duration_ms=round(search_elapsed_ms, 1),
            rerank_duration_ms=0.0,
        )
        if trace is not None:
            trace.record('retrieval', {
                'mode':                'vector',
                'raw_chunks_count':    0,
                'applied_filters':     applied_filters_for_trace,
                'where_clause':        where_clause,
                'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
                'search_elapsed_ms':   round(search_elapsed_ms, 1),
                'term_dictionary_version': term_expansion.dictionary_version,
                'term_dictionary_embedding_terms': term_expansion.embedding_terms,
                'term_dictionary_fts_terms': term_expansion.fts_terms,
                'term_dictionary_matches': [
                    {
                        'alias': match.alias,
                        'canonical': match.canonical,
                        'match_type': match.match_type,
                        'tier': match.tier,
                    }
                    for match in term_expansion.matches
                ],
                'term_dictionary_fuzzy_cap_reached': term_expansion.fuzzy_cap_reached,
                'year_filter':          year_filter,
                'category_filter':     category_filter,
                'extension_filter':    extension_filter,
                'filename_filter':     filename_filter,
                'filename_exclude':    list(filename_exclude or []),
                'block_type_filter':   safe_block_type_filter,
                'block_type_exclude':  list(safe_block_type_exclude),
                'section_filter':      safe_section_filter,
                'max_score':           max_score,
                'file_ids_filter':     file_ids_filter,
                'exclude_upload_sources': exclude_upload_sources,
                'prefer_substantive_sections': prefer_substantive_sections,
                'prefer_title_alignment': prefer_title_alignment,
                'strict_title_alignment': strict_title_alignment,
                'term_expansion_enabled': enable_term_expansion,
                'prefer_within_file_diversity': prefer_within_file_diversity,
            })
        return []

    # 4. Get child chunk texts from SQLite (for reranking)
    # Also fetch parent_id for each child chunk
    if db is None:
        log.warning('retrieve_chunks_no_db', msg='No DB connection provided')
        return []

    child_chunk_ids = [r['chunk_id'] for r in results]

    # Fetch child chunks with their parent_ids
    if not child_chunk_ids:
        return []

    child_chunk_ids_unique = list(dict.fromkeys(child_chunk_ids))
    placeholders = ','.join('?' * len(child_chunk_ids_unique))
    cursor = await db.execute(
        f"""
        SELECT c.id AS chunk_id, c.file_id, f.path AS file_path, f.filename, c.content AS chunk_text,
               c.page_number, c.start_page, c.end_page, c.section_path, c.block_type, c.parent_id
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders})
        """,
        child_chunk_ids_unique,
    )
    rows = await cursor.fetchall()

    # Build mapping: chunk_id -> chunk dict (preserve order from vector search)
    chunk_id_to_dict: dict[int, dict] = {}
    child_to_parent_map: dict[int, int] = {}  # child_chunk_id -> parent_id

    for row in rows:
        chunk_dict = {
            'chunk_id':   row['chunk_id'],
            'file_id':    row['file_id'],
            'file_path':  row['file_path'] or '',
            'filename':   row['filename'] or '',
            'chunk_text': row['chunk_text'] or '',
        }
        try:
            chunk_dict['page_number'] = row['page_number']
        except (KeyError, IndexError):
            chunk_dict['page_number'] = None
        try:
            chunk_dict['start_page'] = row['start_page']
        except (KeyError, IndexError):
            chunk_dict['start_page'] = None
        try:
            chunk_dict['end_page'] = row['end_page']
        except (KeyError, IndexError):
            chunk_dict['end_page'] = None
        try:
            chunk_dict['section_path'] = row['section_path']
        except (KeyError, IndexError):
            chunk_dict['section_path'] = None
        try:
            chunk_dict['block_type'] = row['block_type']
        except (KeyError, IndexError):
            chunk_dict['block_type'] = None

        # Store parent_id mapping
        try:
            parent_id = row['parent_id']
            if parent_id:
                child_to_parent_map[row['chunk_id']] = parent_id
        except (KeyError, IndexError):
            pass

        chunk_id_to_dict[row['chunk_id']] = chunk_dict

    # Preserve order from vector search results; retain vector scores for rerank-bypass path
    vector_score_map: dict[int, float] = {r['chunk_id']: float(r['score']) for r in results if 'score' in r}
    child_chunks: list[dict] = [chunk_id_to_dict[cid] for cid in child_chunk_ids if cid in chunk_id_to_dict]

    # Apply structure-aware filters on fetched chunks (same unified retrieval path).
    filtered_child_chunks = child_chunks
    if safe_block_type_filter is not None:
        block_filtered = [chunk for chunk in filtered_child_chunks if chunk.get('block_type') == safe_block_type_filter]
        if block_filtered:
            filtered_child_chunks = block_filtered
    if safe_block_type_exclude:
        filtered_child_chunks = [
            chunk
            for chunk in filtered_child_chunks
            if chunk.get('block_type') not in safe_block_type_exclude
        ]
    if safe_section_filter is not None:
        section_filtered = [
            chunk
            for chunk in filtered_child_chunks
            if isinstance(chunk.get('section_path'), str) and safe_section_filter in chunk['section_path'].casefold()
        ]
        if section_filtered:
            filtered_child_chunks = section_filtered
    filtered_child_chunks = _filter_structural_chunks_when_possible(
        chunks=filtered_child_chunks,
        prefer_substantive_sections=prefer_substantive_sections,
        top_k=top_k,
    )

    # 5. Rerank child chunks (controlled by rag_rerank / rag_rerank_coverage settings)
    # CPU-bound cross-encoder, run in thread pool to avoid blocking event loop
    # Reranker scores are not mutated after this point — no post-rerank heuristic boosts by policy.
    is_coverage_query  = query_type == QueryType.COVERAGE
    rerank_enabled     = settings.rag_rerank and (not is_coverage_query or settings.rag_rerank_coverage)
    rerank_start       = time.perf_counter()
    pre_rerank_top_ids = [chunk.get('chunk_id') for chunk in filtered_child_chunks[:top_k]]
    if rerank_enabled:
        reranked_children = await asyncio.to_thread(reranker.rerank, query, filtered_child_chunks)
    else:
        # Annotate chunks with vector-search scores so downstream consumers always have a score field
        reranked_children = [
            {**chunk, 'score': vector_score_map.get(chunk['chunk_id'], 0.0)}
            for chunk in filtered_child_chunks
        ]
    reranked_children = _apply_substantive_section_bias(
        chunks=reranked_children,
        prefer_substantive_sections=prefer_substantive_sections,
    )
    reranked_children = _apply_title_alignment_bias(
        chunks=reranked_children,
        query=title_alignment_query or query,
        prefer_title_alignment=prefer_title_alignment,
        strict_title_alignment=strict_title_alignment,
    )
    reranked_children = _apply_strict_title_file_focus(
        chunks=reranked_children,
        query=title_alignment_query or query,
        strict_title_alignment=strict_title_alignment,
    )
    rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000
    top_children = _select_top_children(
        reranked_children=reranked_children,
        top_k=top_k,
        query_type=query_type,
        prefer_within_file_diversity=prefer_within_file_diversity,
    )
    post_rerank_top_ids = [chunk.get('chunk_id') for chunk in top_children]
    rerank_top_k_overlap = len(set(pre_rerank_top_ids) & set(post_rerank_top_ids))

    # 6. Parent Document Retrieval: fetch parent chunks for LLM context
    # Extract parent_ids from top children
    parent_ids: list[int] = []
    for child in top_children:
        child_id = child['chunk_id']
        if child_id in child_to_parent_map:
            parent_id = child_to_parent_map[child_id]
            parent_ids.append(parent_id)

    # Fetch parent chunks (deduplicated automatically by get_chunks_by_parent_ids)
    parent_chunks = await get_chunks_by_parent_ids(db, parent_ids) if parent_ids else []

    # Build mapping: parent_id -> parent chunk for quick lookup
    parent_id_to_chunk: dict[int, dict] = {p['chunk_id']: p for p in parent_chunks}

    # Return parent chunks in order of child relevance (preserve reranking order)
    final: list[dict] = []
    seen_parent_ids: set[int] = set()
    warned_child_ids: set[int] = set()  # Track chunks we've already warned about

    for child in top_children:
        child_id = child['chunk_id']
        parent_id = child_to_parent_map.get(child_id)

        if parent_id and parent_id in parent_id_to_chunk and parent_id not in seen_parent_ids:
            # Use parent chunk for context; propagate child's reranker score for trace/sources
            child_score = _coerce_reranker_score(child.get('score'))
            parent_chunk = {**parent_id_to_chunk[parent_id]}
            if child_score is not None:
                parent_chunk['score'] = child_score
            elif child_id not in warned_child_ids:
                log.warning(
                    'child_chunk_missing_score',
                    child_id=child_id,
                    parent_id=parent_id,
                )
                warned_child_ids.add(child_id)
            final.append(parent_chunk)
            seen_parent_ids.add(parent_id)
        elif not parent_id:
            # Fallback: if child has no parent (shouldn't happen with PDR, but handle gracefully)
            # Use child chunk directly
            # Only warn once per chunk_id per retrieval call to reduce noise
            if child_id not in warned_child_ids:
                log.warning('child_chunk_no_parent', child_id=child_id)
                warned_child_ids.add(child_id)
            final.append(child)
        else:
            # Orphan case: parent_id exists but parent chunk not found in database
            # This can happen if parent chunks were deleted or lookup failed
            # Use child chunk as fallback to ensure we return something
            if child_id not in warned_child_ids:
                log.warning(
                    'child_chunk_orphaned',
                    child_id=child_id,
                    parent_id=parent_id,
                    msg='Parent chunk not found in database, using child as fallback'
                )
                warned_child_ids.add(child_id)
            final.append(child)

    if trace is not None:
        trace_data = {
            'mode':                'vector',
            'raw_chunks_count':    len(results),
            'search_k':            search_k,
            'applied_filters':     applied_filters_for_trace,
            'where_clause':        where_clause,
            'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
            'search_elapsed_ms':   round(search_elapsed_ms, 1),
            'term_dictionary_version': term_expansion.dictionary_version,
            'term_dictionary_embedding_terms': term_expansion.embedding_terms,
            'term_dictionary_fts_terms': term_expansion.fts_terms,
            'term_dictionary_matches': [
                {
                    'alias': match.alias,
                    'canonical': match.canonical,
                    'match_type': match.match_type,
                    'tier': match.tier,
                }
                for match in term_expansion.matches
            ],
            'term_dictionary_fuzzy_cap_reached': term_expansion.fuzzy_cap_reached,
            'year_filter':         year_filter,
            'category_filter':     category_filter,
            'extension_filter':    extension_filter,
            'filename_filter':     filename_filter,
            'filename_exclude':    list(filename_exclude or []),
            'block_type_filter':   safe_block_type_filter,
            'block_type_exclude':  list(safe_block_type_exclude),
            'section_filter':      safe_section_filter,
            'max_score':           max_score,
            'file_ids_filter':     file_ids_filter,
            'exclude_upload_sources': exclude_upload_sources,
            'prefer_substantive_sections': prefer_substantive_sections,
            'prefer_title_alignment': prefer_title_alignment,
            'strict_title_alignment': strict_title_alignment,
            'term_expansion_enabled': enable_term_expansion,
            'prefer_within_file_diversity': prefer_within_file_diversity,
            'children_reranked':   len(reranked_children),
            'children_after_structural_filter': len(filtered_child_chunks),
            'parents_fetched':     len(parent_chunks),
            'fts5_augmented_count': fts5_augmented_count,
        }
        trace.record('retrieval', trace_data)
        trace.record('rerank', {
            'applied':     rerank_enabled,
            'input':       len(child_chunks),
            'output':      len(reranked_children),
            'children_returned': len(top_children),
            'parents_returned': len(final),
            'top_k_overlap_count': rerank_top_k_overlap,
            'top_k_changed_count': max(len(post_rerank_top_ids) - rerank_top_k_overlap, 0),
            'structural_filters_applied': bool(safe_block_type_filter or safe_section_filter),
            'elapsed_ms':  round(rerank_elapsed_ms, 1),
        })

    log.info(
        'retrieval_completed',
        query_type=query_type,
        mode='vector',
        query_length=len(query),
        raw_candidates=len(results),
        children_after_structural_filter=len(filtered_child_chunks),
        children_reranked=len(reranked_children),
        children_returned=len(top_children),
        parents_returned=len(final),
        timeout_occurred=False,
        embed_duration_ms=round(embed_elapsed_ms, 1),
        search_duration_ms=round(search_elapsed_ms, 1),
        rerank_duration_ms=round(rerank_elapsed_ms, 1),
    )

    if timing_output is not None:
        timing_output['embed_ms'] = round(embed_elapsed_ms, 1)
        timing_output['vector_search_ms'] = round(search_elapsed_ms, 1)
        timing_output['rerank_ms'] = round(rerank_elapsed_ms, 1)

    return final
