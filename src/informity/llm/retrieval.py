# ==============================================================================
# Informity AI — Retrieval (v2)
# One retrieval path: embed → vector search with WHERE → rerank → top-k
# ==============================================================================

import asyncio
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
from informity.llm.term_dictionary import expand_query_for_retrieval
from informity.llm.types import BlockType, FilterOperator, QueryType

log = structlog.get_logger(__name__)
_COVERAGE_DIVERSITY_PRIMARY_FILE_CAP = 1
_COVERAGE_DIVERSITY_SECONDARY_FILE_CAP = 2


def _coerce_reranker_score(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _select_top_children(
    *,
    reranked_children: list[dict],
    top_k: int,
    query_type: QueryType,
) -> list[dict]:
    if top_k <= 0:
        return []
    if query_type != QueryType.COVERAGE:
        return reranked_children[:top_k]

    selected: list[dict] = []
    seen_chunk_ids: set[int] = set()
    file_counts: dict[int, int] = {}

    def _try_add(chunk: dict, *, per_file_cap: int | None) -> None:
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

        selected.append(chunk)
        seen_chunk_ids.add(chunk_id)
        if file_id is not None:
            file_counts[file_id] = file_counts.get(file_id, 0) + 1

    for per_file_cap in (
        _COVERAGE_DIVERSITY_PRIMARY_FILE_CAP,
        _COVERAGE_DIVERSITY_SECONDARY_FILE_CAP,
        None,
    ):
        for chunk in reranked_children:
            _try_add(chunk, per_file_cap=per_file_cap)
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
    block_type_filter: str | None = None,
    section_filter: str | None = None,
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
    term_expansion = await expand_query_for_retrieval(db=db, query=query)
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
        safe_extension = extension_filter if extension_filter.startswith('.') else f'.{extension_filter}'
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

    active_filters = list(filters)
    where_clause, where_params = build_where_clause_and_params(active_filters)
    applied_filters_for_trace = [
        {
            'field': metadata_filter.field,
            'operator': metadata_filter.operator,
            'value': metadata_filter.value,
        }
        for metadata_filter in active_filters
    ]
    safe_block_type_filter = (
        block_type_filter
        if block_type_filter in {BlockType.TABLE, BlockType.FORM, BlockType.NARRATIVE}
        else None
    )
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
                'block_type_filter':   safe_block_type_filter,
                'section_filter':      safe_section_filter,
                'max_score':           max_score,
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

    # Preserve order from vector search results
    child_chunks: list[dict] = [chunk_id_to_dict[cid] for cid in child_chunk_ids if cid in chunk_id_to_dict]

    # Apply structure-aware filters on fetched chunks (same unified retrieval path).
    filtered_child_chunks = child_chunks
    if safe_block_type_filter is not None:
        block_filtered = [chunk for chunk in filtered_child_chunks if chunk.get('block_type') == safe_block_type_filter]
        if block_filtered:
            filtered_child_chunks = block_filtered
    if safe_section_filter is not None:
        section_filtered = [
            chunk
            for chunk in filtered_child_chunks
            if isinstance(chunk.get('section_path'), str) and safe_section_filter in chunk['section_path'].casefold()
        ]
        if section_filtered:
            filtered_child_chunks = section_filtered

    # 5. Rerank child chunks (mandatory for all queries)
    # CPU-bound cross-encoder, run in thread pool to avoid blocking event loop
    # NOTE(2.1): We intentionally do NOT mutate reranker scores after this point.
    # Historical behavior (removed): add +0.15 for block_type match and +0.10 for
    # section_path match, then resort by boosted score.
    # Rationale: keep ranking semantics model-consistent and avoid heuristic
    # post-processing bandaids. To re-evaluate, restore the old boost block right
    # below reranker.rerank(...) and compare structure-constrained query quality.
    rerank_start = time.perf_counter()
    reranked_children = await asyncio.to_thread(reranker.rerank, query, filtered_child_chunks)
    pre_rerank_top_ids = [chunk.get('chunk_id') for chunk in filtered_child_chunks[:top_k]]
    rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000
    top_children = _select_top_children(
        reranked_children=reranked_children,
        top_k=top_k,
        query_type=query_type,
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
            'block_type_filter':   safe_block_type_filter,
            'section_filter':      safe_section_filter,
            'max_score':           max_score,
            'children_reranked':   len(reranked_children),
            'children_after_structural_filter': len(filtered_child_chunks),
            'parents_fetched':     len(parent_chunks),
            'fts5_augmented_count': fts5_augmented_count,
        }
        trace.record('retrieval', trace_data)
        trace.record('rerank', {
            'applied':     True,
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
