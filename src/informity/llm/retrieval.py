# ==============================================================================
# Informity AI — Retrieval (v2)
# One retrieval path: embed → vector search with WHERE → rerank → top-k
# ==============================================================================

import asyncio
import time

import aiosqlite
import structlog

from informity.db.sqlite import get_chunks_by_parent_ids
from informity.db.vectors import vector_store
from informity.indexer.embedder import embedder
from informity.indexer.reranker import reranker
from informity.llm.metadata_filters import (
    MetadataFilter,
    build_where_clause_and_params,
    extract_metadata_filters,
)
from informity.config import settings
from informity.llm.model_adapter import get_profile

log = structlog.get_logger(__name__)


def _coerce_reranker_score(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None

async def _get_file_ids_matching_metadata_filters(
    db: aiosqlite.Connection,
    filters: list[MetadataFilter],
) -> list[int]:
    where_clause, where_params = build_where_clause_and_params(filters)
    query = 'SELECT id FROM files'
    if where_clause:
        query = f'{query} WHERE {where_clause}'
        cursor = await db.execute(query, where_params)
    else:
        cursor = await db.execute(query)
    rows = await cursor.fetchall()
    return [row['id'] for row in rows]


async def retrieve_chunks(
    query: str,
    top_k: int,
    max_score: float | None = None,
    year_filter: int | None = None,
    category_filter: str | None = None,
    extension_filter: str | None = None,
    filename_filter: str | None = None,
    source_terms_filter: list[str] | None = None,
    block_type_filter: str | None = None,
    section_filter: str | None = None,
    query_type: str = 'focused',  # 'focused' or 'coverage'
    db: aiosqlite.Connection | None = None,
    trace: object | None = None,
    timing_output: dict | None = None,
) -> list[dict]:
    """
    Unified retrieval path: embed query → vector search (optional WHERE) → rerank → top-k.

    For coverage queries (query_type='coverage'), uses file-anchored retrieval:
    - Gets all file_ids matching filters
    - Retrieves top-1 child chunk per file
    - Ensures all matching files are represented (exhaustive, not probabilistic)

    If trace is provided (TraceWriter protocol), records 'retrieval' and 'rerank' steps.
    """
    # 1. Embed query (CPU-bound, run in thread pool to avoid blocking event loop)
    embed_start = time.perf_counter()
    query_vector = await asyncio.to_thread(embedder.embed_query, query)
    embed_elapsed_ms = (time.perf_counter() - embed_start) * 1000

    start = time.perf_counter()

    # 2. Build WHERE clause from filters using unified filter system
    filters: list[MetadataFilter] = []
    if year_filter:
        filters.append(MetadataFilter(field='year', operator='eq', value=year_filter))
    if category_filter:
        # Sanitize category filter (only alphanumeric, underscore, hyphen)
        safe_category = ''.join(c for c in category_filter if c.isalnum() or c in '_-')
        if safe_category:
            filters.append(MetadataFilter(field='category', operator='eq', value=safe_category))
    if extension_filter:
        # Sanitize extension filter (ensure it starts with dot)
        safe_extension = extension_filter if extension_filter.startswith('.') else f'.{extension_filter}'
        filters.append(MetadataFilter(field='extension', operator='eq', value=safe_extension))
    if filename_filter:
        # Prefer contains matching for retrieval to avoid brittle exact-equality misses
        # when classifier output omits filename prefixes/suffixes (for example, leading year).
        normalized_filename_filter = filename_filter.strip()
        if normalized_filename_filter:
            filters.append(
                MetadataFilter(
                    field='filename',
                    operator='like',
                    value=f'%{normalized_filename_filter}%',
                )
            )
    elif source_terms_filter:
        normalized_source_terms = [
            term.strip()
            for term in source_terms_filter
            if isinstance(term, str) and term.strip()
        ]
        if normalized_source_terms:
            filters.append(
                MetadataFilter(
                    field='filename',
                    operator='contains_any',
                    value=list(dict.fromkeys(normalized_source_terms)),
                )
            )

    # Add generic query-derived metadata filters when they were not provided via classification.
    # This keeps retrieval data-agnostic while allowing explicit user constraints like
    # `filename contains "X" or "Y"` in focused/coverage queries.
    extracted_filters = extract_metadata_filters(query)
    has_year_filter = year_filter is not None
    has_extension_filter = extension_filter is not None
    has_filename_filter = filename_filter is not None
    has_source_terms_filter = bool(source_terms_filter)
    for extracted_filter in extracted_filters:
        if (
            extracted_filter.field == 'year' and not has_year_filter
            or extracted_filter.field == 'extension' and not has_extension_filter
            or extracted_filter.field == 'filename' and not has_filename_filter and not has_source_terms_filter
        ):
            if extracted_filter.field == 'filename' and extracted_filter.operator == 'eq':
                normalized_extracted_filename = str(extracted_filter.value).strip()
                if normalized_extracted_filename:
                    filters.append(
                        MetadataFilter(
                            field='filename',
                            operator='like',
                            value=f'%{normalized_extracted_filename}%',
                        )
                    )
                continue
            filters.append(extracted_filter)

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
    safe_block_type_filter = block_type_filter if block_type_filter in {'table', 'form', 'narrative'} else None
    safe_section_filter = section_filter.strip().casefold() if isinstance(section_filter, str) and section_filter.strip() else None

    # Initialize file_ids and seen_files at function scope (used for coverage queries and trace metrics)
    file_ids: list[int] = []
    seen_files: set[int] = set()  # Track files covered by global search (for file-anchored retrieval)
    fts5_augmented_count: int = 0  # Net-new candidates added by FTS5 augmentation (focused only)
    profile = get_profile()

    # 3. File-anchored retrieval for coverage queries
    filename_filter_relaxed = False

    if query_type == 'coverage' and db is not None:
        # Get all file_ids matching structured metadata filters.
        # This preserves coverage semantics for constrained sets (including filename contains).
        file_ids = await _get_file_ids_matching_metadata_filters(db, active_filters)

        if not file_ids:
            if trace is not None:
                trace.record('retrieval', {
                    'mode':                'file_anchored',
                    'raw_chunks_count':    0,
                    'matching_files':      0,
                    'applied_filters':     applied_filters_for_trace,
                    'where_clause':        where_clause,
                    'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
                    'search_elapsed_ms':    0,
                    'year_filter':          year_filter,
                    'category_filter':      category_filter,
                    'extension_filter':     extension_filter,
                    'filename_filter':      filename_filter,
                    'source_terms_filter':  source_terms_filter,
                    'block_type_filter':    safe_block_type_filter,
                    'section_filter':       safe_section_filter,
                    'max_score':            max_score,
                })
            return []

        # File-anchored retrieval strategy: single global search with fallback for exhaustive coverage
        # Fast path: single query covers most files (performance win)
        # Fallback: per-file searches for missing files (accuracy guarantee)
        coverage_candidate_multiplier = max(1, int(getattr(profile, 'coverage_candidate_multiplier', 3)))
        coverage_min_candidates = max(1, int(getattr(profile, 'coverage_min_candidates', 50)))
        candidate_limit = max(len(file_ids) * coverage_candidate_multiplier, coverage_min_candidates)
        all_results = await asyncio.to_thread(
            vector_store.search_similar,
            query_vector,
            candidate_limit,
            where_clause,  # Original filters (year, category, extension)
            where_params,
        )
        if max_score is not None:
            raw_before_score_filter = len(all_results)
            all_results = [r for r in all_results if r.get('score', float('inf')) <= max_score]
            if raw_before_score_filter > 0 and not all_results:
                log.info(
                    'l2_threshold_eliminated_all_candidates',
                    query_type='coverage',
                    max_score=max_score,
                    raw_candidates=raw_before_score_filter,
                )

        # Group by file_id and take top-1 per file
        file_id_set = set(file_ids)
        seen_files.clear()  # Reset for this query
        all_child_chunks: list[dict] = []

        # Sort by score (lower = more similar for L2 distance)
        for r in sorted(all_results, key=lambda x: x['score']):
            if r['file_id'] in file_id_set and r['file_id'] not in seen_files:
                all_child_chunks.append(r)
                seen_files.add(r['file_id'])

        # Fallback: if we haven't covered all files, fetch top-1 per missing file in one query
        # to avoid N+1 vector queries while preserving exhaustive coverage semantics.
        missing_file_ids = file_id_set - seen_files
        if missing_file_ids:
            per_file_fallback_limit = int(settings.coverage_per_file_fallback_limit)
            capped = len(missing_file_ids) > per_file_fallback_limit
            fallback_file_ids = sorted(missing_file_ids)[:per_file_fallback_limit]
            if capped:
                log.info(
                    'coverage_fallback_batch_capped',
                    missing_count=len(missing_file_ids),
                    cap=per_file_fallback_limit,
                    skipped_count=len(missing_file_ids) - per_file_fallback_limit,
                    total_files=len(file_ids),
                    coverage_pct=round((len(seen_files) / len(file_ids)) * 100, 1),
                )
            else:
                log.debug(
                    'coverage_fallback_batch',
                    missing_count=len(missing_file_ids),
                    total_files=len(file_ids),
                    coverage_pct=round((len(seen_files) / len(file_ids)) * 100, 1),
                )
            fallback_results = await asyncio.to_thread(
                vector_store.search_top1_per_file,
                query_vector,
                fallback_file_ids,
                where_clause,
                where_params,
            )
            if max_score is not None:
                fallback_results = [r for r in fallback_results if r.get('score', float('inf')) <= max_score]
            if fallback_results:
                all_child_chunks.extend(fallback_results)

        search_elapsed_ms = (time.perf_counter() - start) * 1000

        if not all_child_chunks:
            if trace is not None:
                trace.record('retrieval', {
                    'mode':                'file_anchored',
                    'raw_chunks_count':    0,
                    'matching_files':      len(file_ids),
                    'applied_filters':     applied_filters_for_trace,
                    'where_clause':        where_clause,
                    'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
                    'search_elapsed_ms':   round(search_elapsed_ms, 1),
                    'year_filter':         year_filter,
                    'category_filter':     category_filter,
                    'extension_filter':    extension_filter,
                    'filename_filter':     filename_filter,
                    'source_terms_filter': source_terms_filter,
                    'block_type_filter':   safe_block_type_filter,
                    'section_filter':      safe_section_filter,
                    'max_score':           max_score,
                })
            return []

        # Use collected child chunks as results (will be reranked below)
        results = all_child_chunks
        search_k = len(results)  # Track how many we collected

    else:
        # Standard retrieval for focused queries (probabilistic, top-k globally)
        # CPU-bound vector search, run in thread pool to avoid blocking event loop
        search_k = top_k * 2
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
                    query_type='focused',
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
                query,
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
        retrieval_mode = 'file_anchored' if query_type == 'coverage' else 'vector'
        log.info(
            'retrieval_completed',
            query_type=query_type,
            mode=retrieval_mode,
            query_length=len(query),
            matching_files=len(file_ids) if query_type == 'coverage' else None,
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
                'mode':                retrieval_mode,
                'raw_chunks_count':    0,
                'applied_filters':     applied_filters_for_trace,
                'where_clause':        where_clause,
                'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
                'search_elapsed_ms':   round(search_elapsed_ms, 1),
                'year_filter':          year_filter,
                'category_filter':     category_filter,
                'extension_filter':    extension_filter,
                'filename_filter':     filename_filter,
                'source_terms_filter': source_terms_filter,
                'block_type_filter':   safe_block_type_filter,
                'section_filter':      safe_section_filter,
                'max_score':           max_score,
                'filename_filter_relaxed': filename_filter_relaxed,
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
    top_children = reranked_children[:top_k]
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
        retrieval_mode = 'file_anchored' if query_type == 'coverage' else 'vector'
        trace_data = {
            'mode':                retrieval_mode,
            'raw_chunks_count':    len(results),
            'search_k':            search_k,
            'applied_filters':     applied_filters_for_trace,
            'where_clause':        where_clause,
            'embed_elapsed_ms':    round(embed_elapsed_ms, 1),
            'search_elapsed_ms':   round(search_elapsed_ms, 1),
            'year_filter':         year_filter,
            'category_filter':     category_filter,
            'extension_filter':    extension_filter,
            'filename_filter':     filename_filter,
            'source_terms_filter': source_terms_filter,
            'block_type_filter':   safe_block_type_filter,
            'section_filter':      safe_section_filter,
            'max_score':           max_score,
            'children_reranked':   len(reranked_children),
            'children_after_structural_filter': len(filtered_child_chunks),
            'parents_fetched':     len(parent_chunks),
            'filename_filter_relaxed': filename_filter_relaxed,
            'fts5_augmented_count': fts5_augmented_count if query_type != 'coverage' else 0,
        }
        if query_type == 'coverage' and db is not None:
            # Add file-anchored retrieval metrics (reuse file_ids from coverage query branch)
            trace_data['matching_files'] = len(file_ids)
            trace_data['files_covered_by_global_search'] = len(seen_files)
            trace_data['files_covered_by_fallback'] = len(file_ids) - len(seen_files)
            trace_data['files_covered_after_fallback'] = len(file_ids)
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

    retrieval_mode = 'file_anchored' if query_type == 'coverage' else 'vector'
    log.info(
        'retrieval_completed',
        query_type=query_type,
        mode=retrieval_mode,
        query_length=len(query),
        matching_files=len(file_ids) if query_type == 'coverage' else None,
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
