# ==============================================================================
# Informity AI — Search API Routes
# Endpoint for semantic search across indexed documents. Embeds the query,
# searches SQLite vector storage for similar chunks, and returns ranked results enriched
# with file metadata from SQLite.
# ==============================================================================

import asyncio

import aiosqlite
import structlog
from fastapi import APIRouter, Depends, HTTPException

from informity.api.schemas import SearchRequest, SearchResponse, SearchResult
from informity.api.security import EndpointGuard
from informity.db.sqlite import get_db, get_files_by_ids
from informity.db.vectors import vector_store
from informity.indexer.embedder import embedder
from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['search'])
SEARCH_GUARD = EndpointGuard(
    name='search',
    max_in_flight=4,
    max_requests_per_window=60,
    window_seconds=60,
)
MAX_SEARCH_QUERY_CHARS = 4000


# ==============================================================================
# POST /api/search — semantic search across documents
# ==============================================================================

@router.post('/api/search', response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> SearchResponse:
    # Semantic search flow:
    #   1. Embed the query text
    #   2. Search SQLite vector storage for the top-K most similar chunks
    #   3. Enrich results with file metadata from SQLite
    #   4. Apply optional category / file_type filters
    #   5. Return ranked SearchResponse

    async with SEARCH_GUARD.slot():
        query_text = request.query.strip()

        if not query_text:
            raise HTTPException(
                status_code=400,
                detail='Query cannot be empty',
            )
        if len(query_text) > MAX_SEARCH_QUERY_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f'Query too large (max {MAX_SEARCH_QUERY_CHARS} characters).',
            )

        log.info(
            'search_requested',
            query_length = len(query_text),
            limit        = request.limit,
            category     = request.category,
            file_types   = request.file_types,
        )

        # -- Step 1: Embed the query (in thread to avoid blocking event loop) -----
        query_vector = await asyncio.to_thread(embedder.embed_query, query_text)

        # We fetch more results than requested so that post-filtering by category
        # or file_type still yields enough results.
        fetch_limit = request.limit * 3 if (request.category or request.file_types) else request.limit

        # -- Step 2: Search SQLite vector storage -----------------------------------
        raw_results = await asyncio.to_thread(vector_store.search_similar, query_vector, fetch_limit)

        # -- Step 3 & 4: Batch-fetch file metadata and filter ----------------------
        # Collect all distinct file IDs from results for a single DB round-trip.
        all_file_ids = list({hit['file_id'] for hit in raw_results if hit.get('file_id') is not None})
        files_by_id  = await get_files_by_ids(db, all_file_ids)

        results: list[SearchResult] = []

        for hit in raw_results:
            if len(results) >= request.limit:
                break

            file_id = hit.get('file_id')
            if file_id is None:
                continue

            indexed_file = files_by_id.get(file_id)
            if indexed_file is None:
                # File was deleted from DB but vectors remain; skip
                log.debug('search_orphan_vector', file_id=file_id)
                continue

            # Apply category filter
            if request.category and indexed_file.category.value != request.category:
                continue

            # Apply file type filter
            if request.file_types and indexed_file.extension not in request.file_types:
                continue

            results.append(SearchResult(
                file_id  = indexed_file.id or file_id,
                filename = indexed_file.filename,
                path     = indexed_file.path,
                preview  = (hit.get('chunk_text', '') or '')[:MAX_EXTRACTED_TEXT_PREVIEW],
                score    = hit.get('score', 0.0),
                category = indexed_file.category.value,
            ))

        log.info(
            'search_completed',
            query_length   = len(query_text),
            results        = len(results),
            raw_candidates = len(raw_results),
        )

        return SearchResponse(
            results = results,
            total   = len(results),
            query   = query_text,
        )
