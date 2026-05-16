from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from informity.api.schemas import SearchResult
from informity.db.sqlite import (
    get_chunk_count,
    get_file_count,
    get_files,
    get_files_by_ids,
    get_latest_scan,
)
from informity.db.vectors import vector_store
from informity.indexer.embedder import embedder
from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW

MAX_MCP_RESULTS = 200
MAX_SNIPPET_CHARS = 1200
MAX_TOTAL_RESPONSE_BYTES = 500_000


@dataclass(slots=True)
class McpReadScope:
    mode: str = 'metadata_only'
    max_results: int = 50
    max_snippet_chars: int = 320
    max_total_response_bytes: int = MAX_TOTAL_RESPONSE_BYTES

    def normalize(self) -> McpReadScope:
        mode = str(self.mode or 'metadata_only').strip().lower()
        if mode not in {'metadata_only', 'search_snippets', 'full_chunks'}:
            mode = 'metadata_only'
        return McpReadScope(
            mode=mode,
            max_results=max(1, min(int(self.max_results), MAX_MCP_RESULTS)),
            max_snippet_chars=max(0, min(int(self.max_snippet_chars), MAX_SNIPPET_CHARS)),
            max_total_response_bytes=max(16_384, min(int(self.max_total_response_bytes), MAX_TOTAL_RESPONSE_BYTES)),
        )


def _coerce_response_size(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    raw = str(payload)
    if len(raw.encode('utf-8', errors='ignore')) <= max_bytes:
        return payload
    truncated = dict(payload)
    truncated['truncated'] = True
    return truncated


def _apply_scope_to_preview(preview: str, scope: McpReadScope) -> str | None:
    if scope.mode == 'metadata_only':
        return None
    if scope.mode == 'search_snippets':
        return (preview or '')[:scope.max_snippet_chars]
    return preview


async def tool_health() -> dict[str, Any]:
    return {
        'ok': True,
        'timestamp': datetime.now(UTC).isoformat(),
        'component': 'informity.mcp.readonly',
    }


async def tool_files_list(
    db: aiosqlite.Connection,
    *,
    scope: McpReadScope,
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
) -> dict[str, Any]:
    normalized_scope = scope.normalize()
    effective_limit = max(1, min(int(limit), normalized_scope.max_results))
    files, total = await get_files(
        db,
        search=search,
        limit=effective_limit,
        offset=max(0, int(offset)),
    )
    results = []
    for item in files:
        results.append({
            'file_id': int(item.id or 0),
            'filename': item.filename,
            'path': item.path,
            'category': item.category.value,
            'extension': item.extension,
            'indexed_at': item.indexed_at.isoformat() if item.indexed_at else None,
            'preview': None if normalized_scope.mode == 'metadata_only' else (item.extracted_text_preview or ''),
        })
    return _coerce_response_size(
        {
            'results': results,
            'total': int(total),
            'limit': effective_limit,
            'offset': max(0, int(offset)),
            'scope_mode': normalized_scope.mode,
        },
        normalized_scope.max_total_response_bytes,
    )


async def tool_search_semantic(
    db: aiosqlite.Connection,
    *,
    scope: McpReadScope,
    query: str,
    limit: int = 20,
    category: str | None = None,
    file_types: list[str] | None = None,
) -> dict[str, Any]:
    normalized_scope = scope.normalize()
    query_text = str(query or '').strip()
    if not query_text:
        raise ValueError('query cannot be empty')
    effective_limit = max(1, min(int(limit), normalized_scope.max_results))
    fetch_limit = effective_limit * 3 if (category or file_types) else effective_limit

    query_vector = await asyncio.to_thread(embedder.embed_query, query_text)
    raw_results = await asyncio.to_thread(vector_store.search_similar, query_vector, fetch_limit)
    all_file_ids = list({hit['file_id'] for hit in raw_results if hit.get('file_id') is not None})
    files_by_id = await get_files_by_ids(db, all_file_ids)

    results: list[dict[str, Any]] = []
    for hit in raw_results:
        if len(results) >= effective_limit:
            break
        file_id = hit.get('file_id')
        if file_id is None:
            continue
        indexed_file = files_by_id.get(file_id)
        if indexed_file is None:
            continue
        if category and indexed_file.category.value != category:
            continue
        if file_types and indexed_file.extension not in file_types:
            continue
        preview = (hit.get('chunk_text', '') or '')[:MAX_EXTRACTED_TEXT_PREVIEW]
        scoped_preview = _apply_scope_to_preview(preview, normalized_scope)

        result = SearchResult(
            file_id=indexed_file.id or int(file_id),
            filename=indexed_file.filename,
            path=indexed_file.path,
            preview=scoped_preview or '',
            score=float(hit.get('score', 0.0)),
            category=indexed_file.category.value,
        ).model_dump()
        if normalized_scope.mode == 'metadata_only':
            result.pop('preview', None)
        elif normalized_scope.mode == 'search_snippets':
            result['preview'] = scoped_preview or ''
        else:
            result['preview'] = preview
        results.append(result)

    return _coerce_response_size(
        {
            'query': query_text,
            'total': len(results),
            'results': results,
            'scope_mode': normalized_scope.mode,
        },
        normalized_scope.max_total_response_bytes,
    )


async def tool_index_status(db: aiosqlite.Connection) -> dict[str, Any]:
    total_files = await get_file_count(db)
    total_chunks = await get_chunk_count(db)
    return {
        'total_files': int(total_files),
        'total_chunks': int(total_chunks),
    }


async def tool_scan_status(db: aiosqlite.Connection) -> dict[str, Any]:
    latest = await get_latest_scan(db)
    if latest is None:
        return {'status': 'never_run'}
    return {
        'scan_id': int(latest.id or 0),
        'status': latest.status.value,
        'files_scanned': int(latest.files_scanned),
        'files_indexed': int(latest.files_indexed),
        'errors': int(latest.errors),
        'started_at': latest.started_at.isoformat() if latest.started_at else None,
        'completed_at': latest.completed_at.isoformat() if latest.completed_at else None,
    }

