from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from informity.api.schemas import SearchResult
from informity.db.sqlite import (
    get_files,
    get_files_by_ids,
    get_latest_scan,
)
from informity.db.vectors import vector_store
from informity.indexer.embedder import embedder
from informity.mcp.categories import MCP_FILE_CATEGORIES, MCP_FILE_CATEGORIES_SET
from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW

MAX_MCP_RESULTS = 200
MAX_SNIPPET_CHARS = 1200
MAX_TOTAL_RESPONSE_BYTES = 500_000
FILE_TYPE_ALIASES: dict[str, str] = {
    'pdf': '.pdf',
    'docx': '.docx',
    'pptx': '.pptx',
    'epub': '.epub',
    'txt': '.txt',
    'md': '.md',
    'rst': '.rst',
    'log': '.log',
    'csv': '.csv',
    'xlsx': '.xlsx',
    'html': '.html',
    'htm': '.htm',
    'json': '.json',
    'yaml': '.yaml',
    'yml': '.yml',
    'toml': '.toml',
}


@dataclass(slots=True)
class McpReadScope:
    mode: str = 'metadata_only'
    max_results: int = MAX_MCP_RESULTS
    max_snippet_chars: int = 320
    max_total_response_bytes: int = MAX_TOTAL_RESPONSE_BYTES

    def normalize(self) -> McpReadScope:
        mode = str(self.mode or 'metadata_only').strip().lower()
        if mode not in {'metadata_only', 'search_snippets', 'full_content'}:
            mode = 'metadata_only'
        return McpReadScope(
            mode=mode,
            max_results=max(1, min(int(self.max_results), MAX_MCP_RESULTS)),
            max_snippet_chars=max(0, min(int(self.max_snippet_chars), MAX_SNIPPET_CHARS)),
            max_total_response_bytes=max(16_384, min(int(self.max_total_response_bytes), MAX_TOTAL_RESPONSE_BYTES)),
        )


def _coerce_response_size(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    if _serialized_size_bytes(payload) <= max_bytes:
        return payload
    results = payload.get('results')
    if not isinstance(results, list):
        truncated = dict(payload)
        truncated['truncated'] = True
        return truncated

    trimmed = dict(payload)
    original_total = len(results)
    trimmed_results: list[Any] = list(results)
    while trimmed_results:
        candidate = dict(trimmed)
        candidate['results'] = trimmed_results
        candidate['truncated'] = True
        candidate['returned'] = len(trimmed_results)
        candidate['total_before_truncation'] = original_total
        if _serialized_size_bytes(candidate) <= max_bytes:
            return candidate
        trimmed_results = trimmed_results[:-1]

    return {
        'truncated': True,
        'returned': 0,
        'total_before_truncation': original_total,
    }


def _serialized_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8', errors='ignore'))


def _apply_scope_to_preview(preview: str, scope: McpReadScope) -> str | None:
    if scope.mode == 'metadata_only':
        return None
    if scope.mode == 'search_snippets':
        return (preview or '')[:scope.max_snippet_chars]
    return preview


def _normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    normalized = str(category).strip().lower()
    return normalized if normalized in MCP_FILE_CATEGORIES_SET else normalized or None


def _normalize_file_types(file_types: list[str] | None) -> set[str] | None:
    if not file_types:
        return None
    normalized: set[str] = set()
    for raw in file_types:
        item = str(raw or '').strip().lower()
        if not item:
            continue
        canonical = FILE_TYPE_ALIASES.get(item, item)
        if not canonical.startswith('.'):
            canonical = f'.{canonical}'
        normalized.add(canonical)
    return normalized or None


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
    limit: int = 50,
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
        excluded_source_providers=['upload.local'],
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
    limit: int = 50,
    category: str | None = None,
    file_types: list[str] | None = None,
) -> dict[str, Any]:
    normalized_scope = scope.normalize()
    query_text = str(query or '').strip()
    if not query_text:
        raise ValueError('query cannot be empty')
    normalized_category = _normalize_category(category)
    normalized_file_types = _normalize_file_types(file_types)
    effective_limit = max(1, min(int(limit), normalized_scope.max_results))
    fetch_limit = effective_limit * 3 if (normalized_category or normalized_file_types) else effective_limit

    query_vector = await asyncio.to_thread(embedder.embed_query, query_text)
    raw_results = await asyncio.to_thread(vector_store.search_similar, query_vector, fetch_limit)
    all_file_ids = list({hit['file_id'] for hit in raw_results if hit.get('file_id') is not None})
    files_by_id = await get_files_by_ids(db, all_file_ids)

    results: list[dict[str, Any]] = []
    seen_content_hashes: set[str] = set()
    for hit in raw_results:
        if len(results) >= effective_limit:
            break
        file_id = hit.get('file_id')
        if file_id is None:
            continue
        indexed_file = files_by_id.get(file_id)
        if indexed_file is None:
            continue
        if str(getattr(indexed_file, 'source_provider', '') or '').strip().lower() == 'upload.local':
            continue
        if normalized_category and indexed_file.category.value != normalized_category:
            continue
        if normalized_file_types and str(indexed_file.extension or '').strip().lower() not in normalized_file_types:
            continue
        content_hash = str(getattr(indexed_file, 'content_hash', '') or '').strip().lower()
        if content_hash:
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
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

    response_payload = {
        'query': query_text,
        'total': len(results),
        'results': results,
        'scope_mode': normalized_scope.mode,
    }
    if len(results) == 0 and (normalized_category or normalized_file_types):
        filter_options = await _get_filter_options(db)
        response_payload = _with_no_result_hints(
            response_payload,
            normalized_category=normalized_category,
            normalized_file_types=normalized_file_types,
            filter_options=filter_options,
        )

    return _coerce_response_size(response_payload, normalized_scope.max_total_response_bytes)


def _with_no_result_hints(
    payload: dict[str, Any],
    *,
    normalized_category: str | None,
    normalized_file_types: set[str] | None,
    filter_options: dict[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    available_categories = list(filter_options.get('categories', []))
    available_file_types = list(filter_options.get('file_types', []))
    category_set = {str(item).strip().lower() for item in available_categories}
    file_type_set = {str(item).strip().lower() for item in available_file_types}
    unknown_file_types = sorted(
        [item for item in (normalized_file_types or set()) if item not in file_type_set]
    )
    result['hints'] = {
        'reason': 'No results matched current filters',
        'applied_filters': {
            'category': normalized_category,
            'file_types': sorted(normalized_file_types) if normalized_file_types else [],
        },
        'valid_categories': available_categories,
        'valid_file_types': available_file_types,
        'unknown_filters': {
            'unknown_category': bool(normalized_category and normalized_category not in category_set),
            'unknown_file_types': unknown_file_types,
        },
        'guidance': 'Try removing filters or use informity_filter_options for valid values.',
    }
    return result


async def tool_filter_options(db: aiosqlite.Connection) -> dict[str, Any]:
    filter_options = await _get_filter_options(db)
    return {
        'categories': filter_options['categories'],
        'file_types': filter_options['file_types'],
        'notes': {
            'category': 'Optional filter for informity_search_semantic.',
            'file_types': 'Optional filter. Use dot extensions like .pdf.',
        },
    }


async def _get_filter_options(db: aiosqlite.Connection) -> dict[str, list[str]]:
    categories_cursor = await db.execute(
        '''
        SELECT DISTINCT LOWER(category) AS category
        FROM files
        WHERE source_provider != ?
        ORDER BY category ASC
        ''',
        ('upload.local',),
    )
    category_rows = await categories_cursor.fetchall()
    categories = [str(row['category']) for row in category_rows if row and row['category']]
    if not categories:
        categories = sorted(MCP_FILE_CATEGORIES)

    extension_cursor = await db.execute(
        '''
        SELECT DISTINCT LOWER(extension) AS extension
        FROM files
        WHERE source_provider != ? AND extension IS NOT NULL AND TRIM(extension) != ''
        ORDER BY extension ASC
        ''',
        ('upload.local',),
    )
    extension_rows = await extension_cursor.fetchall()
    file_types = [str(row['extension']) for row in extension_rows if row and row['extension']]
    return {
        'categories': categories,
        'file_types': file_types,
    }


async def tool_index_status(db: aiosqlite.Connection) -> dict[str, Any]:
    files_cursor = await db.execute(
        '''
        SELECT COUNT(*) as count
        FROM files
        WHERE source_provider != ?
        ''',
        ('upload.local',),
    )
    files_row = await files_cursor.fetchone()
    total_files = int(files_row['count']) if files_row else 0

    chunks_cursor = await db.execute(
        '''
        SELECT COUNT(*) as count
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE f.source_provider != ?
        ''',
        ('upload.local',),
    )
    chunks_row = await chunks_cursor.fetchone()
    total_chunks = int(chunks_row['count']) if chunks_row else 0
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
