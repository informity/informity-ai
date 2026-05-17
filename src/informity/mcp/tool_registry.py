from __future__ import annotations

from typing import Any

JSON = dict[str, Any]

TOOL_HEALTH = 'informity_health'
TOOL_FILES_LIST = 'informity_files_list'
TOOL_SEARCH_SEMANTIC = 'informity_search_semantic'
TOOL_FILTER_OPTIONS = 'informity_filter_options'
TOOL_INDEX_STATUS = 'informity_index_status'
TOOL_SCAN_STATUS = 'informity_scan_status'
READONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_HEALTH,
        TOOL_FILES_LIST,
        TOOL_SEARCH_SEMANTIC,
        TOOL_FILTER_OPTIONS,
        TOOL_INDEX_STATUS,
        TOOL_SCAN_STATUS,
    }
)

LEGACY_TOOL_ALIASES: dict[str, str] = {
    'informity.health': TOOL_HEALTH,
    'informity.files.list': TOOL_FILES_LIST,
    'informity.search.semantic': TOOL_SEARCH_SEMANTIC,
    'informity.filter.options': TOOL_FILTER_OPTIONS,
    'informity.index.status': TOOL_INDEX_STATUS,
    'informity.scan.status': TOOL_SCAN_STATUS,
}

TOOLS: list[JSON] = [
    {
        'name': TOOL_HEALTH,
        'description': 'Returns MCP service health status.',
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {},
        },
    },
    {
        'name': TOOL_FILES_LIST,
        'description': 'Lists indexed files with optional search/pagination.',
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                'offset': {'type': 'integer', 'minimum': 0},
                'search': {'type': 'string'},
            },
        },
    },
    {
        'name': TOOL_SEARCH_SEMANTIC,
        'description': (
            'Performs semantic search across indexed files. '
            'Use query (required) and optionally limit/category/file_types. '
            'category/file_types are optional filters, not required.'
        ),
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Natural-language search query.',
                },
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                'category': {
                    'type': 'string',
                    'enum': ['document', 'plaintext', 'data', 'web', 'other'],
                    'description': 'Optional filter. Leave unset to search all categories.',
                },
                'file_types': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Optional filter by extension. '
                        'Accepts dot extensions like .pdf or aliases like pdf/docx.'
                    ),
                },
            },
            'required': ['query'],
        },
    },
    {
        'name': TOOL_INDEX_STATUS,
        'description': 'Returns index totals (files/chunks).',
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {},
        },
    },
    {
        'name': TOOL_FILTER_OPTIONS,
        'description': 'Returns valid categories and file type extensions for MCP filtering.',
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {},
        },
    },
    {
        'name': TOOL_SCAN_STATUS,
        'description': 'Returns latest scan status.',
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {},
        },
    },
]
