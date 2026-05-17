from __future__ import annotations

from informity.db.models import FileCategory

MCP_FILE_CATEGORIES: tuple[str, ...] = tuple(category.value for category in FileCategory)
MCP_FILE_CATEGORIES_SET: frozenset[str] = frozenset(MCP_FILE_CATEGORIES)
