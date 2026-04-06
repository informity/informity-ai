# ==============================================================================
# Informity AI — Metadata Query Handler
# Handles metadata queries (count, enumeration, file listing) using SQLite directly
# ==============================================================================

from collections.abc import AsyncGenerator
from dataclasses import replace

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage, IndexedFile
from informity.db.sqlite import (
    get_distinct_categories,
    get_distinct_years,
    row_to_indexed_file,
)
from informity.llm.contract_prompt_parser import EXPLICIT_YEAR_PATTERN
from informity.llm.query_classifier import QueryClassification
from informity.llm.query_patterns import (
    build_aggregation_pattern,
    build_count_pattern,
    build_enumeration_pattern,
    build_file_list_pattern,
)
from informity.llm.types import QueryType, StreamSignalTag

log = structlog.get_logger(__name__)

# File listing display constants
MAX_FILE_LIST_DISPLAY = 50  # Maximum number of files to display in file listing responses

# Pre-compiled patterns (reused across all handler instances)
_AGGREGATION_PATTERN = build_aggregation_pattern()
_COUNT_PATTERN = build_count_pattern()
_ENUMERATION_PATTERN = build_enumeration_pattern()
_FILE_LIST_PATTERN = build_file_list_pattern()
class MetadataHandler:
    """
    Handler for metadata queries (count, enumeration, file listing).

    Routes queries to SQLite directly instead of vector search.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match metadata queries."""
        return classification.intent == QueryType.METADATA

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
        diagnostics_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle metadata query by routing to appropriate SQLite query.

        Query types:
        - Count queries: "how many files", "how many PDFs"
        - Enumeration queries: "what years", "list all categories"
        - Aggregation queries: "date range", "earliest", "latest", "per year"
        - File listing queries: "list all files", "show files from 2023"
        """
        question_lower = question.lower()

        effective_classification = classification
        if classification.year_filter is None:
            fallback_year = self._extract_explicit_single_year_filter(question_lower)
            if fallback_year is not None:
                effective_classification = replace(classification, year_filter=fallback_year)

        if trace is not None:
            # Record intent section with query_type (for diagnostics/metrics extraction)
            trace.record('intent', {
                'intent': classification.intent,
                'query_type': QueryType.METADATA,  # Metadata queries always use metadata query_type
                'is_metadata_query': classification.is_metadata_query,
                'is_file_list_query': classification.is_file_list_query,
                'year_filter': effective_classification.year_filter,
            })

        # 1. Aggregation queries: "date range", "earliest", "latest", "per year"
        # Check aggregation before count to handle "how many files are from each year"
        if _AGGREGATION_PATTERN.search(question_lower):
            aggregation = await self._get_aggregation(db, question_lower, effective_classification)
            response = self._format_aggregation_response(aggregation, question_lower)
            yield response
            yield (StreamSignalTag.METRICS, {'query_type': QueryType.METADATA, 'raw_chunks_count': 0})
            yield []
            return

        # 2. Enumeration queries: "what years", "how many years", "what categories"
        # Check before count so "how many years" returns years count, not file count
        if _ENUMERATION_PATTERN.search(question_lower):
            enumeration = await self._get_enumeration(db, question_lower, effective_classification)
            response = self._format_enumeration_response(enumeration, question_lower)
            yield response
            yield (StreamSignalTag.METRICS, {'query_type': QueryType.METADATA, 'raw_chunks_count': 0})
            yield []
            return

        # 3. Count queries: "how many files", "how many PDFs"
        if _COUNT_PATTERN.search(question_lower):
            count = await self._get_count(db, effective_classification)
            response = self._format_count_response(count, effective_classification)
            yield response
            yield (StreamSignalTag.METRICS, {'query_type': QueryType.METADATA, 'raw_chunks_count': 0})
            yield []
            return

        # 4. File listing queries: explicit inventory/list requests only
        if _FILE_LIST_PATTERN.search(question_lower) or classification.is_file_list_query:
            files, total = await self._get_files_with_filters(db, effective_classification)
            response = self._format_file_list_response(files, total, effective_classification)
            yield response
            yield (StreamSignalTag.METRICS, {'query_type': QueryType.METADATA, 'raw_chunks_count': 0})
            yield []
            return

        # 5. Fallback: generic metadata response
        yield "I can help you with file counts, enumerations (years, categories, file types), aggregations (date ranges, per year), and file listings. Could you rephrase your question?"
        yield (StreamSignalTag.METRICS, {'query_type': QueryType.METADATA, 'raw_chunks_count': 0})
        yield []

    def _extract_explicit_single_year_filter(self, question_lower: str) -> int | None:
        if any(keyword in question_lower for keyword in ('per year', 'each year', 'all years', 'what years')):
            return None
        years = {int(match.group(0)) for match in EXPLICIT_YEAR_PATTERN.finditer(question_lower)}
        if len(years) != 1:
            return None
        year = next(iter(years))
        if not (1900 <= year <= 2099):
            return None
        return year

    async def _get_count(
        self,
        db: aiosqlite.Connection,
        classification: QueryClassification,
    ) -> int:
        """Get file count with optional filters."""
        # Build WHERE clause conditions
        conditions: list[str] = []
        params: list[str | int] = []

        if classification.year_filter:
            conditions.append('year = ?')
            params.append(classification.year_filter)

        if classification.category_filter:
            conditions.append('category = ?')
            params.append(classification.category_filter)

        if classification.file_type_filter:
            # Normalize extension (ensure it starts with dot)
            extension = classification.file_type_filter
            if not extension.startswith('.'):
                extension = f'.{extension}'
            conditions.append('extension = ?')
            params.append(extension)

        if classification.filename_filter:
            # Filename filter for metadata queries (e.g., "how many files named X.pdf")
            conditions.append('filename = ?')
            params.append(classification.filename_filter)

        where_clause = ''
        if conditions:
            where_clause = 'WHERE ' + ' AND '.join(conditions)

        cursor = await db.execute(
            f'SELECT COUNT(*) as cnt FROM files {where_clause}',
            params,
        )
        row = await cursor.fetchone()
        return int(row['cnt']) if row else 0

    async def _get_aggregation(
        self,
        db: aiosqlite.Connection,
        question_lower: str,
        classification: QueryClassification,
    ) -> dict[str, int | list[dict[str, int]]]:
        """
        Get aggregation data (date range, per year counts).

        Returns:
            dict with keys:
            - 'date_range': {'min': int, 'max': int} or None
            - 'per_year': list[{'year': int, 'count': int}] or None
        """
        result: dict[str, dict[str, int] | list[dict[str, int]] | None] = {}

        # Date range queries: "date range", "earliest", "latest", "oldest", "newest"
        if any(keyword in question_lower for keyword in ['date range', 'range of dates', 'earliest', 'latest', 'oldest', 'newest', 'min', 'max', 'minimum', 'maximum']):
            years = await get_distinct_years(db)
            if years:
                result['date_range'] = {'min': min(years), 'max': max(years)}
            else:
                result['date_range'] = None

        # Per year queries: "per year", "from each year", "how many files are from each year"
        if any(keyword in question_lower for keyword in ['per year', 'from each year', 'each year', 'grouped by year']):
            # Build WHERE clause for filters (category, file_type, filename)
            conditions: list[str] = []
            params: list[str | int] = []

            if classification.category_filter:
                conditions.append('category = ?')
                params.append(classification.category_filter)

            if classification.file_type_filter:
                extension = classification.file_type_filter
                if not extension.startswith('.'):
                    extension = f'.{extension}'
                conditions.append('extension = ?')
                params.append(extension)

            if classification.filename_filter:
                conditions.append('filename = ?')
                params.append(classification.filename_filter)

            # Always enforce year presence; append safely with or without existing filters.
            where_parts = list(conditions)
            where_parts.append('year IS NOT NULL')
            where_clause = 'WHERE ' + ' AND '.join(where_parts)

            # Count files per year
            query = f'''
                SELECT year, COUNT(*) as count
                FROM files
                {where_clause}
                GROUP BY year
                ORDER BY year ASC
            '''
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            result['per_year'] = [{'year': int(r['year']), 'count': int(r['count'])} for r in rows]

        return result

    async def _get_enumeration(
        self,
        db: aiosqlite.Connection,
        question_lower: str,
        classification: QueryClassification,
    ) -> dict[str, list[int] | list[str]]:
        """Get enumeration data (years, categories, file types)."""
        result: dict[str, list[int] | list[str]] = {}
        filename_pattern = classification.filename_filter if classification.filename_filter else None

        if 'year' in question_lower:
            years = await get_distinct_years(db, filename_pattern=filename_pattern)
            result['years'] = years

        if 'categor' in question_lower:
            categories = await get_distinct_categories(db)
            result['categories'] = categories

        if 'file type' in question_lower or 'extension' in question_lower:
            # Get distinct extensions from files
            cursor = await db.execute(
                'SELECT DISTINCT extension FROM files WHERE extension IS NOT NULL ORDER BY extension ASC',
            )
            rows = await cursor.fetchall()
            extensions = [str(r['extension']) for r in rows if r['extension']]
            result['extensions'] = extensions

        return result

    def _format_count_response(
        self,
        count: int,
        classification: QueryClassification,
    ) -> str:
        """Format count query response."""
        filters = []
        if classification.year_filter:
            filters.append(f"from {classification.year_filter}")
        if classification.category_filter:
            filters.append(f"in category '{classification.category_filter}'")
        if classification.file_type_filter:
            filters.append(f"with extension '{classification.file_type_filter}'")
        if classification.filename_filter:
            filters.append(f"named '{classification.filename_filter}'")

        filter_text = f" {', '.join(filters)}" if filters else ""
        return f"You have **{count}** file{'s' if count != 1 else ''}{filter_text}."

    def _format_aggregation_response(
        self,
        aggregation: dict[str, dict[str, int] | list[dict[str, int]] | None],
        question_lower: str,
    ) -> str:
        """Format aggregation query response."""
        parts = []

        if 'date_range' in aggregation:
            date_range = aggregation['date_range']
            if date_range:
                parts.append(f"**Date range:** {date_range['min']} to {date_range['max']}")
            else:
                parts.append("**Date range:** No year metadata found in files")

        if 'per_year' in aggregation:
            per_year = aggregation['per_year']
            if per_year:
                year_items = [f"{item['year']}: {item['count']} file{'s' if item['count'] != 1 else ''}" for item in per_year]
                parts.append("**Files per year:**\n" + '\n'.join(f"- {item}" for item in year_items))
            else:
                parts.append("**Files per year:** No files with year metadata found")

        if not parts:
            return "I couldn't determine what aggregation you'd like. Please specify: date range or per year counts."

        return '\n\n'.join(parts)

    def _format_enumeration_response(
        self,
        enumeration: dict[str, list[int] | list[str]],
        question_lower: str,
    ) -> str:
        """Format enumeration query response."""
        # Check if this is a "how many" query (wants count, not list)
        is_count_query = 'how many' in question_lower

        parts = []

        if 'years' in enumeration:
            years = enumeration['years']
            if is_count_query:
                # User wants count of years, not list
                count = len(years)
                if count > 0:
                    years_str = ', '.join(map(str, years))
                    parts.append(f"You have files from **{count}** year{'s' if count != 1 else ''}: {years_str}")
                else:
                    parts.append("You have files from **0** years (no year metadata found)")
            else:
                # User wants list of years
                if years:
                    years_str = ', '.join(map(str, years))
                    parts.append(f"**Years covered:** {years_str}")
                else:
                    parts.append("**Years covered:** None (no year metadata found)")

        if 'categories' in enumeration:
            categories = enumeration['categories']
            if is_count_query:
                count = len(categories)
                if count > 0:
                    categories_str = ', '.join(categories)
                    parts.append(f"You have files in **{count}** categor{'ies' if count != 1 else 'y'}: {categories_str}")
                else:
                    parts.append("You have files in **0** categories")
            else:
                if categories:
                    categories_str = ', '.join(categories)
                    parts.append(f"**Categories:** {categories_str}")
                else:
                    parts.append("**Categories:** None")

        if 'extensions' in enumeration:
            extensions = enumeration['extensions']
            if is_count_query:
                count = len(extensions)
                if count > 0:
                    extensions_str = ', '.join(extensions)
                    parts.append(f"You have **{count}** file type{'s' if count != 1 else ''}: {extensions_str}")
                else:
                    parts.append("You have **0** file types")
            else:
                if extensions:
                    extensions_str = ', '.join(extensions)
                    parts.append(f"**File types:** {extensions_str}")
                else:
                    parts.append("**File types:** None")

        if not parts:
            return "I couldn't determine what you'd like to enumerate. Please specify: years, categories, or file types."

        return '\n\n'.join(parts)

    async def _get_files_with_filters(
        self,
        db: aiosqlite.Connection,
        classification: QueryClassification,
    ) -> tuple[list[IndexedFile], int]:
        """Get files with all filters including year."""
        # Build WHERE clause conditions
        conditions: list[str] = []
        params: list[str | int] = []

        if classification.year_filter:
            conditions.append('year = ?')
            params.append(classification.year_filter)

        if classification.category_filter:
            conditions.append('category = ?')
            params.append(classification.category_filter)

        if classification.file_type_filter:
            # Normalize extension (ensure it starts with dot)
            extension = classification.file_type_filter
            if not extension.startswith('.'):
                extension = f'.{extension}'
            conditions.append('extension = ?')
            params.append(extension)

        if classification.filename_filter:
            conditions.append('filename = ?')
            params.append(classification.filename_filter)

        where_clause = ''
        if conditions:
            where_clause = 'WHERE ' + ' AND '.join(conditions)

        # Get count
        count_cursor = await db.execute(
            f'SELECT COUNT(*) as cnt FROM files {where_clause}',
            params,
        )
        count_row = await count_cursor.fetchone()
        total = int(count_row['cnt']) if count_row else 0

        # Get files (limit to 100 for listing)
        query_params = params + [100, 0]  # limit, offset
        cursor = await db.execute(
            f'SELECT * FROM files {where_clause} ORDER BY filename ASC LIMIT ? OFFSET ?',
            query_params,
        )
        rows = await cursor.fetchall()
        files = [row_to_indexed_file(row) for row in rows]

        return files, total

    def _format_file_list_response(
        self,
        files: list[IndexedFile],
        total: int,
        classification: QueryClassification,
    ) -> str:
        """Format file listing response."""
        if not files:
            filters = []
            if classification.year_filter:
                filters.append(f"from {classification.year_filter}")
            if classification.category_filter:
                filters.append(f"in category '{classification.category_filter}'")
            if classification.file_type_filter:
                filters.append(f"with extension '{classification.file_type_filter}'")
            if classification.filename_filter:
                filters.append(f"named '{classification.filename_filter}'")

            filter_text = f" matching {' and '.join(filters)}" if filters else ""
            return f"No files found{filter_text}."

        # Format as markdown list
        file_items = []
        for file in files[:MAX_FILE_LIST_DISPLAY]:
            file_items.append(f"- {file.filename}")

        if total > MAX_FILE_LIST_DISPLAY:
            file_items.append(f"\n*... and {total - MAX_FILE_LIST_DISPLAY} more files*")

        header = f"**Found {total} file{'s' if total != 1 else ''}:**\n\n"
        return header + '\n'.join(file_items)
