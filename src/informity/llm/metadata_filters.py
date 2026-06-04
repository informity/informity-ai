"""
Informity AI — Metadata Filter SQL Helpers
Unified metadata filter WHERE-clause building.
"""

from dataclasses import dataclass

from informity.llm.types import FilterOperator


@dataclass(frozen=True)
class MetadataFilter:
    """
    A single metadata filter extracted from a query.

    Attributes:
        field: Metadata field name ('year', 'category', 'file_type', etc.)
        operator: Comparison operator ('eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in')
        value: Filter value (int, str, or list for 'in' operator)
    """
    field: str
    operator: FilterOperator
    value: int | str | list[int] | list[str]

_ALLOWED_FILTER_FIELDS = {'year', 'category', 'extension', 'filename', 'block_type', 'file_id'}


def _build_filter_sql(
    filter_item: MetadataFilter,
    params: list[int | str],
) -> str | None:
    if filter_item.field not in _ALLOWED_FILTER_FIELDS:
        return None

    col = filter_item.field
    op = filter_item.operator
    value = filter_item.value

    if op == FilterOperator.EQ:
        params.append(value)
        return f'{col} = ?'
    if op == FilterOperator.NE:
        params.append(value)
        return f'{col} != ?'
    if op == FilterOperator.GT:
        params.append(value)
        return f'{col} > ?'
    if op == FilterOperator.GTE:
        params.append(value)
        return f'{col} >= ?'
    if op == FilterOperator.LT:
        params.append(value)
        return f'{col} < ?'
    if op == FilterOperator.LTE:
        params.append(value)
        return f'{col} <= ?'

    if op == FilterOperator.IN:
        if isinstance(value, list) and value:
            placeholders = ', '.join('?' * len(value))
            params.extend(value)
            return f'{col} IN ({placeholders})'
        return None

    if op == FilterOperator.LIKE:
        if isinstance(value, str) and value:
            params.append(value)
            return f'{col} LIKE ?'
        return None

    if op == FilterOperator.CONTAINS_ANY:
        if not isinstance(value, list):
            return None
        terms = [str(item).strip() for item in value if str(item).strip()]
        if not terms:
            return None
        params.extend(f'%{term}%' for term in terms)
        term_clauses = [f'{col} LIKE ?' for _ in terms]
        return f"({' OR '.join(term_clauses)})"

    return None


def build_where_clause_and_params(
    filters: list[MetadataFilter],
) -> tuple[str | None, list[int | str]]:
    """
    Build parameterized SQL WHERE clause and params from metadata filters.

    Returns:
        tuple of (where_clause_without_WHERE_keyword, params)
    """
    params: list[int | str] = []
    clauses: list[str] = []
    for filter_item in filters:
        if clause := _build_filter_sql(filter_item, params):
            clauses.append(clause)
    if not clauses:
        return None, []
    return ' AND '.join(clauses), params
def build_where_clause(filters: list[MetadataFilter]) -> str | None:
    """
    Build SQL WHERE clause from a list of metadata filters.

    Args:
        filters: List of MetadataFilter objects

    Returns:
        WHERE clause string (e.g., "year = 2023 AND category = 'document'")
        or None if no filters
    """
    where_clause, _ = build_where_clause_and_params(filters)
    return where_clause
