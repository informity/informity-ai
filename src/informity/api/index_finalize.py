"""Module for api index finalize."""

from __future__ import annotations

import aiosqlite
import structlog

from informity.db.sqlite import get_index_integrity_issues
from informity.indexer.adaptive_tuning import update_tuning_cache
from informity.indexer.term_dictionary_builder import rebuild_term_dictionary

log = structlog.get_logger(__name__)

_FINALIZE_INDEX_EXCEPTIONS = (ImportError, OSError, RuntimeError, ValueError, TypeError)


async def finalize_index_operation(
    db: aiosqlite.Connection,
    *,
    scan_id: int,
    integrity_log_name: str,
    adaptive_tuning_failure_log_name: str,
    term_dictionary_log_name: str,
    term_dictionary_failure_log_name: str,
    term_dictionary_run_id_prefix: str,
    vector_skip_log_name: str | None = None,
    vector_skip_reason: str = "exact_search_mode",
) -> None:
    """Finalize index operation."""
    integrity_issues = await get_index_integrity_issues(db)
    non_zero_issues = {key: value for key, value in integrity_issues.items() if value > 0}
    if non_zero_issues:
        log.error(
            integrity_log_name,
            scan_id=scan_id,
            issues=non_zero_issues,
        )

    try:
        await update_tuning_cache(db, force_recompute=True)
    except _FINALIZE_INDEX_EXCEPTIONS as exc:
        log.warning(adaptive_tuning_failure_log_name, error=str(exc))

    try:
        term_dictionary_result = await rebuild_term_dictionary(
            db,
            run_id=f"{term_dictionary_run_id_prefix}{scan_id}",
        )
        log.info(term_dictionary_log_name, scan_id=scan_id, result=term_dictionary_result)
    except _FINALIZE_INDEX_EXCEPTIONS as exc:
        log.warning(term_dictionary_failure_log_name, scan_id=scan_id, error=str(exc))

    if vector_skip_log_name is not None:
        log.debug(vector_skip_log_name, reason=vector_skip_reason)
