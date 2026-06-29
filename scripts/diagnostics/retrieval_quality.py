#!/usr/bin/env python3
"""Read-only retrieval quality diagnostic."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from informity.config import settings  # noqa: E402
from informity.db.sqlite import get_connection, get_chunks_by_parent_ids  # noqa: E402
from informity.db.vectors import vector_store  # noqa: E402
from informity.indexer.embedder import embedder  # noqa: E402
from informity.indexer.reranker import reranker  # noqa: E402
from informity.llm.handlers.rag import (  # noqa: E402
    _build_history_aware_retrieval_query_with_classification,
    _evaluate_minimal_answerability,
    _fetch_term_inventory_rows,
    _fetch_term_inventory_sources,
    _format_term_inventory_answer,
    _resolve_exhaustive_inventory_term_type,
    _resolve_minimal_query_type,
    _should_boost_coverage_top_k,
)
from informity.llm.metadata_filters import MetadataFilter, build_where_clause_and_params  # noqa: E402
from informity.llm.model_adapter import get_profile, get_retrieval_top_k  # noqa: E402
from informity.llm.query_classifier import QueryClassification, classify_query  # noqa: E402
from informity.llm.rag_patterns import (  # noqa: E402
    evaluate_substantive_evidence,
    has_comparison_cue,
    has_explicit_title_reference,
    is_summary_style_request,
    normalize_query_text,
    should_prefer_title_alignment,
)
from informity.llm.rag_runtime.generation_closeout import build_source_references  # noqa: E402
from informity.llm.rag_runtime.structured_numeric import _derive_format_requirements  # noqa: E402
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation  # noqa: E402
from informity.llm.retrieval import (  # noqa: E402
    _COVERAGE_DIVERSITY_PRIMARY_FILE_CAP,
    _COVERAGE_DIVERSITY_SECONDARY_FILE_CAP,
    _apply_coverage_document_breadth_bias,
    _apply_reranker_score_threshold,
    _apply_strict_title_file_focus,
    _apply_substantive_section_bias,
    _apply_title_alignment_bias,
    _file_breadth_bonus,
    _filter_structural_chunks_when_possible,
    _resolve_rerank_min_score,
    _select_top_children,
    _resolve_within_file_location_key,
)
from informity.llm.term_dictionary import TermExpansion, expand_query_for_retrieval  # noqa: E402
from informity.llm.types import BlockType, QueryType  # noqa: E402
from informity.translate_policy import TRANSLATE_ENTITY_TYPE, TRANSLATE_PROVIDER  # noqa: E402
from informity.upload_policy import UPLOAD_ENTITY_TYPE, UPLOAD_PROVIDER  # noqa: E402
from informity.utils.file_utils import normalize_extension  # noqa: E402


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    summary: str
    data: dict[str, Any]


@dataclass
class RetrievalRoundResult:
    stage1: StageResult
    stage2: StageResult
    stage3: StageResult
    stage4: StageResult
    stage5: StageResult
    stage6: StageResult
    stage7: StageResult
    final_chunks: list[dict[str, Any]]
    final_child_chunks: list[dict[str, Any]]
    reranked_children: list[dict[str, Any]]
    filtered_child_chunks: list[dict[str, Any]]
    raw_candidates: list[dict[str, Any]]
    applied_filters: list[dict[str, Any]]
    where_clause: str
    where_params: list[Any]
    query_rewritten: bool
    rewrite_reason: str | None
    retrieval_query: str
    effective_query_type: QueryType
    top_k: int
    search_k: int
    rerank_enabled: bool
    rerank_min_score: float
    fts5_augmented_count: int
    summary_retry_triggered: bool
    comparison_retry_triggered: bool
    summary_retry_used: bool
    comparison_retry_used: bool
    control_child_chunks: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect retrieval quality without generating an answer.')
    parser.add_argument('--query', required=True, help='Query to run through retrieval.')
    parser.add_argument('--scope', default='indexed_corpus', help='Classifier scope kind.')
    parser.add_argument('--output-dir', type=Path, help='Optional output directory for the report.')
    return parser.parse_args()


def _sanitize_report_name(value: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._-')
    return sanitized or 'query'


def _hash_query(query: str, scope: str) -> str:
    payload = f'{scope}\n{normalize_query_text(query)}'.encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def _stage_status(*, has_error: bool, has_warn: bool) -> str:
    if has_error:
        return 'ERROR'
    if has_warn:
        return 'WARN'
    return 'OK'


def _print_stage(stage_number: int, stage: StageResult) -> None:
    print(f'Stage {stage_number} — {stage.name} [{stage.status}] {stage.summary}')


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, '__dict__') and not isinstance(value, (str, bytes, int, float, bool)):
        return {key: _serialize(item) for key, item in vars(value).items()}
    return value


def _distinct_file_count(chunks: list[dict[str, Any]]) -> int:
    file_ids: set[int] = set()
    for chunk in chunks:
        try:
            file_id = int(chunk.get('file_id'))
        except (TypeError, ValueError):
            continue
        file_ids.add(file_id)
    return len(file_ids)


def _chunk_file_names(chunks: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        name = str(chunk.get('filename') or '').strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _chunk_file_ids(chunks: list[dict[str, Any]]) -> list[int]:
    file_ids: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        try:
            file_id = int(chunk.get('file_id'))
        except (TypeError, ValueError):
            continue
        if file_id in seen:
            continue
        seen.add(file_id)
        file_ids.append(file_id)
    return file_ids


def _summarize_top_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for chunk in chunks[:5]:
        summary.append({
            'chunk_id': chunk.get('chunk_id'),
            'file_id': chunk.get('file_id'),
            'filename': chunk.get('filename'),
            'score': round(float(chunk.get('score') or 0.0), 4),
        })
    return summary


def _build_filters(
    *,
    classification: QueryClassification,
    file_ids: list[int] | None,
    effective_block_type_exclude: list[BlockType],
) -> tuple[list[MetadataFilter], list[dict[str, Any]], str, list[Any]]:
    filters: list[MetadataFilter] = []
    if classification.year_filter:
        filters.append(MetadataFilter(field='year', operator='EQ', value=classification.year_filter))
    if classification.category_filter:
        safe_category = ''.join(char for char in classification.category_filter if char.isalnum() or char in '_-')
        if safe_category:
            filters.append(MetadataFilter(field='category', operator='EQ', value=safe_category))
    if classification.file_type_filter:
        filters.append(MetadataFilter(field='extension', operator='EQ', value=normalize_extension(classification.file_type_filter)))
    if classification.filename_filter:
        normalized_filename = classification.filename_filter.strip()
        if normalized_filename:
            filters.append(MetadataFilter(field='filename', operator='LIKE', value=f'%{normalized_filename}%'))
    for excluded_name in classification.filename_exclude:
        normalized_excluded_name = str(excluded_name or '').strip()
        if normalized_excluded_name:
            filters.append(MetadataFilter(field='filename', operator='NE', value=normalized_excluded_name))
    if file_ids:
        normalized_file_ids = sorted({int(file_id) for file_id in file_ids if int(file_id) > 0})
        if normalized_file_ids:
            filters.append(MetadataFilter(field='file_id', operator='IN', value=normalized_file_ids))

    safe_block_type_filter = classification.block_type_filter if classification.block_type_filter in {BlockType.TABLE, BlockType.FORM, BlockType.NARRATIVE} else None
    if safe_block_type_filter is not None:
        filters.append(MetadataFilter(field='block_type', operator='EQ', value=safe_block_type_filter))
    for excluded_block_type in effective_block_type_exclude:
        filters.append(MetadataFilter(field='block_type', operator='NE', value=excluded_block_type))

    active_filters = list(filters)
    vector_filters = [metadata_filter for metadata_filter in active_filters if metadata_filter.field != 'block_type']
    where_clause, where_params = build_where_clause_and_params(vector_filters)
    if not file_ids:
        upload_exclusion_clause = (
            'file_id NOT IN ('
            'SELECT id FROM files WHERE (source_provider = ? AND entity_type = ?)'
            ' OR (source_provider = ? AND entity_type = ?)'
            ')'
        )
        upload_exclusion_params: list[int | str] = [
            UPLOAD_PROVIDER, UPLOAD_ENTITY_TYPE,
            TRANSLATE_PROVIDER, TRANSLATE_ENTITY_TYPE,
        ]
        if where_clause:
            where_clause = f'({where_clause}) AND {upload_exclusion_clause}'
            where_params = [*where_params, *upload_exclusion_params]
        else:
            where_clause = upload_exclusion_clause
            where_params = upload_exclusion_params

    applied_filters_for_trace = [
        {'field': metadata_filter.field, 'operator': metadata_filter.operator, 'value': metadata_filter.value}
        for metadata_filter in active_filters
    ]
    return active_filters, applied_filters_for_trace, where_clause, where_params


async def _fetch_chunk_rows(db: aiosqlite.Connection, chunk_ids: list[int]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    placeholders = ','.join('?' * len(chunk_ids))
    cursor = await db.execute(
        f"""
        SELECT c.id AS chunk_id, c.file_id, f.path AS file_path, f.filename, c.content AS chunk_text,
               c.page_number, c.start_page, c.end_page, c.section_path, c.block_type, c.parent_id,
               f.tags AS file_tags, f.ocr_used, f.page_count, f.tables_count, f.form_items_count,
               f.key_value_items_count, f.pictures_count
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


def _score_value(chunk: dict[str, Any]) -> float:
    try:
        return float(chunk.get('score') or 0.0)
    except (TypeError, ValueError):
        return 0.0


async def _execute_retrieval_round(
    *,
    db: aiosqlite.Connection,
    query: str,
    classification: QueryClassification,
    scope_kind: str,
    file_ids: list[int] | None = None,
    prefer_substantive_sections: bool,
    prefer_title_alignment: bool,
    strict_title_alignment: bool,
    title_alignment_query: str | None,
    enable_term_expansion: bool,
    prefer_within_file_diversity: bool,
    top_k: int,
) -> RetrievalRoundResult:
    profile = get_profile()
    effective_query_type = _resolve_minimal_query_type(classification)
    comparison_style_request = has_comparison_cue(query)
    active_filters, applied_filters_for_trace, where_clause, where_params = _build_filters(
        classification=classification,
        file_ids=file_ids,
        effective_block_type_exclude=[],
    )

    if enable_term_expansion:
        term_expansion = await expand_query_for_retrieval(db=db, query=query)
    else:
        term_expansion = TermExpansion(
            dictionary_version=0,
            embedding_query=query,
            fts_query=query,
        )
    query_for_embedding = term_expansion.embedding_query or query
    query_for_fts = term_expansion.fts_query or query
    query_vector = await asyncio.to_thread(embedder.embed_query, query_for_embedding)

    search_k = max(top_k * 2, int(getattr(profile, 'retrieval_top_k_candidates', 25)))
    raw_candidates = await asyncio.to_thread(
        vector_store.search_similar,
        query_vector,
        search_k,
        where_clause,
        where_params,
    )
    fts5_augmented_count = 0
    if settings.fts5_candidate_limit > 0:
        existing_ids = {candidate['chunk_id'] for candidate in raw_candidates}
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
            raw_candidates = raw_candidates + fts5_candidates

    child_rows = await _fetch_chunk_rows(db, [candidate['chunk_id'] for candidate in raw_candidates])
    chunk_id_to_row: dict[int, dict[str, Any]] = {}
    child_to_parent_map: dict[int, int] = {}
    for row in child_rows:
        chunk_id = int(row['chunk_id'])
        chunk_dict: dict[str, Any] = {
            'chunk_id': chunk_id,
            'file_id': row['file_id'],
            'file_path': row['file_path'] or '',
            'filename': row['filename'] or '',
            'chunk_text': row['chunk_text'] or '',
            'page_number': row.get('page_number'),
            'start_page': row.get('start_page'),
            'end_page': row.get('end_page'),
            'section_path': row.get('section_path'),
            'block_type': row.get('block_type'),
            'file_tags': row.get('file_tags'),
            'ocr_used': row.get('ocr_used'),
            'page_count': row.get('page_count'),
            'tables_count': row.get('tables_count'),
            'form_items_count': row.get('form_items_count'),
            'key_value_items_count': row.get('key_value_items_count'),
            'pictures_count': row.get('pictures_count'),
        }
        parent_id = row.get('parent_id')
        if parent_id:
            child_to_parent_map[chunk_id] = int(parent_id)
        chunk_id_to_row[chunk_id] = chunk_dict

    vector_score_map: dict[int, float] = {candidate['chunk_id']: float(candidate['score']) for candidate in raw_candidates if 'score' in candidate}
    child_chunks = [chunk_id_to_row[candidate_id] for candidate_id in [candidate['chunk_id'] for candidate in raw_candidates] if candidate_id in chunk_id_to_row]

    safe_block_type_filter = classification.block_type_filter if classification.block_type_filter in {BlockType.TABLE, BlockType.FORM, BlockType.NARRATIVE} else None
    safe_block_type_exclude: list[BlockType] = []
    filtered_child_chunks = child_chunks
    if safe_block_type_filter is not None:
        block_filtered = [chunk for chunk in filtered_child_chunks if chunk.get('block_type') == safe_block_type_filter]
        if block_filtered:
            filtered_child_chunks = block_filtered
    if safe_block_type_exclude:
        filtered_child_chunks = [chunk for chunk in filtered_child_chunks if chunk.get('block_type') not in safe_block_type_exclude]
    safe_section_filter = classification.section_filter.strip().casefold() if isinstance(classification.section_filter, str) and classification.section_filter.strip() else None
    if safe_section_filter is not None:
        section_filtered = [
            chunk for chunk in filtered_child_chunks
            if isinstance(chunk.get('section_path'), str) and safe_section_filter in chunk['section_path'].casefold()
        ]
        if section_filtered:
            filtered_child_chunks = section_filtered
    filtered_child_chunks = _filter_structural_chunks_when_possible(
        chunks=filtered_child_chunks,
        prefer_substantive_sections=prefer_substantive_sections,
        top_k=top_k,
    )

    rerank_enabled = settings.rag_rerank and (effective_query_type != QueryType.COVERAGE or settings.rag_rerank_coverage)
    if rerank_enabled:
        reranked_children = await asyncio.to_thread(reranker.rerank, query, filtered_child_chunks)
    else:
        reranked_children = [{**chunk, 'score': vector_score_map.get(chunk['chunk_id'], 0.0)} for chunk in filtered_child_chunks]

    child_chunk_metadata_by_id: dict[int, dict[str, Any]] = {}
    for chunk in filtered_child_chunks:
        try:
            chunk_id = int(chunk.get('chunk_id'))
        except (TypeError, ValueError):
            continue
        child_chunk_metadata_by_id[chunk_id] = chunk

    reranked_children = [
        {**child_chunk_metadata_by_id.get(int(chunk.get('chunk_id')), {}), **chunk}
        for chunk in reranked_children
    ]
    reranked_children = _apply_substantive_section_bias(
        chunks=reranked_children,
        prefer_substantive_sections=prefer_substantive_sections,
    )
    reranked_children = _apply_title_alignment_bias(
        chunks=reranked_children,
        query=title_alignment_query or query,
        prefer_title_alignment=prefer_title_alignment,
        strict_title_alignment=strict_title_alignment,
    )
    reranked_children = _apply_strict_title_file_focus(
        chunks=reranked_children,
        query=title_alignment_query or query,
        strict_title_alignment=strict_title_alignment,
    )
    reranked_children = _apply_coverage_document_breadth_bias(
        chunks=reranked_children,
        query_type=effective_query_type,
        prefer_within_file_diversity=prefer_within_file_diversity,
    )
    base_rerank_min_score = float(getattr(profile, 'rag_rerank_min_score', 0.0) or 0.0)
    rerank_min_score = _resolve_rerank_min_score(
        query_type=effective_query_type,
        prefer_within_file_diversity=prefer_within_file_diversity,
        base_min_score=base_rerank_min_score,
    )
    rerank_threshold_removed_count = 0
    if rerank_enabled and rerank_min_score > 0:
        filtered_reranked_children = _apply_reranker_score_threshold(
            chunks=reranked_children,
            min_score=rerank_min_score,
        )
        rerank_threshold_removed_count = max(len(reranked_children) - len(filtered_reranked_children), 0)
        reranked_children = filtered_reranked_children

    top_children = _select_top_children(
        reranked_children=reranked_children,
        top_k=top_k,
        query_type=effective_query_type,
        prefer_within_file_diversity=prefer_within_file_diversity,
    )
    control_children = _select_top_children(
        reranked_children=reranked_children,
        top_k=top_k,
        query_type=effective_query_type,
        prefer_within_file_diversity=False,
    )

    parent_ids: list[int] = []
    for child in top_children:
        child_id = int(child['chunk_id'])
        parent_id = child_to_parent_map.get(child_id)
        if parent_id:
            parent_ids.append(parent_id)
    parent_chunks = await get_chunks_by_parent_ids(db, parent_ids) if parent_ids else []
    parent_id_to_chunk = {int(parent['chunk_id']): parent for parent in parent_chunks}

    final_chunks: list[dict[str, Any]] = []
    seen_parent_ids: set[int] = set()
    for child in top_children:
        child_id = int(child['chunk_id'])
        parent_id = child_to_parent_map.get(child_id)
        if parent_id and parent_id in parent_id_to_chunk and parent_id not in seen_parent_ids:
            parent_chunk = {**parent_id_to_chunk[parent_id]}
            child_score = _score_value(child)
            parent_chunk['score'] = child_score
            parent_chunk['source_rank'] = len(final_chunks) + 1
            final_chunks.append(parent_chunk)
            seen_parent_ids.add(parent_id)
        elif not parent_id:
            final_chunks.append({**child, 'source_rank': len(final_chunks) + 1})
        else:
            final_chunks.append({**child, 'source_rank': len(final_chunks) + 1})

    final_file_names = _chunk_file_names(final_chunks)
    control_file_names = _chunk_file_names(control_children)
    final_file_ids = _chunk_file_ids(final_chunks)
    control_file_ids = _chunk_file_ids(control_children)
    added_files = [name for name in final_file_names if name not in control_file_names]
    removed_files = [name for name in control_file_names if name not in final_file_names]

    stage3 = StageResult(
        name='Vector search',
        status='OK' if raw_candidates else 'WARN',
        summary=f'top_k={search_k}; candidates={len(raw_candidates)}; distinct_files={_distinct_file_count(raw_candidates)}',
        data={
            'top_k': search_k,
            'raw_candidates_returned': len(raw_candidates),
            'distinct_file_count': _distinct_file_count(raw_candidates),
            'top_5_candidates': _summarize_top_chunks(raw_candidates),
            'fts5_augmented_count': fts5_augmented_count,
        },
    )
    stage4 = StageResult(
        name='Reranking',
        status=_stage_status(
            has_error=rerank_enabled and bool(False),
            has_warn=bool(rerank_threshold_removed_count),
        ),
        summary=(
            f'after_rerank={len(reranked_children)}; distinct_files={_distinct_file_count(reranked_children)}; '
            f'dropped={max(len(filtered_child_chunks) - len(reranked_children), 0)}'
        ),
        data={
            'rerank_enabled': rerank_enabled,
            'total_candidates_after_rerank': len(reranked_children),
            'distinct_file_count_after_rerank': _distinct_file_count(reranked_children),
            'top_5_reranked_candidates': _summarize_top_chunks(reranked_children),
            'dropped_candidates': max(len(filtered_child_chunks) - len(reranked_children), 0),
            'kept_candidates': len(reranked_children),
            'base_rerank_min_score': base_rerank_min_score,
            'rerank_min_score': rerank_min_score,
            'rerank_threshold_removed_count': rerank_threshold_removed_count,
            'structural_filter_removed_count': max(len(child_chunks) - len(filtered_child_chunks), 0),
            'filtered_child_count': len(filtered_child_chunks),
        },
    )
    stage5 = StageResult(
        name='Diversity / structure',
        status=_stage_status(
            has_error=False,
            has_warn=bool(added_files or removed_files or len(filtered_child_chunks) != len(control_children)),
        ),
        summary=(
            f'final={len(final_chunks)}; distinct_files={_distinct_file_count(final_chunks)}; '
            f'added={len(added_files)}; removed={len(removed_files)}'
        ),
        data={
            'final_candidate_count': len(final_chunks),
            'final_distinct_file_count': _distinct_file_count(final_chunks),
            'final_file_names': final_file_names,
            'control_candidate_count': len(control_children),
            'control_distinct_file_count': _distinct_file_count(control_children),
            'control_file_names': control_file_names,
            'files_added_by_diversity': added_files,
            'files_removed_by_diversity': removed_files,
            'control_file_ids': control_file_ids,
            'final_file_ids': final_file_ids,
            'structural_filter_removed_count': max(len(child_chunks) - len(filtered_child_chunks), 0),
            'file_breadth_bonus_preview': round(_file_breadth_bonus(reranked_children), 4) if reranked_children else 0.0,
        },
    )

    answerability_passed, answerability_score, answerability_threshold, min_chunks = _evaluate_minimal_answerability(
        final_chunks,
        query_type=effective_query_type,
    )
    stage6 = StageResult(
        name='Answerability gate',
        status='OK' if answerability_passed else 'WARN',
        summary=(
            f'passed={answerability_passed}; score={answerability_score:.4f}; '
            f'threshold={answerability_threshold}; min_chunks={min_chunks}'
        ),
        data={
            'passed': answerability_passed,
            'score': round(answerability_score, 4),
            'threshold': answerability_threshold,
            'min_chunks': min_chunks,
            'chunk_count': len(final_chunks),
        },
    )
    stage7 = StageResult(
        name='Citation verification',
        status='OK',
        summary='skipped in read-only diagnostic; no generated answer text',
        data={
            'applicable': False,
            'reason': 'no_generated_answer_text',
            'sources_entered_verification': 0,
            'sources_survived': 0,
            'sources_filtered': 0,
            'filtered_reasons': [],
            'potential_sources_for_generation': len(final_chunks),
        },
    )

    stage1 = StageResult(
        name='Query rewrite',
        status='OK',
        summary=f'original → {query}',
        data={
            'original_query': query,
            'rewritten_query': query,
            'rewritten': False,
            'rewrite_reason': None,
            'scope_kind': scope_kind,
        },
    )
    stage2 = StageResult(
        name='Filter resolution',
        status='OK',
        summary=f'filters={len(applied_filters_for_trace)}; scope={scope_kind}',
        data={
            'scope_kind': scope_kind,
            'applied_filters': applied_filters_for_trace,
            'where_clause': where_clause,
            'where_params': where_params,
            'term_expansion_enabled': enable_term_expansion,
            'query_type': effective_query_type,
        },
    )

    return RetrievalRoundResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        stage6=stage6,
        stage7=stage7,
        final_chunks=final_chunks,
        final_child_chunks=top_children,
        reranked_children=reranked_children,
        filtered_child_chunks=filtered_child_chunks,
        raw_candidates=raw_candidates,
        applied_filters=applied_filters_for_trace,
        where_clause=where_clause,
        where_params=where_params,
        query_rewritten=False,
        rewrite_reason=None,
        retrieval_query=query,
        effective_query_type=effective_query_type,
        top_k=top_k,
        search_k=search_k,
        rerank_enabled=rerank_enabled,
        rerank_min_score=rerank_min_score,
        fts5_augmented_count=fts5_augmented_count,
        summary_retry_triggered=False,
        comparison_retry_triggered=False,
        summary_retry_used=False,
        comparison_retry_used=False,
        control_child_chunks=control_children,
    )


async def _run_diagnostic(query: str, scope_kind: str, output_dir: Path | None) -> dict[str, Any]:
    resolved_query = str(query or '').strip()
    if not resolved_query:
        raise ValueError('query must not be empty')

    report_root = output_dir or (Path.home() / '.informity' / 'diagnostics')
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    query_hash = _hash_query(resolved_query, scope_kind)
    report_path = report_root / f'retrieval_diag_{query_hash}_{timestamp}.json'

    classification = classify_query(
        resolved_query,
        history=None,
        chat_mode='researcher',
        scope_kind=scope_kind,
    )
    profile = get_profile()
    effective_query_type = _resolve_minimal_query_type(classification)
    inventory_term_type = _resolve_exhaustive_inventory_term_type(resolved_query, classification)

    db = await get_connection()
    try:
        if inventory_term_type is not None:
            inventory_rows = await _fetch_term_inventory_rows(db=db, term_type=inventory_term_type)
            inventory_sources = await _fetch_term_inventory_sources(db=db, term_type=inventory_term_type)
            answer_text = _format_term_inventory_answer(term_type=inventory_term_type, rows=inventory_rows)
            report: dict[str, Any] = {
                'query': resolved_query,
                'scope_kind': scope_kind,
                'timestamp': timestamp,
                'query_hash': query_hash,
                'classification': _serialize(asdict(classification)),
                'profile': profile.to_display_dict(),
                'deterministic_inventory': {
                    'term_type': inventory_term_type,
                    'row_count': len(inventory_rows),
                    'source_count': len(inventory_sources),
                    'answer_preview': answer_text[:500],
                },
                'stages': {
                    'Query rewrite': {
                        'name': 'Query rewrite',
                        'status': 'OK',
                        'summary': 'deterministic inventory shortcut',
                        'data': {'original_query': resolved_query, 'rewritten_query': resolved_query, 'rewritten': False},
                    },
                    'Filter resolution': {
                        'name': 'Filter resolution',
                        'status': 'OK',
                        'summary': 'deterministic inventory shortcut',
                        'data': {'scope_kind': scope_kind, 'query_type': effective_query_type},
                    },
                    'Vector search': {
                        'name': 'Vector search',
                        'status': 'OK',
                        'summary': 'skipped: deterministic inventory shortcut',
                        'data': {},
                    },
                    'Reranking': {
                        'name': 'Reranking',
                        'status': 'OK',
                        'summary': 'skipped: deterministic inventory shortcut',
                        'data': {},
                    },
                    'Diversity / structure': {
                        'name': 'Diversity / structure',
                        'status': 'OK',
                        'summary': 'skipped: deterministic inventory shortcut',
                        'data': {},
                    },
                    'Answerability gate': {
                        'name': 'Answerability gate',
                        'status': 'OK',
                        'summary': 'skipped: deterministic inventory shortcut',
                        'data': {},
                    },
                    'Citation verification': {
                        'name': 'Citation verification',
                        'status': 'OK',
                        'summary': 'skipped in read-only diagnostic; no generated answer text',
                        'data': {'applicable': False, 'reason': 'no_generated_answer_text', 'potential_sources_for_generation': len(inventory_sources)},
                    },
                },
                'report_path': str(report_path),
            }
            report_path.write_text(json.dumps(_serialize(report), indent=2, ensure_ascii=False), encoding='utf-8')
            return report

        prefer_title_alignment = bool(
            classification.focus_resolved and classification.focus_prefer_title_alignment
        ) or should_prefer_title_alignment(question=resolved_query, classification=classification)
        if classification.focus_resolved:
            title_alignment_query = classification.focus_title_alignment_query or resolved_query
            strict_title_alignment = bool(classification.focus_strict_title_alignment)
            enable_term_expansion = not bool(classification.focus_disable_term_expansion)
        else:
            title_alignment_query = resolved_query
            strict_title_alignment = bool(has_explicit_title_reference(resolved_query) and not has_comparison_cue(resolved_query))
            enable_term_expansion = True

        effective_top_k = get_retrieval_top_k(effective_query_type)
        if _should_boost_coverage_top_k(resolved_query, classification):
            effective_top_k = min(60, effective_top_k + 8)

        initial_round = await _execute_retrieval_round(
            db=db,
            query=classification.retrieval_content_query or resolved_query,
            classification=classification,
            scope_kind=scope_kind,
            file_ids=None,
            prefer_substantive_sections=is_summary_style_request(resolved_query, classification),
            prefer_title_alignment=prefer_title_alignment,
            strict_title_alignment=strict_title_alignment,
            title_alignment_query=title_alignment_query,
            enable_term_expansion=enable_term_expansion,
            prefer_within_file_diversity=is_summary_style_request(resolved_query, classification) or has_comparison_cue(resolved_query),
            top_k=effective_top_k,
        )
        initial_final_chunks = initial_round.final_chunks
        summary_style_request = is_summary_style_request(resolved_query, classification)
        explicit_title_reference = bool(classification.focus_explicit_title_reference) if classification.focus_resolved else has_explicit_title_reference(resolved_query)
        comparison_style_request = has_comparison_cue(resolved_query)

        summary_retry_triggered = False
        summary_retry_used = False
        comparison_retry_triggered = False
        comparison_retry_used = False
        active_round = initial_round

        if summary_style_request and explicit_title_reference:
            evidence_profile = evaluate_substantive_evidence(initial_final_chunks)
            dominant_ratio = 0.0
            if initial_final_chunks:
                file_counts: dict[int, int] = {}
                for chunk in initial_final_chunks:
                    file_id = chunk.get('file_id')
                    if isinstance(file_id, int):
                        file_counts[file_id] = file_counts.get(file_id, 0) + 1
                dominant_ratio = max(file_counts.values()) / max(1, len(initial_final_chunks)) if file_counts else 0.0
            weak_summary_evidence = (
                float(evidence_profile.get('substantive_ratio') or 0.0) < 0.55
                and dominant_ratio >= 0.6
            )
            if weak_summary_evidence:
                summary_retry_triggered = True
                summary_retry_round = await _execute_retrieval_round(
                    db=db,
                    query=resolved_query,
                    classification=classification,
                    scope_kind=scope_kind,
                    file_ids=None,
                    prefer_substantive_sections=True,
                    prefer_title_alignment=True,
                    strict_title_alignment=True,
                    title_alignment_query=resolved_query,
                    enable_term_expansion=False,
                    prefer_within_file_diversity=True,
                    top_k=effective_top_k,
                )
                retry_evidence_profile = evaluate_substantive_evidence(summary_retry_round.final_chunks)
                if (
                    summary_retry_round.final_chunks
                    and float(retry_evidence_profile.get('substantive_ratio') or 0.0)
                    >= float(evidence_profile.get('substantive_ratio') or 0.0)
                ):
                    active_round = summary_retry_round
                    summary_retry_used = True
        if comparison_style_request and _distinct_file_count(active_round.final_chunks) < 2:
            comparison_retry_triggered = True
            comparison_retry_round = await _execute_retrieval_round(
                db=db,
                query=resolved_query,
                classification=classification,
                scope_kind=scope_kind,
                file_ids=None,
                prefer_substantive_sections=False,
                prefer_title_alignment=False,
                strict_title_alignment=False,
                title_alignment_query=None,
                enable_term_expansion=True,
                prefer_within_file_diversity=True,
                top_k=min(max(effective_top_k + 4, effective_top_k), 16),
            )
            if _distinct_file_count(comparison_retry_round.final_chunks) > _distinct_file_count(active_round.final_chunks):
                active_round = comparison_retry_round
                comparison_retry_used = True

        answerability_passed, answerability_score, answerability_threshold, min_chunks = _evaluate_minimal_answerability(
            active_round.final_chunks,
            query_type=active_round.effective_query_type,
        )
        final_stage6 = StageResult(
            name='Answerability gate',
            status='OK' if answerability_passed else 'WARN',
            summary=(
                f'passed={answerability_passed}; score={answerability_score:.4f}; '
                f'threshold={answerability_threshold}; min_chunks={min_chunks}'
            ),
            data={
                'passed': answerability_passed,
                'score': round(answerability_score, 4),
                'threshold': answerability_threshold,
                'min_chunks': min_chunks,
                'chunk_count': len(active_round.final_chunks),
            },
        )

        report = {
            'query': resolved_query,
            'scope_kind': scope_kind,
            'timestamp': timestamp,
            'query_hash': query_hash,
            'classification': _serialize(asdict(classification)),
            'profile': profile.to_display_dict(),
            'stages': {
                'Query rewrite': asdict(active_round.stage1),
                'Filter resolution': asdict(active_round.stage2),
                'Vector search': asdict(active_round.stage3),
                'Reranking': asdict(active_round.stage4),
                'Diversity / structure': asdict(active_round.stage5),
                'Answerability gate': asdict(final_stage6),
                'Citation verification': asdict(active_round.stage7),
            },
            'retry_analysis': {
                'summary_retry_triggered': summary_retry_triggered,
                'summary_retry_used': summary_retry_used,
                'comparison_retry_triggered': comparison_retry_triggered,
                'comparison_retry_used': comparison_retry_used,
            },
            'final_chunks': [
                {
                    'chunk_id': chunk.get('chunk_id'),
                    'file_id': chunk.get('file_id'),
                    'filename': chunk.get('filename'),
                    'score': round(float(chunk.get('score') or 0.0), 4),
                    'source_rank': chunk.get('source_rank'),
                }
                for chunk in active_round.final_chunks
            ],
            'report_path': str(report_path),
        }
        report_path.write_text(json.dumps(_serialize(report), indent=2, ensure_ascii=False), encoding='utf-8')
        return report
    finally:
        await db.close()


def main() -> int:
    args = parse_args()
    report = asyncio.run(_run_diagnostic(args.query, args.scope, args.output_dir))
    stages = report['stages']
    for idx, stage_name in enumerate(stages.keys(), 1):
        stage = StageResult(**stages[stage_name])
        _print_stage(idx, stage)
    print(f"Overall [OK] report written to {report['report_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
