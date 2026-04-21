# ==============================================================================
# Informity AI — Chat API Routes
# Endpoints for RAG-based chat: send messages (SSE streaming), list
# chats, and retrieve chat history.
# ==============================================================================

import asyncio
import contextlib
import os
import re
import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sse_starlette.sse import EventSourceResponse
from structlog.contextvars import get_contextvars

from informity import answer_sanitization
from informity.answer_sanitization import build_display_answer, sanitize_display_answer
from informity.api.chat_closeout import build_display_blocks, build_done_payload
from informity.api.chat_completion_policy import resolve_completion_and_action
from informity.api.context_scope_manager import (
    INDEXED_CORPUS_SCOPE_KIND,
    normalize_indexed_corpus_scope_key,
    resolve_retrieval_context_scope_key,
)
from informity.api.chat_continuation import (
    build_auto_continue_pass_prompt as _build_auto_continue_pass_prompt,
)
from informity.api.chat_continuation import (
    detect_structural_incomplete_reason as _detect_structural_incomplete_reason,
)
from informity.api.chat_continuation import (
    enforce_continuation_chat_binding as _enforce_continuation_chat_binding,
)
from informity.api.chat_continuation import (
    is_continuation_request as _is_continuation_request,
)
from informity.api.chat_continuation import (
    is_duplicate_continuation_pass as _is_duplicate_continuation_pass,
)
from informity.api.chat_continuation import (
    normalize_continuation_classification as _normalize_continuation_classification,
)
from informity.api.chat_continuation import (
    resolve_auto_continue_policy as _resolve_auto_continue_policy,
)
from informity.api.chat_continuation import (
    resolve_continuation_anchor_question as _resolve_continuation_anchor_question,
)
from informity.api.chat_continuation import (
    resolve_next_action as _resolve_next_action,
)
from informity.api.chat_orchestrator import ChatOrchestrator
from informity.api.chat_sources import merge_sources, serialize_sources
from informity.api.chat_sse import SSE_PHASE_ORDER, SseContractTracker, SseStatusEmitter
from informity.api.chat_stream_registry import CHAT_STREAM_REGISTRY
from informity.api.error_messages import to_client_error_message
from informity.api.schemas import (
    ChatPreferencesUpdateRequest,
    ChatRequest,
    ChatSourceReference,
    ChatStopRequest,
)
from informity.api.security import EndpointGuard
from informity.chat_trace import get_trace_writer
from informity.config import settings
from informity.db.models import ChatMessage, ChatUploadAttachment, ContinuationPassArtifact
from informity.db.sqlite import (
    append_chat_upload_reference_message,
    delete_chat,
    get_chat,
    get_chat_count,
    get_chat_message_by_id,
    get_chat_preferences,
    get_chat_upload_attachment_by_upload_id,
    get_chat_upload_attachments,
    get_chat_upload_size_bytes,
    get_chats,
    get_connection,
    get_db,
    get_file_by_id,
    get_file_by_path,
    get_files_by_ids,
    insert_chat_message,
    insert_chat_upload_attachment,
    insert_continuation_pass_artifact,
    insert_diagnostics_metrics,
    set_chat_title,
    update_chat_upload_attachment_state,
    upsert_chat_preferences,
)
from informity.diagnostics.observer import EvalMetrics, detect_issues, estimate_evidence_metrics
from informity.diagnostics.resource_snapshot import build_resource_delta, capture_resource_snapshot
from informity.indexer.pipeline import index_file, remove_file
from informity.llm.chat_mode import resolve_chat_mode
from informity.llm.classification_policy import classify_query_with_timing
from informity.llm.contract_gate import (
    build_contract_spec,
    build_repair_guidance,
    enforce_required_sections,
    validate_contract,
)
from informity.llm.rag import answer_question
from informity.llm.timeout_policy import is_terminal_timeout_reason, normalize_timeout_reason
from informity.llm.types import (
    ChatRole,
    CompletionMode,
    ContinuationResolutionReason,
    DiagnosticsQueryType,
    NextAction,
    StreamSignalTag,
    StructuralGapReason,
    TimeoutReason,
)
from informity.scanner.crawler import scanned_file_for_path
from informity.upload_policy import (
    MAX_UPLOAD_FILES_PER_CHAT,
    UPLOAD_ENTITY_TYPE,
    UPLOAD_PROVIDER,
    is_allowed_extension,
    is_allowed_mime,
    max_upload_file_size_bytes,
    max_upload_total_size_bytes,
    upload_root_dir,
)
from informity.utils.json_utils import serialize_api_response
from informity.utils.number_utils import safe_float, safe_int

# Trace logging constants
MAX_ANSWER_PREVIEW_LENGTH = 1500  # Maximum length of answer preview in trace logs
MAX_CHAT_MESSAGE_CHARS = 20000
_VALID_COMPLETION_MODES = {
    CompletionMode.COMPLETE,
    CompletionMode.PARTIAL,
    CompletionMode.SCOPED_COMPLETE,
    CompletionMode.STOPPED,
}
_PERSISTENCE_EXCEPTIONS = (aiosqlite.Error, ValueError, RuntimeError, OSError)
_STREAM_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, ConnectionError, aiosqlite.Error)
_STOP_FINALIZE_GRACE_SECONDS = 2.5
_STOP_FINALIZATION_TASKS: dict[str, asyncio.Task[None]] = {}
_CONTINUING_STATUS_MESSAGE = 'Continuing response...'
_ANSWER_STREAM_HEARTBEAT_SECONDS = 12.0
_ACTIVE_UPLOAD_STATES = {'uploading', 'indexing', 'ready'}
_UPLOAD_DELETE_RETRY_ATTEMPTS = 3
_SCOPE_SIGNAL_PATTERN = re.compile(r'(?i)\b(compare|vs|versus|only|just|between)\b')
_FILENAME_CANDIDATE_PATTERN = re.compile(
    r'(?i)\b([a-z0-9][a-z0-9_\-\(\)\[\]\.]{0,140}\.[a-z0-9]{1,10})\b'
)
_QUOTED_TEXT_PATTERN = re.compile(r'["\']([^"\']{1,180})["\']')
_OUT_OF_CORPUS_RESPONSE_PATTERN = re.compile(
    r'(?is)\b(?:provided|indexed|these)?\s*(?:documents?|records?|context)\b.{0,120}\b'
    r'(?:do\s+not|does\s+not|cannot|can\'t|not)\b.{0,120}\b'
    r'(?:contain|include|cover|mention|provide|have)\b'
)


def _normalize_diagnostics_query_type(value: object) -> str:
    normalized = str(value or '').strip().lower()
    try:
        return DiagnosticsQueryType(normalized).value
    except ValueError:
        return DiagnosticsQueryType.UNKNOWN.value


def _answer_signals_out_of_corpus(text: str) -> bool:
    return bool(_OUT_OF_CORPUS_RESPONSE_PATTERN.search(str(text or '')))


def _sanitize_upload_filename(filename: str) -> str:
    name = Path(str(filename or '')).name.strip()
    if not name:
        return 'upload.txt'
    return ''.join(ch for ch in name if ch.isprintable())[:255] or 'upload.txt'


def _upload_chat_dir(chat_id: str) -> Path:
    return upload_root_dir() / str(chat_id).strip()


def _upload_file_dir(chat_id: str, upload_id: str) -> Path:
    return _upload_chat_dir(chat_id) / str(upload_id).strip()


def _normalize_filename_token(value: str) -> str:
    token = ' '.join(str(value or '').strip().split())
    token = token.strip('.,;:()[]{}')
    token = re.sub(r'(?i)^(?:compare|vs|versus|and|with|between)\s+', '', token).strip()
    token = token.strip('.,;:()[]{}')
    return token


def _extract_filename_candidates(text: str) -> list[str]:
    message = str(text or '')
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _FILENAME_CANDIDATE_PATTERN.finditer(message):
        token = _normalize_filename_token(match.group(1))
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            candidates.append(token)
    for match in _QUOTED_TEXT_PATTERN.finditer(message):
        token = _normalize_filename_token(match.group(1))
        if '.' not in token:
            continue
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            candidates.append(token)
    return candidates


def _resolve_upload_scope_from_filename_candidates(
    *,
    candidates: list[str],
    attachments: list[ChatUploadAttachment],
) -> tuple[list[ChatUploadAttachment], str | None]:
    if not candidates:
        return [], None
    selected_by_upload_id: dict[str, ChatUploadAttachment] = {}
    for raw_candidate in candidates:
        candidate = _normalize_filename_token(raw_candidate).lower()
        exact_matches = [
            attachment
            for attachment in attachments
            if _normalize_filename_token(attachment.filename_at_upload).lower() == candidate
        ]
        if len(exact_matches) == 1:
            selected_by_upload_id[exact_matches[0].upload_id] = exact_matches[0]
            continue
        if len(exact_matches) > 1:
            return [], f'Ambiguous upload reference "{raw_candidate}". Select files explicitly in the attachment pills.'
        partial_matches = [
            attachment
            for attachment in attachments
            if candidate in _normalize_filename_token(attachment.filename_at_upload).lower()
        ]
        if len(partial_matches) == 1:
            selected_by_upload_id[partial_matches[0].upload_id] = partial_matches[0]
            continue
        if len(partial_matches) > 1:
            return [], f'Ambiguous upload reference "{raw_candidate}". Select files explicitly in the attachment pills.'
        return [], f'No uploaded file matched "{raw_candidate}".'
    return list(selected_by_upload_id.values()), None


_RETRIEVAL_SCOPE_ASSISTANT = 'assistant_mode'
_RETRIEVAL_SCOPE_INDEXED_CORPUS = 'indexed_corpus'
_RETRIEVAL_SCOPE_INDEXED_FILES = 'indexed_files'
_RETRIEVAL_SCOPE_CHAT_UPLOADS = 'chat_uploads'


def _build_retrieval_scope(
    *,
    chat_mode: str,
    scoped_file_ids: list[int] | None,
    upload_attachments: list[ChatUploadAttachment] | None = None,
    upload_attachments_all: list[ChatUploadAttachment] | None = None,
    selected_upload_ids: list[str] | None = None,
) -> tuple[str, str]:
    if chat_mode != 'researcher':
        return _RETRIEVAL_SCOPE_ASSISTANT, _RETRIEVAL_SCOPE_ASSISTANT
    active_uploads = [
        attachment
        for attachment in (upload_attachments or [])
        if attachment.state in _ACTIVE_UPLOAD_STATES and str(attachment.upload_id or '').strip()
    ]
    if active_uploads:
        # Reconstruct active-count transitions from upload/remove timestamps so
        # we can keep one stable key for the current contiguous upload session.
        events: list[tuple[str, int, str]] = []
        for attachment in (upload_attachments_all or upload_attachments or []):
            upload_id = str(attachment.upload_id or '').strip()
            if not upload_id:
                continue
            if attachment.uploaded_at is not None:
                events.append((attachment.uploaded_at.isoformat(), 1, upload_id))  # upload (+1)
            if attachment.removed_at is not None:
                events.append((attachment.removed_at.isoformat(), 0, upload_id))  # remove (-1), process first on tie
        events.sort(key=lambda item: (item[0], item[1], item[2]))

        active_count = 0
        session_anchor_upload_id: str | None = None
        for _, event_type, upload_id in events:
            if event_type == 0:
                active_count = max(0, active_count - 1)
                continue
            if active_count == 0:
                session_anchor_upload_id = upload_id
            active_count += 1
        if not session_anchor_upload_id:
            # Fallback for malformed/legacy rows missing uploaded_at.
            fallback_sorted = sorted(
                active_uploads,
                key=lambda item: (
                    item.uploaded_at.isoformat() if item.uploaded_at is not None else '',
                    str(item.upload_id or ''),
                ),
            )
            session_anchor_upload_id = str(fallback_sorted[0].upload_id or '').strip()

        scope_key = f'{_RETRIEVAL_SCOPE_CHAT_UPLOADS}:{session_anchor_upload_id}'
        active_upload_ids = {
            str(item.upload_id or '').strip()
            for item in active_uploads
            if str(item.upload_id or '').strip()
        }
        normalized_selected_ids = sorted({
            str(upload_id).strip()
            for upload_id in (selected_upload_ids or [])
            if str(upload_id).strip() in active_upload_ids
        })
        if normalized_selected_ids and set(normalized_selected_ids) != active_upload_ids:
            scope_key = f'{scope_key}|sel:{",".join(normalized_selected_ids)}'
        return _RETRIEVAL_SCOPE_CHAT_UPLOADS, scope_key
    normalized_file_ids = sorted({int(file_id) for file_id in (scoped_file_ids or []) if int(file_id) > 0})
    if normalized_file_ids:
        return _RETRIEVAL_SCOPE_INDEXED_FILES, ','.join(str(file_id) for file_id in normalized_file_ids)
    return _RETRIEVAL_SCOPE_INDEXED_CORPUS, _RETRIEVAL_SCOPE_INDEXED_CORPUS


def _filter_history_for_scope(
    *,
    history: list[ChatMessage],
    chat_mode: str,
    retrieval_scope_kind: str,
    retrieval_scope_key: str,
) -> list[ChatMessage]:
    if chat_mode != 'researcher':
        return list(history)
    target_scope_key = str(retrieval_scope_key or '').strip()
    target_scope_key_normalized = (
        normalize_indexed_corpus_scope_key(target_scope_key)
        if retrieval_scope_kind == INDEXED_CORPUS_SCOPE_KIND
        else target_scope_key
    )
    include_legacy_indexed_rows = (
        retrieval_scope_kind == INDEXED_CORPUS_SCOPE_KIND
        and target_scope_key_normalized == f'{INDEXED_CORPUS_SCOPE_KIND}|g:0'
    )

    scoped: list[ChatMessage] = []
    for message in history:
        message_chat_mode = str(message.chat_mode or '').strip()
        if message_chat_mode and message_chat_mode != chat_mode:
            continue
        message_scope_kind = str(message.retrieval_scope_kind or '').strip()
        message_scope_key = str(message.retrieval_scope_key or '').strip()
        if message_scope_kind and message_scope_key:
            message_scope_key_normalized = (
                normalize_indexed_corpus_scope_key(message_scope_key)
                if message_scope_kind == INDEXED_CORPUS_SCOPE_KIND
                else message_scope_key
            )
            if (
                message_scope_kind == retrieval_scope_kind
                and message_scope_key_normalized == target_scope_key_normalized
            ):
                scoped.append(message)
            continue
        # Legacy pre-scope rows remain available only for indexed corpus turns.
        if include_legacy_indexed_rows:
            scoped.append(message)
    return scoped


async def _sweep_chat_upload_orphans(
    *,
    db: aiosqlite.Connection,
    chat_id: str,
) -> dict[str, int]:
    removed_orphan_dirs = 0
    removed_deleted_dirs = 0
    repaired_failed_states = 0
    removed_empty_chat_dirs = 0
    attachments = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=True)
    attachments_by_id = {str(item.upload_id): item for item in attachments}
    chat_dir = _upload_chat_dir(chat_id)
    if chat_dir.exists():
        for child in chat_dir.iterdir():
            if not child.is_dir():
                continue
            upload_id = str(child.name).strip()
            attachment = attachments_by_id.get(upload_id)
            if attachment is None:
                try:
                    shutil.rmtree(child)
                    removed_orphan_dirs += 1
                except (OSError, RuntimeError) as exc:
                    log.warning(
                        'chat_upload_orphan_dir_remove_failed',
                        chat_id=chat_id,
                        upload_id=upload_id,
                        error=str(exc),
                    )
                continue
            if attachment.state == 'deleted':
                try:
                    shutil.rmtree(child)
                    removed_deleted_dirs += 1
                except (OSError, RuntimeError) as exc:
                    log.warning(
                        'chat_upload_deleted_dir_remove_failed',
                        chat_id=chat_id,
                        upload_id=upload_id,
                        error=str(exc),
                    )
    for attachment in attachments:
        if attachment.state not in {'uploading', 'indexing'}:
            continue
        attachment_dir = _upload_file_dir(chat_id, attachment.upload_id)
        if attachment_dir.exists():
            continue
        await update_chat_upload_attachment_state(
            db,
            upload_id=attachment.upload_id,
            chat_id=chat_id,
            state='failed',
        )
        repaired_failed_states += 1
    if chat_dir.exists():
        try:
            if not any(chat_dir.iterdir()):
                chat_dir.rmdir()
                removed_empty_chat_dirs += 1
        except (OSError, RuntimeError):
            # Best-effort cleanup only; stale empty dirs are harmless.
            pass
    return {
        'removed_orphan_dirs': removed_orphan_dirs,
        'removed_deleted_dirs': removed_deleted_dirs,
        'repaired_failed_states': repaired_failed_states,
        'removed_empty_chat_dirs': removed_empty_chat_dirs,
    }


# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['chat'])
CHAT_GUARD = EndpointGuard(
    name='chat',
    max_in_flight=1,
    max_requests_per_window=30,
    window_seconds=60,
)


class UserStopRequestedError(Exception):
    """Raised when the user explicitly requests to stop an in-flight stream."""


async def _finalize_stopped_stream_if_active(
    *,
    stream_id: str,
    chat_id: str | None,
    request_id: str | None,
) -> None:
    try:
        await asyncio.sleep(_STOP_FINALIZE_GRACE_SECONDS)
        if not await CHAT_STREAM_REGISTRY.has_stream(stream_id):
            return
        message_persisted = False
        if chat_id:
            try:
                persist_db = await get_connection()
                try:
                    history = await get_chat(persist_db, chat_id)
                    latest_assistant = next((m for m in reversed(history) if m.role == ChatRole.ASSISTANT), None)
                    has_terminal_assistant = bool(
                        latest_assistant
                        and (
                            bool(latest_assistant.stopped_by_user)
                            or latest_assistant.completion_mode in {CompletionMode.STOPPED, CompletionMode.PARTIAL}
                        )
                    )
                    if not has_terminal_assistant:
                        await insert_chat_message(
                            persist_db,
                            ChatMessage(
                                chat_id=chat_id,
                                role='assistant',
                                content='',
                                sources=[],
                                model_filename=settings.llm_model_filename,
                                completion_mode=CompletionMode.STOPPED,
                                stopped_by_user=True,
                                has_remaining_scope=True,
                                next_action=NextAction.REGENERATE,
                                next_action_reason='stopped',
                                chat_mode=latest_assistant.chat_mode if latest_assistant is not None else None,
                                retrieval_scope_kind=(
                                    latest_assistant.retrieval_scope_kind if latest_assistant is not None else None
                                ),
                                retrieval_scope_key=(
                                    latest_assistant.retrieval_scope_key if latest_assistant is not None else None
                                ),
                                is_internal=False,
                            ),
                        )
                        message_persisted = True
                finally:
                    await persist_db.close()
            except _PERSISTENCE_EXCEPTIONS as persist_err:
                log.warning(
                    'chat_stop_forced_finalize_persist_failed',
                    chat_id=chat_id,
                    stream_id=stream_id,
                    request_id=request_id,
                    error=str(persist_err),
                )
        removed = await CHAT_STREAM_REGISTRY.unregister(stream_id)
        if not removed:
            return
        log.info(
            'chat_response_cancelled',
            chat_id=chat_id,
            stream_id=stream_id,
            request_id=request_id,
            cancellation_reason='user_stop_forced_finalize',
            stopped_by_user=True,
            tokens_streamed=0,
            generation_seconds=None,
            message_persisted=message_persisted,
        )
        log.info(
            'chat_stream_unregistered',
            chat_id=chat_id,
            stream_id=stream_id,
            request_id=request_id,
            terminal_state='stopped_forced',
        )
    finally:
        _STOP_FINALIZATION_TASKS.pop(stream_id, None)


async def _persist_terminal_assistant_message(
    *,
    chat_id: str,
    content: str,
    sources: list[dict[str, object]],
    generation_seconds: float,
    completion_mode: CompletionMode,
    stopped_by_user: bool,
    has_remaining_scope: bool,
    next_action: NextAction,
    next_action_reason: str | None,
    chat_mode: str,
    retrieval_scope_kind: str | None = None,
    retrieval_scope_key: str | None = None,
) -> tuple[ChatMessage | None, bool]:
    """
    Persist terminal assistant messages (stopped/cancelled/partial) via one path.
    """
    assistant_message = ChatMessage(
        chat_id=chat_id,
        role='assistant',
        content=content,
        sources=sources,
        model_filename=settings.llm_model_filename,
        generation_seconds=generation_seconds,
        completion_mode=completion_mode,
        stopped_by_user=stopped_by_user,
        has_remaining_scope=has_remaining_scope,
        next_action=next_action,
        next_action_reason=next_action_reason,
        chat_mode=chat_mode,
        retrieval_scope_kind=retrieval_scope_kind,
        retrieval_scope_key=retrieval_scope_key,
        is_internal=False,
    )
    try:
        persist_db = await get_connection()
        try:
            assistant_message = await insert_chat_message(persist_db, assistant_message)
            return assistant_message, assistant_message.id is not None
        finally:
            await persist_db.close()
    except _PERSISTENCE_EXCEPTIONS as persist_err:
        log.warning('chat_terminal_persist_failed', chat_id=chat_id, error=str(persist_err))
        return None, False


@router.get('/api/chat/chats/{chat_id}/uploads')
async def list_chat_uploads(
    chat_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    sweep_summary = await _sweep_chat_upload_orphans(db=db, chat_id=chat_id)
    attachments = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=False)
    return {
        'chat_id': chat_id,
        'sweep_summary': sweep_summary,
        'attachments': [attachment.model_dump(mode='json') for attachment in attachments],
    }


@router.post('/api/chat/uploads')
async def upload_chat_file(
    chat_id: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    resolved_chat_id = str(chat_id or '').strip() or str(uuid.uuid4())
    filename = _sanitize_upload_filename(file.filename or '')
    if not is_allowed_extension(filename):
        raise HTTPException(status_code=400, detail='Unsupported file type for chat upload.')
    if not is_allowed_mime(file.content_type):
        raise HTTPException(status_code=400, detail='Unsupported upload MIME type.')

    existing_attachments = await get_chat_upload_attachments(db, chat_id=resolved_chat_id, include_deleted=False)
    active_attachments = [a for a in existing_attachments if a.state in {'uploading', 'indexing', 'ready'}]
    if len(active_attachments) >= MAX_UPLOAD_FILES_PER_CHAT:
        raise HTTPException(
            status_code=413,
            detail=f'Maximum uploads per chat reached ({MAX_UPLOAD_FILES_PER_CHAT}).',
        )

    raw_bytes = await file.read()
    size_bytes = len(raw_bytes)
    if size_bytes <= 0:
        raise HTTPException(status_code=400, detail='Uploaded file is empty.')
    if size_bytes > max_upload_file_size_bytes():
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds max upload size ({max_upload_file_size_bytes() // (1024 * 1024)} MB).',
        )
    current_chat_upload_size = await get_chat_upload_size_bytes(db, chat_id=resolved_chat_id)
    if current_chat_upload_size + size_bytes > max_upload_total_size_bytes():
        raise HTTPException(
            status_code=413,
            detail=f'Chat upload total size limit exceeded ({max_upload_total_size_bytes() // (1024 * 1024)} MB).',
        )

    upload_id = str(uuid.uuid4())
    attachment = await insert_chat_upload_attachment(
        db,
        upload_id=upload_id,
        chat_id=resolved_chat_id,
        filename_at_upload=filename,
        size_bytes=size_bytes,
        state='uploading',
    )
    file_dir = _upload_file_dir(resolved_chat_id, upload_id)
    file_path = file_dir / filename
    try:
        os.makedirs(file_dir, exist_ok=True)
        file_path.write_bytes(raw_bytes)
        await update_chat_upload_attachment_state(
            db,
            upload_id=upload_id,
            chat_id=resolved_chat_id,
            state='indexing',
        )
        scanned = scanned_file_for_path(file_path)
        if scanned is None:
            raise HTTPException(status_code=422, detail='Unable to process uploaded file for indexing.')
        index_result = await index_file(
            db,
            scanned,
            source_provider=UPLOAD_PROVIDER,
            entity_type=UPLOAD_ENTITY_TYPE,
        )
        if not index_result.success:
            await update_chat_upload_attachment_state(
                db,
                upload_id=upload_id,
                chat_id=resolved_chat_id,
                state='failed',
            )
            raise HTTPException(status_code=422, detail=f'Failed to index upload: {index_result.error or "unknown error"}')
        file_record = await get_file_by_path(db, str(file_path))
        if file_record is None or file_record.id is None:
            await update_chat_upload_attachment_state(
                db,
                upload_id=upload_id,
                chat_id=resolved_chat_id,
                state='failed',
            )
            raise HTTPException(status_code=500, detail='Upload indexed but file record was not found.')
        await update_chat_upload_attachment_state(
            db,
            upload_id=upload_id,
            chat_id=resolved_chat_id,
            state='ready',
            file_id=file_record.id,
            content_hash=file_record.content_hash,
        )
        latest_attachment = await get_chat_upload_attachment_by_upload_id(
            db,
            upload_id=upload_id,
            chat_id=resolved_chat_id,
        )
        return {
            'chat_id': resolved_chat_id,
            'upload_id': upload_id,
            'attachment': (latest_attachment.model_dump(mode='json') if latest_attachment else attachment.model_dump(mode='json')),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        await update_chat_upload_attachment_state(
            db,
            upload_id=upload_id,
            chat_id=resolved_chat_id,
            state='failed',
        )
        log.error(
            'chat_upload_failed',
            chat_id=resolved_chat_id,
            upload_id=upload_id,
            filename=filename,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail='Upload failed during processing.') from exc


async def _delete_chat_upload_artifacts(
    *,
    db: aiosqlite.Connection,
    attachment: ChatUploadAttachment,
) -> bool:
    upload_id = str(attachment.upload_id)
    chat_id = str(attachment.chat_id)
    file_id = attachment.file_id
    await update_chat_upload_attachment_state(
        db,
        upload_id=upload_id,
        chat_id=chat_id,
        state='deleting',
    )

    deletion_succeeded = True
    if file_id is not None:
        file_record = await get_file_by_id(db, int(file_id))
        if file_record is not None:
            removed = False
            for attempt in range(1, _UPLOAD_DELETE_RETRY_ATTEMPTS + 1):
                removed = await remove_file(db, file_record)
                if removed:
                    break
                await asyncio.sleep(0.05 * attempt)
            if not removed:
                log.warning('chat_upload_file_remove_failed', chat_id=chat_id, upload_id=upload_id, file_id=file_id)
                deletion_succeeded = False

    file_dir = _upload_file_dir(chat_id, upload_id)
    if file_dir.exists():
        deleted_from_disk = False
        for attempt in range(1, _UPLOAD_DELETE_RETRY_ATTEMPTS + 1):
            try:
                shutil.rmtree(file_dir)
                deleted_from_disk = True
                break
            except (OSError, RuntimeError) as exc:
                if attempt >= _UPLOAD_DELETE_RETRY_ATTEMPTS:
                    log.warning('chat_upload_storage_remove_failed', chat_id=chat_id, upload_id=upload_id, error=str(exc))
                    break
                await asyncio.sleep(0.05 * attempt)
        if not deleted_from_disk and file_dir.exists():
            log.warning('chat_upload_storage_delete_incomplete', chat_id=chat_id, upload_id=upload_id)
            deletion_succeeded = False

    await update_chat_upload_attachment_state(
        db,
        upload_id=upload_id,
        chat_id=chat_id,
        state='deleted' if deletion_succeeded else 'failed',
        removed_at=datetime.now(UTC) if deletion_succeeded else None,
    )
    return deletion_succeeded


@router.delete('/api/chat/uploads/{upload_id}')
async def delete_chat_upload(
    upload_id: str,
    chat_id: str = Query(..., min_length=1),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    attachment = await get_chat_upload_attachment_by_upload_id(
        db,
        upload_id=upload_id,
        chat_id=chat_id,
    )
    if attachment is None:
        raise HTTPException(status_code=404, detail='Upload not found.')

    if attachment.state == 'deleted':
        return {'chat_id': chat_id, 'upload_id': upload_id, 'deleted': True, 'fallback_to_scanned_documents': False}
    deletion_succeeded = await _delete_chat_upload_artifacts(db=db, attachment=attachment)
    if not deletion_succeeded:
        raise HTTPException(
            status_code=500,
            detail='Failed to fully delete uploaded file artifacts. Please retry.',
        )
    sweep_summary = await _sweep_chat_upload_orphans(db=db, chat_id=chat_id)
    remaining = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=False)
    active_remaining = [item for item in remaining if item.state in {'uploading', 'indexing', 'ready'}]
    fallback_to_scanned_documents = len(active_remaining) == 0
    return {
        'chat_id': chat_id,
        'upload_id': upload_id,
        'deleted': True,
        'fallback_to_scanned_documents': fallback_to_scanned_documents,
        'toast_message': (
            'No uploaded files. Using your scanned documents.'
            if fallback_to_scanned_documents
            else None
        ),
        'sweep_summary': sweep_summary,
    }



# ==============================================================================
# POST /api/chat — send a message and stream the response via SSE
# ==============================================================================

@router.post('/api/chat')
async def chat(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> EventSourceResponse:
    # Validate the incoming message
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail='Message cannot be empty')
    if len(message_text) > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f'Message too large (max {MAX_CHAT_MESSAGE_CHARS} characters).',
        )
    requested_scoped_file_ids: list[int] | None = None
    if request.scoped_file_ids is not None:
        candidate_file_ids = [int(file_id) for file_id in request.scoped_file_ids]
        files_by_id = await get_files_by_ids(db, candidate_file_ids)
        missing_file_ids = [file_id for file_id in candidate_file_ids if file_id not in files_by_id]
        if missing_file_ids:
            missing_ids_text = ', '.join(str(file_id) for file_id in missing_file_ids)
            raise HTTPException(status_code=404, detail=f'Files not found in index: {missing_ids_text}.')
        requested_scoped_file_ids = candidate_file_ids
    requested_scoped_upload_ids: list[str] | None = None
    if request.scoped_upload_ids is not None:
        requested_scoped_upload_ids = [
            str(upload_id).strip()
            for upload_id in request.scoped_upload_ids
            if str(upload_id).strip()
        ]
    await CHAT_GUARD.check_rate_limit()
    requested_run_id = str(request.run_id or '').strip() or None
    resolved_chat_mode = resolve_chat_mode(request.mode)
    _enforce_continuation_chat_binding(question=message_text, chat_id=request.chat_id)

    # Resolve chat ID — create a new one if not provided
    chat_id = request.chat_id or str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    client_request_id = str(request.request_id or '').strip()
    request_id = (
        client_request_id
        or str(get_contextvars().get('request_id') or '').strip()
        or str(uuid.uuid4())
    )
    artifact_request_id = request_id

    upload_attachments = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=False)
    await _sweep_chat_upload_orphans(db=db, chat_id=chat_id)
    upload_attachments = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=False)
    upload_attachments_all = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=True)
    uploads_by_id = {str(item.upload_id): item for item in upload_attachments}
    upload_ready_file_ids = sorted({
        int(item.file_id)
        for item in upload_attachments
        if item.state == 'ready' and item.file_id is not None
    })
    upload_indexing_ids = [
        item.upload_id
        for item in upload_attachments
        if item.state in {'uploading', 'indexing'}
    ]
    upload_active_ids = [
        item.upload_id
        for item in upload_attachments
        if item.state in _ACTIVE_UPLOAD_STATES
    ]
    if requested_scoped_file_ids is not None and requested_scoped_upload_ids is not None:
        raise HTTPException(
            status_code=409,
            detail='Provide either scoped_file_ids or scoped_upload_ids, not both.',
        )
    if resolved_chat_mode != 'researcher' and (upload_active_ids or requested_scoped_upload_ids):
        raise HTTPException(
            status_code=409,
            detail='Uploaded files are available only in Researcher mode.',
        )
    if requested_scoped_file_ids is not None and upload_active_ids:
        raise HTTPException(
            status_code=409,
            detail='Mixed library scope and chat uploads are not supported in one turn.',
        )
    scoped_file_ids: list[int] | None = requested_scoped_file_ids
    upload_scope_omitted_ids: list[str] = []
    upload_scope_selected_ids: list[str] = []
    upload_scope_resolution_mode = 'default'
    message_filename_candidates = _extract_filename_candidates(message_text)
    message_scope_signal = bool(_SCOPE_SIGNAL_PATTERN.search(message_text))
    if requested_scoped_upload_ids is not None:
        upload_scope_resolution_mode = 'explicit_upload_ids'
        if not upload_attachments:
            raise HTTPException(status_code=404, detail='No uploaded files found for this chat.')
        selected_uploads: list[ChatUploadAttachment] = []
        missing_upload_ids: list[str] = []
        for upload_id in requested_scoped_upload_ids:
            attachment = uploads_by_id.get(upload_id)
            if attachment is None or attachment.state not in _ACTIVE_UPLOAD_STATES:
                missing_upload_ids.append(upload_id)
                continue
            selected_uploads.append(attachment)
        if missing_upload_ids:
            raise HTTPException(
                status_code=404,
                detail=f'Upload not found or inactive: {", ".join(missing_upload_ids)}.',
            )
        indexing_selected = [item.upload_id for item in selected_uploads if item.state in {'uploading', 'indexing'}]
        if indexing_selected:
            raise HTTPException(
                status_code=409,
                detail='Selected uploaded file is still indexing. Please retry in a moment.',
            )
        selected_ready_file_ids = sorted({
            int(item.file_id)
            for item in selected_uploads
            if item.state == 'ready' and item.file_id is not None
        })
        if not selected_ready_file_ids:
            raise HTTPException(
                status_code=409,
                detail='Selected uploaded files are not ready yet. Please retry in a moment.',
            )
        scoped_file_ids = selected_ready_file_ids
        upload_scope_selected_ids = [item.upload_id for item in selected_uploads]
    elif scoped_file_ids is None and upload_active_ids:
        resolved_selected_uploads: list[ChatUploadAttachment] = []
        if message_filename_candidates or message_scope_signal:
            resolved_selected_uploads, resolution_error = _resolve_upload_scope_from_filename_candidates(
                candidates=message_filename_candidates,
                attachments=[item for item in upload_attachments if item.state in _ACTIVE_UPLOAD_STATES],
            )
            if resolution_error:
                raise HTTPException(status_code=409, detail=resolution_error)
            if resolved_selected_uploads:
                upload_scope_resolution_mode = 'nlp_filename_scope'
                selected_indexing = [
                    item.upload_id for item in resolved_selected_uploads if item.state in {'uploading', 'indexing'}
                ]
                if selected_indexing:
                    raise HTTPException(
                        status_code=409,
                        detail='Selected uploaded file is still indexing. Please retry in a moment.',
                    )
                selected_ready_ids = sorted({
                    int(item.file_id)
                    for item in resolved_selected_uploads
                    if item.file_id is not None and item.state == 'ready'
                })
                if not selected_ready_ids:
                    raise HTTPException(
                        status_code=409,
                        detail='Selected uploaded files are not ready yet. Please retry in a moment.',
                    )
                scoped_file_ids = selected_ready_ids
                upload_scope_selected_ids = [item.upload_id for item in resolved_selected_uploads]
            elif message_scope_signal and not message_filename_candidates:
                upload_scope_resolution_mode = 'nlp_scope_all_uploads'
        if upload_ready_file_ids:
            if not scoped_file_ids:
                scoped_file_ids = upload_ready_file_ids
                upload_scope_omitted_ids = list(upload_indexing_ids)
                upload_scope_selected_ids = list(upload_active_ids)
                upload_scope_resolution_mode = 'default_all_uploads'
        else:
            raise HTTPException(
                status_code=409,
                detail='Uploaded files are still indexing. Please retry in a moment.',
            )
    retrieval_scope_kind, retrieval_scope_key = _build_retrieval_scope(
        chat_mode=resolved_chat_mode,
        scoped_file_ids=scoped_file_ids,
        upload_attachments=upload_attachments,
        upload_attachments_all=upload_attachments_all,
        selected_upload_ids=upload_scope_selected_ids,
    )
    full_history = await get_chat(db, chat_id)
    retrieval_scope_key, context_scope_resolution = resolve_retrieval_context_scope_key(
        chat_mode=resolved_chat_mode,
        retrieval_scope_kind=retrieval_scope_kind,
        retrieval_scope_key=retrieval_scope_key,
        message_text=message_text,
        history=full_history,
    )
    # Fetch chat history (excluding the current message we're about to add)
    history = _filter_history_for_scope(
        history=full_history,
        chat_mode=resolved_chat_mode,
        retrieval_scope_kind=retrieval_scope_kind,
        retrieval_scope_key=retrieval_scope_key,
    )
    existing_chat_preferences = await get_chat_preferences(db, chat_id)
    resolved_chat_web_search_enabled = (
        bool(existing_chat_preferences.get('chat_web_search_enabled'))
        if request.chat_web_search_enabled is None
        else bool(request.chat_web_search_enabled)
    )
    resolved_chat_web_search_privacy_override = (
        bool(existing_chat_preferences.get('chat_web_search_privacy_override'))
        if request.chat_web_search_privacy_override is None
        else bool(request.chat_web_search_privacy_override)
    )
    user_message_is_internal = _is_continuation_request(message_text)

    # Persist the user message
    user_message = ChatMessage(
        chat_id = chat_id,
        role    = 'user',
        content = message_text,
        chat_mode = resolved_chat_mode,
        retrieval_scope_kind = retrieval_scope_kind,
        retrieval_scope_key = retrieval_scope_key,
        model_filename = settings.llm_model_filename,
        is_internal = user_message_is_internal,
    )
    await insert_chat_message(db, user_message)
    await upsert_chat_preferences(
        db,
        chat_id,
        chat_web_search_enabled=resolved_chat_web_search_enabled,
        chat_web_search_privacy_override=resolved_chat_web_search_privacy_override,
    )

    log.info(
        'chat_message_received',
        chat_id          = chat_id,
        client_request_id = client_request_id or None,
        stream_request_id = request_id,
        run_id           = requested_run_id,
        chat_mode        = resolved_chat_mode,
        topic_shift_reset = bool(context_scope_resolution.get('topic_shift_reset')),
        scope_transition_reset = bool(context_scope_resolution.get('scope_transition_reset')),
        context_generation = context_scope_resolution.get('generation'),
        message_length   = len(message_text),
        history_messages = len(history),
    )

    message_id = str(uuid.uuid4())
    trace_writer = get_trace_writer(chat_id, message_id, run_id=requested_run_id)
    request_resource_snapshot = capture_resource_snapshot()
    if trace_writer is not None:
        trace_writer.record('request', {
            'chat_id':          chat_id,
            'question':         message_text,
            'question_length':  len(message_text),
            'history_messages': len(history),
            'chat_mode':        resolved_chat_mode,
            'model_filename':   settings.llm_model_filename,
            'chat_web_search_enabled': resolved_chat_web_search_enabled,
            'chat_web_search_privacy_override': resolved_chat_web_search_privacy_override,
            'resource_snapshot': request_resource_snapshot,
        })

    # Build the SSE event generator
    async def _event_stream() -> AsyncGenerator[dict]:
        async with CHAT_GUARD.slot(check_rate=False):
            start_time = time.time()
            orchestrator = ChatOrchestrator()
            sse_tracker = SseContractTracker()
            status_emitter = SseStatusEmitter(chat_id=chat_id, start_time=start_time)
            stop_event = asyncio.Event()
            registry_registered = False
            await CHAT_STREAM_REGISTRY.register(
                stream_id=stream_id,
                chat_id=chat_id,
                request_id=request_id,
                stop_event=stop_event,
                task=asyncio.current_task(),
            )
            registry_registered = True
            log.info(
                'chat_stream_registered',
                chat_id=chat_id,
                stream_id=stream_id,
                request_id=request_id,
            )

            def _raise_if_user_stopped() -> None:
                if stop_event.is_set() and CHAT_STREAM_REGISTRY.is_stopped_by_user(stream_id):
                    raise UserStopRequestedError

            async def _flush_trace_writer_safe() -> None:
                if trace_writer is None:
                    return
                try:
                    await trace_writer.flush()
                except (RuntimeError, ValueError, TypeError, OSError) as trace_exc:
                    log.warning(
                        'chat_trace_flush_failed',
                        chat_id=chat_id,
                        stream_id=stream_id,
                        request_id=request_id,
                        error=str(trace_exc),
                    )

            def _update_sse_phase(event_name: str) -> None:
                if not sse_tracker.update(event_name):
                    log.warning(
                        'chat_sse_out_of_order',
                        chat_id=chat_id,
                        event_name=event_name,
                        current_phase=sse_tracker.current_phase,
                        event_phase=SSE_PHASE_ORDER.get(event_name, 0),
                    )

            _update_sse_phase('chat')
            chat_event = orchestrator.prepare_request(
                chat_id=chat_id,
                stream_id=stream_id,
                request_id=request_id if request_id else None,
            )
            yield {'event': chat_event['event'], 'data': serialize_api_response(chat_event['data'])}

            answer_parts: list[str] = []
            sources: list[ChatSourceReference] = []
            generation_seconds: float | None = None
            timeout_occurred = False
            timeout_reason: TimeoutReason | str | None = None
            assistant_message_id: int | None = None
            assistant_message_record: ChatMessage | None = None
            message_persisted = False
            cleaned_answer = ''
            completion_mode_override: CompletionMode | str | None = None
            budget_metrics: dict[str, object] = {}
            budget_checkpoints: list[dict[str, object]] = []
            has_remaining_scope = False
            stopped_by_user = False
            finalized_sources = False
            finalized_cleaned = False
            metrics_query_type = DiagnosticsQueryType.UNKNOWN.value
            metrics_raw_chunks_count = 0
            continuation_passes = 0
            pass_details: list[dict[str, object]] = []
            researcher_out_of_corpus = False
            continuation_resolution_reason: (
                ContinuationResolutionReason | StructuralGapReason | TimeoutReason | str | None
            ) = None
            continuation_progress_state: str | None = None
            status_transitions: list[dict[str, object]] = status_emitter.status_transitions
            resource_end_snapshot: dict[str, object] | None = None
            resource_metrics: dict[str, object] = {}
            previous_pass_raw_answer: str | None = None
            pre_classification_elapsed_ms: float | None = None
            sanitization_elapsed_ms: float | None = None
            terminal_state = 'unknown'

            try:
                _raise_if_user_stopped()
                if resolved_chat_mode != 'assistant':
                    classifying_status = status_emitter.build_event(
                        'classifying',
                        message='Analyzing your request...',
                    )
                    if classifying_status is not None:
                        yield classifying_status

                generation_started = False
                source_map: dict[tuple[str, str], ChatSourceReference] = {}
                continuation_request = _is_continuation_request(message_text)
                continuation_anchor_question = _resolve_continuation_anchor_question(
                    question=message_text,
                    history=history,
                )

                # Pre-classify to gate planning and lock classification for all passes.
                # Planning adds structural value only for multi-section synthesis routes;
                # running it on focused/simple/metadata queries wastes 9-28s with no benefit.
                # On failure, falls through to None so planning runs unconditionally (safe fallback).
                locked_classification = None
                if resolved_chat_mode != 'assistant':
                    try:
                        locked_classification, pre_classification_elapsed_ms = await classify_query_with_timing(
                            continuation_anchor_question,
                            scoped_file_active=bool(scoped_file_ids),
                        )
                        log.info(
                            'query_pre_classified',
                            chat_id=chat_id,
                            route_candidate=locked_classification.route_candidate,
                            confidence=locked_classification.confidence,
                            chat_mode=resolved_chat_mode,
                        )
                    except (RuntimeError, ValueError, TypeError, OSError) as exc:
                        log.warning('pre_classification_failed', chat_id=chat_id, error=str(exc), chat_mode=resolved_chat_mode)
                        locked_classification = None

                # Override continuation_request with the authoritative classifier result.
                if locked_classification is not None:
                    continuation_request = bool(continuation_request or locked_classification.is_continuation)
                    if continuation_request:
                        locked_classification = _normalize_continuation_classification(
                            classification=locked_classification,
                            continuation_anchor_question=continuation_anchor_question,
                        )

                auto_continue_enabled, max_auto_continue_rounds, auto_continue_prompt = _resolve_auto_continue_policy()
                base_history = list(history)
                max_total_passes = 1 + (max_auto_continue_rounds if auto_continue_enabled else 0)
                contract_spec = build_contract_spec(
                    question=continuation_anchor_question,
                    classification=locked_classification if resolved_chat_mode != 'assistant' else None,
                )
                if contract_spec.required_headings or contract_spec.min_year_count > 0:
                    max_total_passes = max(max_total_passes, 2)
                pass_index = 1
                continuation_contract_guidance: str | None = None
                while pass_index <= max_total_passes:
                    _raise_if_user_stopped()
                    pass_question = message_text
                    if continuation_request or pass_index > 1:
                        pass_question = _build_auto_continue_pass_prompt(
                            auto_continue_prompt=auto_continue_prompt,
                            original_question=continuation_anchor_question,
                        )
                        if continuation_contract_guidance:
                            pass_question = f"{pass_question}\n\n{continuation_contract_guidance}".strip()
                    pass_history = history
                    if pass_index > 1:
                        assistant_history_content = ''.join(answer_parts).strip()
                        pass_history = [
                            *base_history,
                            user_message,
                            ChatMessage(
                                chat_id=chat_id,
                                role='assistant',
                                content=assistant_history_content,
                                sources=serialize_sources(list(source_map.values())),
                                model_filename=settings.llm_model_filename,
                                completion_mode=CompletionMode.SCOPED_COMPLETE,
                                has_remaining_scope=True,
                                chat_mode=resolved_chat_mode,
                                retrieval_scope_kind=retrieval_scope_kind,
                                retrieval_scope_key=retrieval_scope_key,
                            ),
                        ]
                        continuing_status = status_emitter.build_event(
                            'continuing',
                            message=_CONTINUING_STATUS_MESSAGE,
                            pass_index=pass_index,
                            pass_total=max_total_passes,
                        )
                        if continuing_status is not None:
                            yield continuing_status

                    pass_sources: list[ChatSourceReference] = []
                    pass_has_remaining_scope = False
                    pass_completion_mode_override: CompletionMode | str | None = None
                    pass_answer_parts: list[str] = []
                    answer_before_pass = ''.join(answer_parts)
                    answer_length_before_pass = len(answer_before_pass)

                    # Emit "retrieving" status for pass 1 when classification is pre-computed.
                    # Normally this fires from the __classification__ intercept below, but when
                    # locked_classification is pre-set answer_question() skips classification
                    # and never yields ('__classification__', ...).
                    if pass_index == 1 and locked_classification is not None and resolved_chat_mode != 'assistant':
                        _retrieval_message = (
                            'Checking document index...'
                            if locked_classification.is_metadata_query
                            else 'Searching for relevant information...'
                        )
                        retrieving_status = status_emitter.build_event('retrieving', message=_retrieval_message)
                        if retrieving_status is not None:
                            yield retrieving_status

                    log.info(
                        'chat_answer_stream_begin',
                        chat_id=chat_id,
                        request_id=request_id,
                        pass_index=pass_index,
                        chat_mode=resolved_chat_mode,
                        question_length=len(pass_question),
                        history_messages=len(pass_history) if pass_history else 0,
                    )
                    answer_stream_started_at = time.perf_counter()
                    answer_items_seen = 0
                    answer_tokens_seen = 0
                    answer_list_events = 0
                    answer_next_task: asyncio.Task[object] | None = None
                    answer_iter = answer_question(
                        question=pass_question,
                        chat_id=chat_id,
                        file_ids=scoped_file_ids,
                        history=pass_history,
                        db=db,
                        trace=trace_writer,
                        classification=locked_classification,
                        chat_mode=resolved_chat_mode,
                        chat_web_search_enabled=resolved_chat_web_search_enabled,
                        chat_web_search_privacy_override=resolved_chat_web_search_privacy_override,
                    ).__aiter__()
                    try:
                        while True:
                            _raise_if_user_stopped()
                            if answer_next_task is None:
                                answer_next_task = asyncio.create_task(answer_iter.__anext__())
                            done, _ = await asyncio.wait(
                                {answer_next_task},
                                timeout=_ANSWER_STREAM_HEARTBEAT_SECONDS,
                            )
                            if not done:
                                log.warning(
                                    'chat_answer_stream_heartbeat_waiting',
                                    chat_id=chat_id,
                                    request_id=request_id,
                                    pass_index=pass_index,
                                    wait_seconds=_ANSWER_STREAM_HEARTBEAT_SECONDS,
                                    elapsed_seconds=round(time.perf_counter() - answer_stream_started_at, 1),
                                    items_seen=answer_items_seen,
                                    tokens_seen=answer_tokens_seen,
                                    list_events_seen=answer_list_events,
                                    sse_phase=sse_tracker.current_phase,
                                )
                                continue
                            try:
                                item = answer_next_task.result()
                            except StopAsyncIteration:
                                answer_next_task = None
                                break
                            finally:
                                answer_next_task = None
                            answer_items_seen += 1

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.CLASSIFICATION:
                                locked_classification = item[1]
                                _classification = item[1]
                                if resolved_chat_mode != 'assistant':
                                    _retrieval_message = (
                                        'Checking document index...'
                                        if _classification.is_metadata_query
                                        else 'Searching for relevant information...'
                                    )
                                    retrieving_status = status_emitter.build_event(
                                        'retrieving',
                                        message=_retrieval_message,
                                    )
                                    if retrieving_status is not None:
                                        yield retrieving_status
                                continue

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.SEARCHING_STATUS:
                                searching_payload = item[1] if isinstance(item[1], dict) else {}
                                searching_status = status_emitter.build_event(
                                    'searching',
                                    message=str(searching_payload.get('message') or 'Searching the web...'),
                                )
                                if searching_status is not None:
                                    yield searching_status
                                continue

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.TIMEOUT:
                                timeout_occurred = True
                                timeout_payload = item[1] if isinstance(item[1], dict) else {}
                                timeout_seconds = float(timeout_payload.get('timeout_seconds') or 0.0)
                                timeout_reason = normalize_timeout_reason(timeout_payload.get('reason'))
                                timeout_allows_continuation = not is_terminal_timeout_reason(timeout_reason)
                                pass_has_remaining_scope = timeout_allows_continuation
                                if timeout_allows_continuation:
                                    has_remaining_scope = True
                                if continuation_resolution_reason is None:
                                    continuation_resolution_reason = timeout_reason
                                _update_sse_phase('timeout')
                                yield {
                                    'event': 'timeout',
                                    'data': serialize_api_response({
                                        'message': f'Response truncated: generation time limit ({int(timeout_seconds) if timeout_seconds else "unknown"}s) reached',
                                        'elapsed_seconds': round(time.time() - start_time, 1),
                                        'timeout_seconds': timeout_seconds if timeout_seconds else None,
                                        'timeout_reason': timeout_reason,
                                    }),
                                }
                                continue

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.BUDGET_CHECKPOINT:
                                checkpoint_payload = item[1] if isinstance(item[1], dict) else {}
                                budget_checkpoints.append(checkpoint_payload)
                                _update_sse_phase('budget')
                                yield {
                                    'event': 'budget',
                                    'data': serialize_api_response(checkpoint_payload),
                                }
                                continue

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.PLAN_STEP:
                                step_payload = item[1] if isinstance(item[1], dict) else {}
                                _update_sse_phase('plan_step')
                                yield {
                                    'event': 'plan_step',
                                    'data': serialize_api_response(step_payload),
                                }
                                continue

                            if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.METRICS:
                                metrics_payload = item[1] if isinstance(item[1], dict) else {}
                                budget_metrics = metrics_payload
                                metrics_query_value = metrics_payload.get('query_type')
                                if isinstance(metrics_query_value, str) and metrics_query_value.strip():
                                    metrics_query_type = _normalize_diagnostics_query_type(metrics_query_value)
                                metrics_raw_chunks_count = safe_int(
                                    metrics_payload.get('raw_chunks_count'),
                                    default=metrics_raw_chunks_count,
                                )
                                remaining_scope_value = metrics_payload.get('has_remaining_scope')
                                if isinstance(remaining_scope_value, bool):
                                    pass_has_remaining_scope = remaining_scope_value
                                    has_remaining_scope = remaining_scope_value
                                suggested_mode = metrics_payload.get('suggested_completion_mode')
                                if isinstance(suggested_mode, str):
                                    try:
                                        normalized_mode = CompletionMode(suggested_mode.strip().lower())
                                    except ValueError:
                                        normalized_mode = None
                                    if normalized_mode is not None:
                                        pass_completion_mode_override = normalized_mode
                                        completion_mode_override = normalized_mode
                                if (
                                    resolved_chat_mode == 'researcher'
                                    and metrics_query_type in {DiagnosticsQueryType.FOCUSED.value, DiagnosticsQueryType.COVERAGE.value}
                                    and bool(metrics_payload.get('generation_skipped'))
                                    and not bool(metrics_payload.get('answerability_passed'))
                                    and not bool(getattr(locked_classification, 'is_metadata_query', False))
                                ):
                                    researcher_out_of_corpus = True
                                log.info(
                                    'chat_answer_stream_metrics_received',
                                    chat_id=chat_id,
                                    request_id=request_id,
                                    pass_index=pass_index,
                                    raw_chunks_count=metrics_raw_chunks_count,
                                    answerability_passed=metrics_payload.get('answerability_passed'),
                                    generation_skipped=metrics_payload.get('generation_skipped'),
                                    elapsed_seconds=round(time.perf_counter() - answer_stream_started_at, 2),
                                )
                                continue

                            if isinstance(item, str):
                                answer_tokens_seen += 1
                                if not generation_started:
                                    generation_started = True
                                    log.info(
                                        'chat_answer_stream_first_token',
                                        chat_id=chat_id,
                                        request_id=request_id,
                                        pass_index=pass_index,
                                        elapsed_seconds=round(time.perf_counter() - answer_stream_started_at, 2),
                                    )
                                    generating_status = status_emitter.build_event(
                                        'generating',
                                        message='Generating response...',
                                    )
                                    if generating_status is not None:
                                        yield generating_status
                                answer_parts.append(item)
                                pass_answer_parts.append(item)
                                _update_sse_phase('token')
                                yield {'event': 'token', 'data': item}
                            elif isinstance(item, list):
                                answer_list_events += 1
                                pass_sources = item
                    finally:
                        if answer_next_task is not None and not answer_next_task.done():
                            answer_next_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await answer_next_task
                    log.info(
                        'chat_answer_stream_end',
                        chat_id=chat_id,
                        request_id=request_id,
                        pass_index=pass_index,
                        elapsed_seconds=round(time.perf_counter() - answer_stream_started_at, 2),
                        items_seen=answer_items_seen,
                        tokens_seen=answer_tokens_seen,
                        list_events_seen=answer_list_events,
                    )

                    merge_sources(source_map, pass_sources)

                    pass_raw_answer = ''.join(pass_answer_parts).strip()
                    pass_cleaned_answer = sanitize_display_answer(pass_raw_answer) if pass_raw_answer else ''
                    pass_reasoning_only_output = bool(pass_raw_answer) and not pass_cleaned_answer and (
                        '<think>' in pass_raw_answer.lower() or '<<think>>' in pass_raw_answer.lower()
                    )
                    if pass_reasoning_only_output:
                        log.warning(
                            'chat_pass_reasoning_only_output_detected',
                            chat_id=chat_id,
                            pass_index=pass_index,
                        )

                    added_answer_chars = len(''.join(answer_parts)) - answer_length_before_pass
                    if pass_completion_mode_override is not None:
                        completion_mode_override = pass_completion_mode_override
                    contract_answer_text = pass_cleaned_answer if pass_cleaned_answer else pass_raw_answer
                    contract_validation = validate_contract(
                        answer=contract_answer_text,
                        spec=contract_spec,
                    )
                    has_contract_gap = contract_validation.has_gap
                    if has_contract_gap:
                        pass_has_remaining_scope = True
                        has_remaining_scope = True

                    pass_sources_exhausted = len(pass_sources) == 0
                    pass_continue_worthy_gap = pass_has_remaining_scope or has_contract_gap
                    pass_detail = {
                        'pass_index': pass_index,
                        'is_continuation': pass_index > 1,
                        'raw_answer_length': len(pass_raw_answer),
                        'cleaned_answer_length': len(pass_cleaned_answer),
                        'reasoning_only_output_detected': pass_reasoning_only_output,
                        'sources_count': len(pass_sources),
                        'pass_requires_more_work': pass_continue_worthy_gap,
                        'raw_has_remaining_scope_signal': pass_has_remaining_scope,
                        'missing_required_headings': contract_validation.missing_required_headings,
                        'required_year_count': contract_validation.required_year_count,
                        'observed_year_count': contract_validation.observed_year_count,
                    }
                    pass_details.append(pass_detail)
                    pass_artifact_has_remaining_scope = pass_continue_worthy_gap
                    pass_completion_mode = pass_completion_mode_override
                    try:
                        pass_completion_mode = CompletionMode(str(pass_completion_mode).strip().lower())
                    except ValueError:
                        pass_completion_mode = (
                            CompletionMode.SCOPED_COMPLETE
                            if pass_continue_worthy_gap
                            else CompletionMode.COMPLETE
                        )
                    if pass_continue_worthy_gap and pass_completion_mode == CompletionMode.COMPLETE:
                        pass_completion_mode = CompletionMode.SCOPED_COMPLETE
                    elif (not pass_continue_worthy_gap) and pass_completion_mode == CompletionMode.SCOPED_COMPLETE:
                        pass_completion_mode = CompletionMode.COMPLETE
                    pass_artifact_has_remaining_scope = pass_completion_mode in {
                        CompletionMode.PARTIAL,
                        CompletionMode.SCOPED_COMPLETE,
                        CompletionMode.STOPPED,
                    }
                    pass_next_action_reason = timeout_reason if timeout_occurred and timeout_reason else None
                    pass_sources_payload = serialize_sources(pass_sources)
                    pass_stitch_mode = 'append'
                    pass_artifact = ContinuationPassArtifact(
                        chat_id=chat_id,
                        request_id=artifact_request_id,
                        pass_index=pass_index,
                        stitch_mode=pass_stitch_mode,
                        raw_answer=pass_raw_answer,
                        cleaned_answer=pass_cleaned_answer,
                        has_remaining_scope=pass_artifact_has_remaining_scope,
                        completion_mode=pass_completion_mode,
                        next_action_reason=pass_next_action_reason,
                        sources=pass_sources_payload,
                        pass_details=pass_detail,
                        status_transitions=list(status_transitions),
                    )
                    try:
                        await insert_continuation_pass_artifact(db, pass_artifact)
                    except _PERSISTENCE_EXCEPTIONS as artifact_error:
                        log.warning(
                            'chat_pass_artifact_persist_failed',
                            chat_id=chat_id,
                            request_id=artifact_request_id,
                            pass_index=pass_index,
                            error=str(artifact_error),
                        )
                    pass_has_unresolved_targets = (
                        pass_continue_worthy_gap
                    )
                    pass_duplicate_of_previous = (
                        pass_index > 1
                        and _is_duplicate_continuation_pass(previous_pass_raw_answer, pass_raw_answer)
                    )
                    if pass_duplicate_of_previous and pass_has_unresolved_targets:
                        continuation_resolution_reason = ContinuationResolutionReason.DUPLICATE_CONTINUATION_DETECTED
                        continuation_progress_state = 'stalled'
                        completion_mode_override = CompletionMode.SCOPED_COMPLETE
                        has_remaining_scope = False
                        break
                    can_auto_continue = (
                        pass_index < max_total_passes
                        and pass_has_unresolved_targets
                        and not pass_sources_exhausted
                        and added_answer_chars > 0
                    )
                    if not can_auto_continue:
                        break

                    continuation_contract_guidance = build_repair_guidance(contract_validation)
                    previous_pass_raw_answer = pass_raw_answer

                    continuation_passes += 1
                    pass_index += 1
                    continue

                sources = list(source_map.values())
                _raise_if_user_stopped()

                finalizing_status = status_emitter.build_event(
                    'finalizing',
                    message='Finalizing answer...',
                )
                if finalizing_status is not None:
                    yield finalizing_status

                source_dicts = serialize_sources(sources)
                _update_sse_phase('sources')
                yield {'event': 'sources', 'data': serialize_api_response(source_dicts)}
                finalized_sources = True

                full_answer = ''.join(answer_parts).strip()
                if not full_answer:
                    full_answer = 'I could not find enough information to answer your question.'
                    log.warning('chat_empty_after_cleaning', chat_id=chat_id)
                requested_max_words = answer_sanitization.extract_requested_max_words(message_text)
                if isinstance(requested_max_words, int) and requested_max_words > 0:
                    before_word_count = answer_sanitization.count_words(full_answer)
                    full_answer, word_limit_applied = answer_sanitization.truncate_to_word_limit(
                        full_answer,
                        requested_max_words,
                    )
                    if word_limit_applied:
                        after_word_count = answer_sanitization.count_words(full_answer)
                        log.info(
                            'chat_word_limit_enforced',
                            chat_id=chat_id,
                            max_words=requested_max_words,
                            before_words=before_word_count,
                            after_words=after_word_count,
                        )
                generation_seconds = time.time() - start_time
                sanitize_started = time.perf_counter()
                cleaned_answer, reasoning_only_output = build_display_answer(full_answer)
                sanitization_elapsed_ms = (time.perf_counter() - sanitize_started) * 1000.0
                finalized_contract_answer = cleaned_answer if cleaned_answer else full_answer
                finalized_contract_answer, missing_sections_filled = enforce_required_sections(
                    answer=finalized_contract_answer,
                    spec=contract_spec,
                )
                if missing_sections_filled:
                    if cleaned_answer:
                        cleaned_answer = finalized_contract_answer
                    else:
                        full_answer = finalized_contract_answer
                        cleaned_answer = sanitize_display_answer(full_answer)
                    log.info(
                        'contract_sections_filled_at_closeout',
                        chat_id=chat_id,
                        missing_required_headings=missing_sections_filled,
                    )
                structural_incomplete_reason = _detect_structural_incomplete_reason(cleaned_answer or full_answer)
                if structural_incomplete_reason and completion_mode_override != CompletionMode.STOPPED:
                    log.info(
                        'structural_gap_triggers_continuation',
                        source='structural_incomplete_reason',
                        reason=structural_incomplete_reason,
                    )
                    has_remaining_scope = True
                    completion_mode_override = CompletionMode.SCOPED_COMPLETE
                    if continuation_resolution_reason is None:
                        continuation_resolution_reason = structural_incomplete_reason
                estimated_unsupported_claim_count, estimated_evidence_coverage_rate, estimated_not_found_count = (
                    estimate_evidence_metrics(
                        answer=cleaned_answer if cleaned_answer else full_answer,
                        source_texts=[str(source.get('chunk_preview', '') or '') for source in source_dicts],
                    )
                )
                budget_metrics['unsupported_claim_count'] = estimated_unsupported_claim_count
                budget_metrics['evidence_coverage_rate'] = estimated_evidence_coverage_rate
                budget_metrics['not_found_count'] = estimated_not_found_count
                query_type = DiagnosticsQueryType.UNKNOWN.value
                if trace_writer is not None and hasattr(trace_writer, 'get_sections'):
                    sections = trace_writer.get_sections()
                    query_type = str(sections.get('intent', {}).get('query_type', DiagnosticsQueryType.UNKNOWN.value))

                if reasoning_only_output:
                    log.warning(
                        'chat_reasoning_only_output_detected',
                        chat_id=chat_id,
                        query_type=query_type,
                        answer_length=len(full_answer),
                        sources_count=len(source_dicts),
                    )

                if metrics_query_type == DiagnosticsQueryType.UNKNOWN.value:
                    metrics_query_type = _normalize_diagnostics_query_type(query_type)
                if continuation_passes > 0 and continuation_progress_state is None:
                    continuation_progress_state = 'progressed' if not has_remaining_scope else 'budget_exhausted'
                if pre_classification_elapsed_ms is not None:
                    budget_metrics['classification_duration_ms'] = round(pre_classification_elapsed_ms, 1)
                if sanitization_elapsed_ms is not None:
                    budget_metrics['sanitization_duration_ms'] = round(sanitization_elapsed_ms, 1)

                _update_sse_phase('cleaned')
                yield {'event': 'cleaned', 'data': cleaned_answer}
                finalized_cleaned = True

                (
                    message_completion_mode,
                    message_has_remaining_scope,
                    message_next_action,
                    message_next_action_reason,
                ) = resolve_completion_and_action(
                    completion_mode_override=completion_mode_override,
                    timeout_occurred=timeout_occurred,
                    timeout_reason=timeout_reason,
                    has_remaining_scope=has_remaining_scope,
                    stopped_by_user=False,
                    continuation_resolution_reason=continuation_resolution_reason,
                    chat_mode=resolved_chat_mode,
                    researcher_out_of_corpus=researcher_out_of_corpus,
                    answer_signals_out_of_corpus=_answer_signals_out_of_corpus(cleaned_answer),
                )
                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role='assistant',
                    content=full_answer,
                    sources=source_dicts,
                    model_filename=settings.llm_model_filename,
                    generation_seconds=generation_seconds,
                    completion_mode=message_completion_mode,
                    has_remaining_scope=message_has_remaining_scope,
                    next_action=message_next_action,
                    next_action_reason=message_next_action_reason,
                    chat_mode=resolved_chat_mode,
                    retrieval_scope_kind=retrieval_scope_kind,
                    retrieval_scope_key=retrieval_scope_key,
                    is_internal=False,
                )
                persist_db = await get_connection()
                try:
                    assistant_message = await insert_chat_message(persist_db, assistant_message)
                    assistant_message_record = assistant_message
                    assistant_message_id = assistant_message.id
                    message_persisted = assistant_message_id is not None
                except _PERSISTENCE_EXCEPTIONS as persist_error:
                    log.warning('chat_response_persist_retry', chat_id=chat_id, error=str(persist_error))
                    assistant_message = await insert_chat_message(db, assistant_message)
                    assistant_message_record = assistant_message
                    assistant_message_id = assistant_message.id
                    message_persisted = assistant_message_id is not None
                finally:
                    await persist_db.close()

                if assistant_message_id is not None:
                    seen_source_file_ids: set[int] = set()
                    for source in source_dicts:
                        file_id_raw = source.get('file_id')
                        try:
                            file_id = int(file_id_raw) if file_id_raw is not None else None
                        except (TypeError, ValueError):
                            file_id = None
                        if file_id is None or file_id <= 0:
                            continue
                        if file_id in seen_source_file_ids:
                            continue
                        seen_source_file_ids.add(file_id)
                        await append_chat_upload_reference_message(
                            db,
                            chat_id=chat_id,
                            file_id=file_id,
                            message_id=int(assistant_message_id),
                        )

                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    timing_trace_payload: dict[str, object] = {
                        'elapsed_seconds': round(generation_seconds, 3),
                    }
                    for key in (
                        'classification_duration_ms',
                        'retrieval_duration_ms',
                        'embed_ms',
                        'vector_search_ms',
                        'rerank_ms',
                        'prompt_duration_ms',
                        'prompt_build_ms',
                        'first_token_latency_ms',
                        'ttft_ms',
                        'stream_duration_ms',
                        'sanitization_duration_ms',
                    ):
                        value = budget_metrics.get(key)
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            timing_trace_payload[key] = round(float(value), 3)
                    trace_writer.record('timing', timing_trace_payload)
                    trace_writer.record('response', {
                        'answer_length': len(full_answer),
                        'display_answer_length': len(cleaned_answer),
                        'answer_preview': cleaned_answer[:MAX_ANSWER_PREVIEW_LENGTH] if cleaned_answer else '',
                        'display_answer_preview': cleaned_answer[:MAX_ANSWER_PREVIEW_LENGTH] if cleaned_answer else '',
                        'raw_answer_preview': full_answer[:MAX_ANSWER_PREVIEW_LENGTH] if full_answer else '',
                        'sources_count': len(sources),
                        'sources': source_dicts,
                        'unsupported_claim_count': safe_int(
                            budget_metrics.get('unsupported_claim_count'),
                            default=0,
                        ),
                        'evidence_coverage_rate': safe_float(
                            budget_metrics.get('evidence_coverage_rate'),
                            default=0.0,
                        ),
                        'not_found_count': safe_int(
                            budget_metrics.get('not_found_count'),
                            default=0,
                        ),
                        'continuation_passes': continuation_passes,
                        'pass_details': pass_details,
                        'status_transitions': status_transitions,
                        'resource_metrics': resource_metrics,
                    })
                    await _flush_trace_writer_safe()

            except UserStopRequestedError:
                terminal_state = 'stopped'
                stopped_by_user = True
                generation_seconds = time.time() - start_time
                partial_answer = ''.join(answer_parts).strip()
                partial_sources = serialize_sources(sources) if sources else []
                has_remaining_scope = True
                completion_mode_override = CompletionMode.STOPPED
                cleaned_answer = sanitize_display_answer(partial_answer) if partial_answer else ''

                finalizing_status = status_emitter.build_event(
                    'finalizing',
                    message='Finalizing answer...',
                )
                if finalizing_status is not None:
                    yield finalizing_status

                if not finalized_sources:
                    _update_sse_phase('sources')
                    yield {'event': 'sources', 'data': serialize_api_response(partial_sources)}
                    finalized_sources = True
                if not finalized_cleaned:
                    _update_sse_phase('cleaned')
                    yield {'event': 'cleaned', 'data': cleaned_answer}
                    finalized_cleaned = True

                persisted_message, persisted = await _persist_terminal_assistant_message(
                    chat_id=chat_id,
                    content=partial_answer,
                    sources=partial_sources,
                    generation_seconds=generation_seconds,
                    completion_mode=CompletionMode.STOPPED,
                    stopped_by_user=True,
                    has_remaining_scope=True,
                    next_action=NextAction.REGENERATE,
                    next_action_reason='stopped',
                    chat_mode=resolved_chat_mode,
                    retrieval_scope_kind=retrieval_scope_kind,
                    retrieval_scope_key=retrieval_scope_key,
                )
                if persisted_message is not None:
                    assistant_message_record = persisted_message
                    assistant_message_id = persisted_message.id
                message_persisted = persisted
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_cancelled', {
                        'generation_seconds': generation_seconds,
                        'tokens_generated': len(answer_parts),
                        'stopped_by_user': True,
                        'resource_metrics': resource_metrics,
                    })
                    await _flush_trace_writer_safe()
                log.info(
                    'chat_response_cancelled',
                    chat_id=chat_id,
                    stream_id=stream_id,
                    request_id=request_id,
                    cancellation_reason='user_stop',
                    tokens_streamed=len(answer_parts),
                    generation_seconds=round(generation_seconds, 3),
                )

            except asyncio.CancelledError:
                terminal_state = 'cancelled'
                generation_seconds = time.time() - start_time
                partial_answer = ''.join(answer_parts).strip()
                stream_stopped_by_user = CHAT_STREAM_REGISTRY.is_stopped_by_user(stream_id) or stop_event.is_set()
                if partial_answer or stream_stopped_by_user:
                    cancelled_next_action, cancelled_next_action_reason = _resolve_next_action(
                        stopped_by_user=stream_stopped_by_user,
                        timeout_occurred=False,
                        has_remaining_scope=stream_stopped_by_user,
                        continuation_resolution_reason=None,
                    )
                    cancelled_sources = serialize_sources(sources) if sources else []
                    assistant_message_record, _ = await _persist_terminal_assistant_message(
                        chat_id=chat_id,
                        content=partial_answer,
                        sources=cancelled_sources,
                        generation_seconds=generation_seconds,
                        completion_mode=(
                            CompletionMode.STOPPED if stream_stopped_by_user else CompletionMode.PARTIAL
                        ),
                        stopped_by_user=stream_stopped_by_user,
                        has_remaining_scope=stream_stopped_by_user,
                        next_action=cancelled_next_action,
                        next_action_reason=cancelled_next_action_reason,
                        chat_mode=resolved_chat_mode,
                        retrieval_scope_kind=retrieval_scope_kind,
                        retrieval_scope_key=retrieval_scope_key,
                    )
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_cancelled', {
                        'generation_seconds': generation_seconds,
                        'tokens_generated': len(answer_parts),
                        'stopped_by_user': stream_stopped_by_user,
                        'resource_metrics': resource_metrics,
                    })
                    await _flush_trace_writer_safe()
                log.info(
                    'chat_response_cancelled',
                    chat_id=chat_id,
                    stream_id=stream_id,
                    request_id=request_id,
                    cancellation_reason='task_cancelled',
                    stopped_by_user=stream_stopped_by_user,
                    tokens_streamed=len(answer_parts),
                    generation_seconds=round(generation_seconds, 3),
                )
                if registry_registered:
                    removed = await CHAT_STREAM_REGISTRY.unregister(stream_id)
                    registry_registered = False
                    if removed:
                        log.info(
                            'chat_stream_unregistered',
                            chat_id=chat_id,
                            stream_id=stream_id,
                            request_id=request_id,
                            terminal_state=terminal_state,
                        )
                raise

            except _STREAM_RUNTIME_EXCEPTIONS as exc:
                terminal_state = 'error'
                generation_seconds = time.time() - start_time
                log.error(
                    'chat_stream_error',
                    chat_id=chat_id,
                    error=str(exc),
                    generation_seconds=generation_seconds,
                    exc_info=True,
                )
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_error', {
                        'error': str(exc),
                        'resource_metrics': resource_metrics,
                    })
                    await _flush_trace_writer_safe()
                _update_sse_phase('error')
                yield {
                    'event': 'error',
                    'data': serialize_api_response({'error': to_client_error_message(exc)}),
                }

            completion_mode, done_has_remaining_scope, next_action, next_action_reason = (
                resolve_completion_and_action(
                    completion_mode_override=completion_mode_override,
                    timeout_occurred=timeout_occurred,
                    timeout_reason=timeout_reason,
                    has_remaining_scope=has_remaining_scope,
                    stopped_by_user=stopped_by_user,
                    continuation_resolution_reason=continuation_resolution_reason,
                    chat_mode=resolved_chat_mode,
                    researcher_out_of_corpus=researcher_out_of_corpus,
                    answer_signals_out_of_corpus=_answer_signals_out_of_corpus(cleaned_answer),
                )
            )
            resolved_completion_mode = completion_mode
            resolved_has_remaining_scope = done_has_remaining_scope
            resolved_next_action = next_action
            resolved_next_action_reason = next_action_reason
            if assistant_message_record is not None:
                resolved_completion_mode = assistant_message_record.completion_mode or resolved_completion_mode
                resolved_has_remaining_scope = bool(assistant_message_record.has_remaining_scope)
                resolved_next_action = assistant_message_record.next_action
                resolved_next_action_reason = assistant_message_record.next_action_reason

            if metrics_raw_chunks_count <= 0 and trace_writer is not None and hasattr(trace_writer, 'get_sections'):
                sections = trace_writer.get_sections()
                retrieval = sections.get('retrieval', {}) if isinstance(sections, dict) else {}
                metrics_raw_chunks_count = safe_int(retrieval.get('raw_chunks_count'), default=0)
                if metrics_query_type == DiagnosticsQueryType.UNKNOWN.value:
                    intent = sections.get('intent', {}) if isinstance(sections, dict) else {}
                    inferred_query_type = intent.get('query_type') if isinstance(intent, dict) else None
                    if isinstance(inferred_query_type, str) and inferred_query_type.strip():
                        metrics_query_type = _normalize_diagnostics_query_type(inferred_query_type)

            final_answer = ''.join(answer_parts).strip()
            refusal_text = cleaned_answer if cleaned_answer else final_answer
            metrics_unsupported_claim_count = safe_int(
                budget_metrics.get('unsupported_claim_count'),
                default=0,
            )
            metrics_evidence_coverage_rate = safe_float(
                budget_metrics.get('evidence_coverage_rate'),
                default=0.0,
            )
            metrics_not_found_count = safe_int(
                budget_metrics.get('not_found_count'),
                default=0,
            )
            metrics_model = EvalMetrics(
                chat_id=chat_id,
                question=message_text,
                model_filename=settings.llm_model_filename,
                query_type=metrics_query_type,
                raw_chunks_count=metrics_raw_chunks_count,
                sources_count=len(sources),
                generation_seconds=safe_float(generation_seconds, default=0.0),
                answer_length=len(final_answer),
                timeout_occurred=bool(timeout_occurred),
                has_empty_answer=not final_answer,
                has_refusal_pattern=False,
                unsupported_claim_count=metrics_unsupported_claim_count,
                evidence_coverage_rate=metrics_evidence_coverage_rate,
                not_found_count=metrics_not_found_count,
            )
            detected_issues = detect_issues(refusal_text, metrics_model)
            issue_strings = [issue.value for issue in detected_issues]
            try:
                await insert_diagnostics_metrics(
                    db=db,
                    metrics=metrics_model,
                    detected_issues=issue_strings,
                    run_id=requested_run_id,
                )
            except _PERSISTENCE_EXCEPTIONS as metrics_exc:
                log.warning(
                    'chat_metrics_persist_failed',
                    chat_id=chat_id,
                    error=str(metrics_exc),
                )

            if resource_end_snapshot is None:
                resource_end_snapshot = capture_resource_snapshot()
            if not resource_metrics:
                resource_metrics = {
                    'before': request_resource_snapshot,
                    'after': resource_end_snapshot,
                    'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                }
            done_data = build_done_payload(
                elapsed_seconds=generation_seconds,
                request_id=request_id if request_id else None,
                chat_mode=resolved_chat_mode,
                timeout_occurred=timeout_occurred,
                timeout_reason=timeout_reason,
                completion_mode=resolved_completion_mode,
                has_remaining_scope=resolved_has_remaining_scope,
                stopped_by_user=stopped_by_user,
                next_action=resolved_next_action,
                next_action_reason=resolved_next_action_reason,
                sources_count=len(sources),
                message_persisted=message_persisted,
                cleaned_answer=cleaned_answer,
                budget_metrics=budget_metrics,
                budget_checkpoints=budget_checkpoints,
                continuation_passes=continuation_passes,
                continuation_resolution_reason=continuation_resolution_reason,
                continuation_progress_state=continuation_progress_state,
                pass_details=pass_details,
                status_transitions=status_transitions,
                resource_metrics=resource_metrics,
                message_id=assistant_message_id,
            )
            done_data['upload_scope'] = {
                'active_upload_ids': upload_active_ids,
                'selected_upload_ids': upload_scope_selected_ids,
                'ready_file_ids': upload_ready_file_ids,
                'indexing_upload_ids': upload_indexing_ids,
                'omitted_upload_ids': upload_scope_omitted_ids,
                'resolution_mode': upload_scope_resolution_mode,
            }
            log.info(
                'chat_response_completed',
                chat_id=chat_id,
                chat_mode=resolved_chat_mode,
                completion_mode=done_data.get('completion_mode'),
                has_remaining_scope=done_data.get('has_remaining_scope'),
                next_action=done_data.get('next_action'),
                next_action_reason=done_data.get('next_action_reason'),
                timeout_occurred=timeout_occurred,
                timeout_reason=timeout_reason,
                message_persisted=message_persisted,
                sources_count=len(sources),
                tokens_streamed=len(answer_parts),
                continuation_passes=continuation_passes,
                duration_ms=round((generation_seconds or 0.0) * 1000, 1),
                process_rss_mb=resource_end_snapshot.get('process_rss_mb') if isinstance(resource_end_snapshot, dict) else None,
                process_cpu_percent=resource_end_snapshot.get('process_cpu_percent') if isinstance(resource_end_snapshot, dict) else None,
                system_cpu_percent=resource_end_snapshot.get('system_cpu_percent') if isinstance(resource_end_snapshot, dict) else None,
            )
            terminal_state = 'done'
            _update_sse_phase('done')
            yield {'event': 'done', 'data': serialize_api_response(done_data)}
            if registry_registered:
                removed = await CHAT_STREAM_REGISTRY.unregister(stream_id)
                registry_registered = False
                if removed:
                    log.info(
                        'chat_stream_unregistered',
                        chat_id=chat_id,
                        stream_id=stream_id,
                        request_id=request_id,
                        terminal_state=terminal_state,
                    )

    return EventSourceResponse(_event_stream())


@router.post('/api/chat/stop')
async def stop_chat(request: ChatStopRequest) -> dict:
    stop_outcome = await CHAT_STREAM_REGISTRY.mark_stopped_by_user(
        stream_id=request.stream_id,
        request_id=request.request_id,
        chat_id=request.chat_id,
    )
    stop_status = stop_outcome.status
    resolved_stream_id = stop_outcome.stream_id or request.stream_id
    resolved_request_id = stop_outcome.request_id or request.request_id
    resolved_chat_id = stop_outcome.chat_id or request.chat_id
    log.info(
        'chat_stop_acknowledged',
        chat_id=resolved_chat_id,
        stream_id=resolved_stream_id,
        request_id=resolved_request_id,
        stop_status=stop_status,
    )
    if stop_status == 'stopped_now' and resolved_stream_id:
        existing = _STOP_FINALIZATION_TASKS.get(resolved_stream_id)
        if existing is None or existing.done():
            _STOP_FINALIZATION_TASKS[resolved_stream_id] = asyncio.create_task(
                _finalize_stopped_stream_if_active(
                    stream_id=resolved_stream_id,
                    chat_id=resolved_chat_id,
                    request_id=resolved_request_id,
                ),
            )
    return {
        'status': stop_status,
        'stopped': stop_status == 'stopped_now',
        'stream_id': resolved_stream_id,
        'request_id': resolved_request_id,
    }


# ==============================================================================
# GET /api/chat/messages/{message_id}/raw — fetch raw content for a message
# ==============================================================================

@router.get('/api/chat/messages/{message_id}/raw')
async def get_message_raw(
    message_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Return raw content for a message (with <think> blocks). Used for on-demand
    # display when enable_raw_output_control is enabled. Only assistant messages
    # have meaningful raw content; user messages return their content as-is.
    message = await get_chat_message_by_id(db, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail='Message not found')
    return {'content': message.content}


# ==============================================================================
# GET /api/chat/chats — list recent chats
# ==============================================================================

@router.get('/api/chat/chats')
async def list_chats(
    limit:  int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    search_param = search.strip() if search and search.strip() else None
    chats = await get_chats(db, limit=limit, offset=offset, search=search_param)
    total = await get_chat_count(db, search=search_param)

    return {
        'chats':   chats,
        'total':   total,
        'limit':   limit,
        'offset':  offset,
    }


# ==============================================================================
# GET /api/chat/chats/{chat_id} — get chat messages
# ==============================================================================

@router.get('/api/chat/chats/{chat_id}')
async def get_chat_messages(
    chat_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    messages = await get_chat(db, chat_id)

    if not messages:
        raise HTTPException(status_code=404, detail='Chat not found')

    serialized_messages: list[dict[str, object]] = []
    for message in messages:
        payload = message.model_dump(mode='json')
        if message.role == ChatRole.ASSISTANT:
            cleaned_content, _ = build_display_answer(message.content)
            payload['content'] = cleaned_content
            payload['display_blocks'] = build_display_blocks(cleaned_content)
        serialized_messages.append(payload)
    chat_preferences = await get_chat_preferences(db, chat_id)

    return {
        'chat_id':                           chat_id,
        'messages':                          serialized_messages,
        'total':                             len(messages),
        'chat_web_search_enabled':           bool(chat_preferences.get('chat_web_search_enabled')),
        'chat_web_search_privacy_override':  bool(chat_preferences.get('chat_web_search_privacy_override')),
    }


# ==============================================================================
# PUT /api/chat/chats/{chat_id}/preferences — update chat preferences
# ==============================================================================

@router.put('/api/chat/chats/{chat_id}/preferences')
async def update_chat_preferences(
    chat_id: str,
    request: ChatPreferencesUpdateRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    messages = await get_chat(db, chat_id)
    if not messages:
        raise HTTPException(status_code=404, detail='Chat not found')
    preferences = await upsert_chat_preferences(
        db,
        chat_id,
        chat_web_search_enabled=request.chat_web_search_enabled,
        chat_web_search_privacy_override=request.chat_web_search_privacy_override,
    )
    return {
        'chat_id': chat_id,
        'chat_web_search_enabled': bool(preferences.get('chat_web_search_enabled')),
        'chat_web_search_privacy_override': bool(preferences.get('chat_web_search_privacy_override')),
    }


# ==============================================================================
# PUT /api/chat/chats/{chat_id}/title — rename chat
# ==============================================================================

@router.put('/api/chat/chats/{chat_id}/title')
async def update_chat_title(
    chat_id: str,
    title: str = Query(..., min_length=1, max_length=200),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    messages = await get_chat(db, chat_id)
    if not messages:
        raise HTTPException(status_code=404, detail='Chat not found')

    await set_chat_title(db, chat_id, title.strip())
    log.info('chat_title_updated', chat_id=chat_id, title=title)

    return {
        'chat_id': chat_id,
        'title':   title.strip(),
    }


# ==============================================================================
# DELETE /api/chat/chats/{chat_id} — delete chat
# ==============================================================================

@router.delete('/api/chat/chats/{chat_id}')
async def delete_chat_endpoint(
    chat_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    attachments = await get_chat_upload_attachments(db, chat_id=chat_id, include_deleted=True)
    uploads_deleted = 0
    upload_delete_failures = 0
    for attachment in attachments:
        if attachment.state != 'deleted':
            deleted = await _delete_chat_upload_artifacts(db=db, attachment=attachment)
            if not deleted:
                upload_delete_failures += 1
        uploads_deleted += 1
    if upload_delete_failures > 0:
        raise HTTPException(
            status_code=500,
            detail='Failed to fully delete one or more uploaded file artifacts. Retry chat deletion.',
        )
    chat_upload_dir = _upload_chat_dir(chat_id)
    if chat_upload_dir.exists():
        try:
            shutil.rmtree(chat_upload_dir)
        except (OSError, RuntimeError) as exc:
            log.warning('chat_upload_chat_dir_remove_failed', chat_id=chat_id, error=str(exc))

    deleted = await delete_chat(db, chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Chat not found')

    log.info('chat_deleted', chat_id=chat_id, uploads_deleted=uploads_deleted)

    return {
        'chat_id':  chat_id,
        'deleted':  True,
        'uploads_deleted': uploads_deleted,
    }
