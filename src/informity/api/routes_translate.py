# ==============================================================================
# Informity AI — Translation Routes (v3)
# Upload, job management, SSE streaming, result export.
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import aiosqlite
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from informity.api.security import LLM_BUSY_DETAIL, raise_if_llm_busy
from informity.api.upload_helpers import index_uploaded_file
from informity.config import settings
from informity.db.sqlite import (
    create_translate_job,
    create_translate_section,
    get_db,
    get_file_by_id,
    get_translate_job,
    get_translate_job_result,
    get_translate_sections,
    update_translate_job,
    update_translate_section,
)
from informity.indexer.pipeline import remove_file
from informity.llm.engine import llm_engine
from informity.log_events import emit_log_event
from informity.scanner.crawler import scanned_file_for_path
from informity.translate_languages import normalize_translate_language
from informity.translate_policy import (
    TONE_INSTRUCTIONS,
    TRANSLATE_AVG_SECTION_SECONDS,
    TRANSLATE_AVG_TOKENS_PER_PAGE,
    TRANSLATE_BATCH_TARGET_TOKENS,
    TRANSLATE_ENTITY_TYPE,
    TRANSLATE_GLOSSARY_INPUT_TOKENS,
    TRANSLATE_GLOSSARY_MAX_TOKENS,
    TRANSLATE_GLOSSARY_TEMPERATURE,
    TRANSLATE_GLOSSARY_TERM_COUNT,
    TRANSLATE_GLOSSARY_TIMEOUT_S,
    TRANSLATE_JOB_MAX_RUNTIME_S,
    TRANSLATE_JOB_STALL_S,
    TRANSLATE_PROVIDER,
    TRANSLATE_RETRY_TOKEN_CAP,
    TRANSLATE_SECTION_RETRY_MAX,
    TRANSLATE_SECTION_TIMEOUT_S,
    TRANSLATE_SOFT_PAGE_LIMIT,
    TRANSLATE_SOFT_SECTION_LIMIT,
    TRANSLATE_STORAGE_DIRNAME,
    TRANSLATE_TEMPERATURE,
)

log = structlog.get_logger(__name__)
TRANSLATE_CANCEL_ERROR_TOKEN = 'cancelled_by_user'


def capitalize(s: str) -> str:
    return s.capitalize() if s else s
router = APIRouter()

# One translation job at a time (local LLM is single-instance).
_translate_lock = asyncio.Lock()

# Per-job event queues: SSE stream reads from these; worker pushes to them.
# Sentinel value None signals the stream to close.
_job_queues: dict[str, asyncio.Queue] = {}

# Per-job cancel events: set() immediately closes the active generate_stream
# generator so the LLM lock releases without waiting for section timeout.
_job_cancel_events: dict[str, asyncio.Event] = {}

_SENTINEL = None
_MAX_UPLOAD_MB = 50


# ==============================================================================
# Helpers
# ==============================================================================

def _translate_storage_dir() -> Path:
    return settings.app_data_dir / TRANSLATE_STORAGE_DIRNAME


def _upload_dir(upload_id: str) -> Path:
    return _translate_storage_dir() / upload_id


def _sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r'[^\w\s\-.]', '_', name)
    return name.strip() or 'upload'


def _count_tokens(text: str) -> int:
    try:
        return llm_engine.count_tokens(text)
    except Exception as exc:
        log.warning('translate_count_tokens_fallback', error=str(exc), error_type=type(exc).__name__)
        return max(1, len(text.split()))


async def _emit(job_id: str, event: str, data: dict) -> None:
    q = _job_queues.get(job_id)
    if q is not None:
        await q.put({'event': event, 'data': json.dumps(data)})


def _build_glossary_block(glossary_json: str | None) -> str:
    if not glossary_json:
        return ''
    try:
        terms = json.loads(glossary_json)
        if not terms:
            return ''
        lines = '\n'.join(f'- {t["source"]} → {t["translation"]}' for t in terms if t.get('source') and t.get('translation'))
        return f'\nTerminology (use these translations consistently):\n{lines}\n' if lines else ''
    except Exception as exc:
        log.warning('translate_glossary_block_fallback', error=str(exc), error_type=type(exc).__name__)
        return ''


def _is_cancelled_translate_row(row: object) -> bool:
    return bool(
        row
        and str(row['status']) == 'stalled'
        and str(row['error'] or '') == TRANSLATE_CANCEL_ERROR_TOKEN
    )


def _split_text_for_translation(text: str, max_tokens: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    def _pack_units(units: list[str], joiner: str) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            normalized = unit.strip()
            if not normalized:
                continue
            token_count = _count_tokens(normalized)
            if current and current_tokens + token_count > max_tokens:
                packed.append(joiner.join(current).strip())
                current = []
                current_tokens = 0
            current.append(normalized)
            current_tokens += token_count
        if current:
            packed.append(joiner.join(current).strip())
        return [chunk for chunk in packed if chunk]

    paragraph_chunks = [chunk for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    if len(paragraph_chunks) > 1:
        packed = _pack_units(paragraph_chunks, '\n\n')
        if len(packed) > 1 or _count_tokens(packed[0]) <= max_tokens:
            return packed

    sentence_chunks = [chunk for chunk in re.split(r'(?<=[.!?])\s+', text) if chunk.strip()]
    if len(sentence_chunks) > 1:
        packed = _pack_units(sentence_chunks, ' ')
        if len(packed) > 1 or _count_tokens(packed[0]) <= max_tokens:
            return packed

    line_chunks = [chunk for chunk in text.splitlines() if chunk.strip()]
    if len(line_chunks) > 1:
        packed = _pack_units(line_chunks, '\n')
        if len(packed) > 1 or _count_tokens(packed[0]) <= max_tokens:
            return packed

    words = text.split()
    if not words:
        return [text]

    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _count_tokens(word)
        if current and current_tokens + word_tokens > max_tokens:
            packed.append(' '.join(current).strip())
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        packed.append(' '.join(current).strip())
    return [chunk for chunk in packed if chunk]


# ==============================================================================
# Upload endpoint
# ==============================================================================

@router.post('/api/translate/upload')
async def upload_translate_file(
    file: UploadFile = File(...),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    filename = _sanitize_filename(file.filename or 'upload')
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail='Uploaded file is empty.')
    if len(raw) > _MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f'File exceeds {_MAX_UPLOAD_MB} MB limit.')

    upload_id = str(uuid.uuid4())
    file_dir = _upload_dir(upload_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir / filename
    file_path.write_bytes(raw)

    scanned = scanned_file_for_path(file_path)
    if scanned is None:
        raise HTTPException(status_code=422, detail='Unable to process file for indexing.')

    result, indexed = await index_uploaded_file(
        db,
        scanned,
        source_provider=TRANSLATE_PROVIDER,
        entity_type=TRANSLATE_ENTITY_TYPE,
    )
    if not result.success:
        raise HTTPException(status_code=422, detail=f'Indexing failed: {result.error or "unknown"}')

    if indexed is None or indexed.id is None:
        raise HTTPException(status_code=500, detail='File indexed but record not found.')

    page_count = getattr(indexed, 'page_count', None)

    log.info('translate_upload_indexed', file_id=indexed.id, filename=filename)
    return {
        'file_id': indexed.id,
        'filename': filename,
        'page_count': page_count,
        'size_bytes': len(raw),
    }


@router.delete('/api/translate/upload/{file_id}')
async def delete_translate_upload(
    file_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    indexed = await get_file_by_id(db, file_id)
    if not indexed:
        raise HTTPException(status_code=404, detail='File not found.')
    if getattr(indexed, 'source_provider', '') != TRANSLATE_PROVIDER:
        raise HTTPException(status_code=403, detail='Not a translate upload.')
    upload_dir = Path(indexed.path).parent
    await remove_file(db, indexed)
    shutil.rmtree(upload_dir, ignore_errors=True)
    log.info('translate_upload_deleted', file_id=file_id)
    return {'deleted': True}


# ==============================================================================
# Job endpoints
# ==============================================================================

@router.post('/api/translate/jobs')
async def create_translate_job_endpoint(
    body: dict,
    background_tasks: BackgroundTasks,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    raise_if_llm_busy(_translate_lock.locked())

    file_id = int(body.get('file_id') or 0)
    target_language = normalize_translate_language(str(body.get('target_language') or 'Spanish').strip())
    tone = str(body.get('tone') or 'natural').strip()

    if not file_id:
        raise HTTPException(status_code=400, detail='file_id is required.')
    if tone not in TONE_INSTRUCTIONS:
        raise HTTPException(status_code=400, detail=f'Invalid tone. Choose: {list(TONE_INSTRUCTIONS)}')

    indexed = await get_file_by_id(db, file_id)
    if not indexed:
        raise HTTPException(status_code=404, detail='File not found.')

    job_id = str(uuid.uuid4())
    await create_translate_job(
        db, job_id=job_id, file_id=file_id,
        target_language=target_language, tone=tone,
    )

    _job_queues[job_id] = asyncio.Queue()
    _job_cancel_events[job_id] = asyncio.Event()
    background_tasks.add_task(_run_translate_job, job_id, file_id, target_language, tone)

    log.info('translate_job_created', job_id=job_id, file_id=file_id, language=target_language)
    return {'job_id': job_id}


@router.get('/api/translate/jobs/{job_id}')
async def get_translate_job_status(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    row = await get_translate_job(db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail='Job not found.')
    return dict(row)


@router.delete('/api/translate/jobs/{job_id}')
async def cancel_translate_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Guard: don't overwrite a job that already completed or failed.
    row = await get_translate_job(db, job_id)
    if row and str(row['status']) in ('done', 'failed'):
        return {'cancelled': False}
    # Mark cancelled in DB so the inter-section check stops the loop
    await update_translate_job(db, job_id, status='stalled', error=TRANSLATE_CANCEL_ERROR_TOKEN)
    # Immediately cancel the active generate_stream (releases LLM lock now)
    cancel_ev = _job_cancel_events.get(job_id)
    if cancel_ev:
        cancel_ev.set()
    # Signal the SSE stream to close
    q = _job_queues.get(job_id)
    if q:
        await q.put(_SENTINEL)

    # User-facing activity log entry. Distinguishable from a timeout stall
    # by the 'cancelled' wording in the message.
    if row:
        try:
            file_id           = int(row['file_id'])
            target_language   = str(row['target_language'])
            completed_count   = int(row['completed_sections'] or 0)
            section_count     = int(row['section_count'] or 0)
            file_row          = await get_file_by_id(db, file_id)
            fname             = file_row.filename if file_row else f'file #{file_id}'
            sections_note     = f' · {completed_count}/{section_count} sections completed' if section_count else ''
            await emit_log_event(
                event_name='translate_job_cancelled',
                source='translate',
                message=f'Translation cancelled: \'{fname}\' → {target_language}{sections_note}',
                details={
                    'job_id':              job_id,
                    'language':            target_language,
                    'sections_completed':  completed_count,
                    'sections_total':      section_count,
                    'cancelled_by_user':   True,
                    'cancel_error_token':  TRANSLATE_CANCEL_ERROR_TOKEN,
                },
                file_id=file_id,
                db=db,
            )
        except Exception:
            pass  # activity log is non-critical; never fail cancel
    return {'cancelled': True}


@router.get('/api/translate/jobs/{job_id}/events')
async def translate_job_events(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> EventSourceResponse:
    row = await get_translate_job(db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail='Job not found.')

    async def _stream():
        # Replay already-completed sections so reconnecting clients catch up.
        sections = await get_translate_sections(db, job_id)
        job_status = str(row['status'])

        if row['glossary_json']:
            try:
                terms = json.loads(str(row['glossary_json']))
                yield {'event': 'glossary_done', 'data': json.dumps({'term_count': len(terms)})}
            except Exception:
                pass

        if row['section_count']:
            yield {'event': 'sections_ready', 'data': json.dumps({'section_count': int(row['section_count'])})}

        for s in sections:
            if str(s['status']) == 'done':
                yield {
                    'event': 'section_done',
                    'data': json.dumps({
                        'section_index': int(s['section_index']),
                        'section_title': s['section_title'],
                        'text': s['result_text'],
                    }),
                }

        # If already terminal, close immediately.
        if job_status in ('done', 'failed', 'stalled'):
            if _is_cancelled_translate_row(row):
                event_name = 'job_cancelled'
            else:
                event_name = 'job_done' if job_status == 'done' else f'job_{job_status}'
            yield {'event': event_name, 'data': json.dumps({})}
            return

        # Wait for new events from the worker.
        q = _job_queues.get(job_id)
        if q is None:
            return

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
            except TimeoutError:
                yield {'event': 'ping', 'data': '{}'}
                continue
            if msg is _SENTINEL:
                break
            yield msg

    return EventSourceResponse(_stream())


@router.get('/api/translate/jobs/{job_id}/result')
async def get_translate_result(
    job_id: str,
    format: str = 'markdown',
    db: aiosqlite.Connection = Depends(get_db),
) -> PlainTextResponse:
    row = await get_translate_job(db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail='Job not found.')
    parts = await get_translate_job_result(db, job_id)
    text = '\n\n'.join(parts)
    if format == 'text':
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'[*_]{1,2}(.+?)[*_]{1,2}', r'\1', text)
    return PlainTextResponse(text, media_type='text/plain; charset=utf-8')


# ==============================================================================
# Estimate endpoint (Phase 6)
# ==============================================================================

@router.post('/api/translate/jobs/estimate')
async def estimate_translate_job(
    body: dict,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    file_id = int(body.get('file_id') or 0)
    if not file_id:
        raise HTTPException(status_code=400, detail='file_id is required.')
    indexed = await get_file_by_id(db, file_id)
    if not indexed:
        raise HTTPException(status_code=404, detail='File not found.')

    page_count = int(getattr(indexed, 'page_count', 0) or 0)
    # Fall back to token-based estimate when page_count is unavailable (e.g. EPUB)
    if page_count > 0:
        token_estimate = page_count * TRANSLATE_AVG_TOKENS_PER_PAGE
    else:
        # Query actual token count from chunks table
        cur = await db.execute(
            'SELECT COALESCE(SUM(token_count), 0) FROM chunks WHERE file_id = ? AND parent_id IS NULL',
            (file_id,),
        )
        row = await cur.fetchone()
        token_estimate = int(row[0]) if row else 0
        page_count = max(1, token_estimate // TRANSLATE_AVG_TOKENS_PER_PAGE)

    section_count_estimate = max(1, -(-token_estimate // TRANSLATE_BATCH_TARGET_TOKENS))  # ceiling div
    estimated_minutes = round((section_count_estimate * TRANSLATE_AVG_SECTION_SECONDS) / 60, 1)
    exceeds_soft_limit = page_count > TRANSLATE_SOFT_PAGE_LIMIT

    # Soft limit: also flag by section count for files without reliable page_count
    exceeds_soft_limit = exceeds_soft_limit or section_count_estimate > TRANSLATE_SOFT_SECTION_LIMIT

    return {
        'page_count': page_count,
        'token_estimate': token_estimate,
        'section_count_estimate': section_count_estimate,
        'estimated_minutes': estimated_minutes,
        'exceeds_soft_limit': exceeds_soft_limit,
    }


# ==============================================================================
# Background worker — runs in FastAPI background task
# ==============================================================================

async def _run_translate_job(job_id: str, file_id: int, target_language: str, tone: str) -> None:
    db = None
    try:
        from informity.db.sqlite import get_connection
        db = await get_connection()

        cancel_event = _job_cancel_events.get(job_id)

        # Fast-fail if another job holds the lock. The endpoint-level .locked()
        # check has a TOCTOU window: two near-simultaneous requests can both pass
        # the check before either background task starts. Re-checking here (before
        # any yields inside the lock block) closes that window — once this task
        # resumes from `await get_connection()` above, the event loop is single-
        # threaded and won't context-switch between this check and the acquire.
        if _translate_lock.locked():
            log.warning('translate_job_lock_busy', job_id=job_id)
            if db:
                await update_translate_job(db, job_id, status='failed', error=LLM_BUSY_DETAIL)
            await _emit(job_id, 'job_failed', {'error': LLM_BUSY_DETAIL})
            return

        async with asyncio.timeout(TRANSLATE_JOB_MAX_RUNTIME_S):
            async with _translate_lock:
                await update_translate_job(db, job_id, status='running')

                # --- Phase 2: Glossary extraction ---
                glossary_json = await _extract_glossary(db, file_id, target_language, cancel_event)
                if glossary_json:
                    await update_translate_job(db, job_id, glossary_json=glossary_json)
                try:
                    term_count = len(json.loads(glossary_json)) if glossary_json else 0
                except Exception:
                    term_count = 0
                await _emit(job_id, 'glossary_done', {'term_count': term_count})

                # --- Phase 3: Section-aware chunking ---
                sections = await _build_sections(db, file_id)
                await update_translate_job(db, job_id, section_count=len(sections))
                for s_idx, s in enumerate(sections):
                    await create_translate_section(
                        db,
                        section_id=str(uuid.uuid4()),
                        job_id=job_id,
                        section_index=s_idx,
                        section_title=s.get('title'),
                    )
                await db.commit()
                await _emit(job_id, 'sections_ready', {'section_count': len(sections)})

                # Fetch section rows (with IDs) for update calls
                section_rows = await get_translate_sections(db, job_id)

                # --- Phase 4: Translation loop ---
                completed = 0
                failed = 0
                truncated = 0
                job_loop_start = time.monotonic()
                # Stall deadline resets at each section START (not just section done).
                # This means: "no section has even begun in TRANSLATE_JOB_STALL_S seconds".
                # An active section translating for up to TRANSLATE_SECTION_TIMEOUT_S
                # does NOT trigger a stall — the deadline advances when the section starts.
                stall_deadline = time.monotonic() + TRANSLATE_JOB_STALL_S
                glossary_block = _build_glossary_block(glossary_json)

                for s_idx, (section, row) in enumerate(zip(sections, section_rows, strict=True)):
                    # Check for user cancellation between sections
                    job_row = await get_translate_job(db, job_id)
                    if _is_cancelled_translate_row(job_row):
                        log.info('translate_job_cancelled', job_id=job_id, section=s_idx)
                        return

                    section_id = str(row['section_id'])
                    # Reset stall deadline at section start — active work is not a stall
                    stall_deadline = time.monotonic() + TRANSLATE_JOB_STALL_S
                    await update_translate_section(db, section_id, status='running')
                    await db.commit()
                    await _emit(job_id, 'section_started', {
                        'section_index': s_idx,
                        'section_title': section.get('title'),
                    })

                    translated = None
                    last_error = None
                    last_finish_reason: str | None = None
                    total_attempts = 0
                    section_start_ms = time.monotonic()
                    for attempt in range(TRANSLATE_SECTION_RETRY_MAX + 1):
                        total_attempts += 1
                        token_cap = TRANSLATE_RETRY_TOKEN_CAP if attempt > 0 else TRANSLATE_BATCH_TARGET_TOKENS
                        # Use 1.6× expansion ratio — academic English→Spanish expands 50-60%.
                        # The previous 1.35× caused frequent mid-sentence truncation.
                        max_out = int(token_cap * 1.6) + 100
                        source = section['text']
                        if _count_tokens(source) > token_cap:
                            words = source.split()
                            cap_words = max(1, int(token_cap / 1.3))
                            source = ' '.join(words[:cap_words])
                        try:
                            # _translate_section uses generate_stream internally, so it has
                            # a real cancel_event-backed timeout — no zombie threads.
                            translated, last_finish_reason = await _translate_section(
                                source, target_language, tone, glossary_block, max_out,
                                cancel_event=cancel_event,
                            )
                            if translated:
                                # Accept truncated output as-is. A retry uses the smaller
                                # TRANSLATE_RETRY_TOKEN_CAP and would only produce a shorter
                                # truncated result — counterproductive. Retry only fires for
                                # genuine error/empty cases (no `translated`) below.
                                if last_finish_reason == 'length' and attempt == 0:
                                    log.warning('translate_section_truncated_accepted',
                                                job_id=job_id, section=s_idx,
                                                chars_output=len(translated))
                                break
                        except Exception as exc:
                            last_error = str(exc)
                            log.warning('translate_section_attempt_failed',
                                        job_id=job_id, section=s_idx, attempt=attempt, error=last_error)
                            await _emit(job_id, 'section_retry', {
                                'section_index': s_idx, 'attempt': attempt + 1, 'error': last_error,
                            })

                    section_elapsed_ms = int((time.monotonic() - section_start_ms) * 1000)
                    section_truncated = last_finish_reason == 'length'
                    if section_truncated:
                        truncated += 1

                    await update_translate_section(db, section_id, attempt_count=total_attempts)

                    if translated:
                        await update_translate_section(db, section_id, status='done', result_text=translated)
                        completed += 1
                        await update_translate_job(db, job_id, completed_sections=completed)
                        await db.commit()
                        log.info('translate_section_completed',
                                 job_id=job_id, section_index=s_idx,
                                 section_title=section.get('title'),
                                 status='done', finish_reason=last_finish_reason,
                                 truncated=section_truncated,
                                 chars_output=len(translated),
                                 duration_ms=section_elapsed_ms,
                                 attempt_count=total_attempts)
                        await _emit(job_id, 'section_done', {
                            'section_index': s_idx,
                            'section_title': section.get('title'),
                            'text': translated,
                        })
                    else:
                        await update_translate_section(db, section_id, status='failed', error=last_error or 'unknown')
                        failed += 1
                        await update_translate_job(db, job_id, failed_sections=failed)
                        await db.commit()
                        log.warning('translate_section_completed',
                                    job_id=job_id, section_index=s_idx,
                                    section_title=section.get('title'),
                                    status='failed', finish_reason=last_finish_reason,
                                    truncated=section_truncated,
                                    chars_output=0,
                                    duration_ms=section_elapsed_ms,
                                    attempt_count=total_attempts,
                                    error=last_error)
                        await _emit(job_id, 'section_failed', {
                            'section_index': s_idx, 'error': last_error or 'Translation failed',
                        })

                    # Stall check
                    if time.monotonic() > stall_deadline:
                        await update_translate_job(db, job_id, status='stalled',
                                                   error='No progress within stall window.')
                        await _emit(job_id, 'job_stalled', {})
                        await emit_log_event(
                            event_name='translate_job_stalled',
                            source='translate',
                            message=f'Translation stalled: {completed}/{len(sections)} sections completed',
                            file_id=file_id, db=db,
                        )
                        return

                final_status = 'done' if completed > 0 else 'failed'
                await update_translate_job(db, job_id, status=final_status)
                event = 'job_done' if final_status == 'done' else 'job_failed'
                await _emit(job_id, event, {'completed_sections': completed, 'failed_sections': failed})
                job_elapsed_s = int(time.monotonic() - job_loop_start)
                elapsed_str = (
                    f'{job_elapsed_s // 60}m {job_elapsed_s % 60}s'
                    if job_elapsed_s >= 60 else f'{job_elapsed_s}s'
                )
                log.info('translate_job_completed',
                         job_id=job_id, language=target_language, tone=tone,
                         status=final_status,
                         sections_total=len(sections),
                         sections_completed=completed,
                         sections_failed=failed,
                         sections_truncated=truncated,
                         truncation_rate=round(truncated / max(len(sections), 1), 3),
                         glossary_terms=term_count,
                         total_elapsed_ms=int(job_elapsed_s * 1000))

                # Fetch filename for a human-readable activity log message
                try:
                    file_row = await get_file_by_id(db, file_id)
                    fname = file_row.filename if file_row else f'file #{file_id}'
                except Exception:
                    fname = f'file #{file_id}'
                trunc_note = f' · {truncated} truncated' if truncated else ''
                if final_status == 'done':
                    await emit_log_event(
                        event_name='translate_job_completed',
                        source='translate',
                        message=(
                            f'Translated \'{fname}\' → {target_language} · {capitalize(tone)} tone'
                            f' · {completed}/{len(sections)} sections · {elapsed_str}{trunc_note}'
                        ),
                        details={
                            'job_id': job_id,
                            'language': target_language,
                            'tone': tone,
                            'sections_completed': completed,
                            'sections_total': len(sections),
                            'sections_truncated': truncated,
                            'elapsed_s': job_elapsed_s,
                        },
                        file_id=file_id,
                        db=db,
                    )
                else:
                    await emit_log_event(
                        event_name='translate_job_failed',
                        source='translate',
                        message=(
                            f'Translation failed: \'{fname}\' → {target_language}'
                            f' · {completed}/{len(sections)} sections completed'
                        ),
                        file_id=file_id,
                        db=db,
                    )

    except TimeoutError:
        log.error('translate_job_hard_timeout', job_id=job_id)
        if db:
            await update_translate_job(db, job_id, status='stalled', error='Job exceeded maximum runtime.')
        await _emit(job_id, 'job_stalled', {'error': 'Job exceeded maximum runtime.'})
    except Exception as exc:
        log.error('translate_job_error', job_id=job_id, error=str(exc), exc_info=True)
        if db:
            await update_translate_job(db, job_id, status='failed', error=str(exc))
        await _emit(job_id, 'job_failed', {'error': str(exc)})
    finally:
        if db:
            await db.close()
        _job_cancel_events.pop(job_id, None)
        q = _job_queues.pop(job_id, None)
        if q:
            await q.put(_SENTINEL)


async def _extract_glossary(
    db: aiosqlite.Connection,
    file_id: int,
    target_language: str,
    cancel_event: asyncio.Event | None = None,
) -> str | None:
    """
    Extract a translation glossary using generate_stream with a simple line-delimited format.

    Avoids JSON parsing entirely — asks the model for "term = translation" lines, which
    are far more reliably produced in free-form generation than structured JSON.
    Uses generate_stream to avoid zombie C++ threads (no asyncio.to_thread wrapper).
    """
    cursor = await db.execute(
        'SELECT content FROM chunks WHERE file_id = ? ORDER BY chunk_index ASC LIMIT 10',
        (file_id,),
    )
    rows = await cursor.fetchall()
    combined = '\n\n'.join(str(r['content']) for r in rows)
    words = combined.split()
    cap = int(TRANSLATE_GLOSSARY_INPUT_TOKENS / 1.3)
    source_sample = ' '.join(words[:cap])

    messages = [
        {
            'role': 'system',
            'content': (
                f'Extract up to {TRANSLATE_GLOSSARY_TERM_COUNT} key technical terms, proper nouns, '
                f'and domain-specific phrases from the text below and translate each to {target_language}. '
                'Output ONLY the list, one term per line, in this exact format:\n'
                'term = translation\n'
                'No numbering, no bullets, no preamble, no explanation.'
            ),
        },
        {'role': 'user', 'content': source_sample},
    ]

    parts: list[str] = []
    try:
        gen = llm_engine.generate_stream(
            messages,
            max_tokens=TRANSLATE_GLOSSARY_MAX_TOKENS,
            temperature=TRANSLATE_GLOSSARY_TEMPERATURE,
            timeout_seconds=float(TRANSLATE_GLOSSARY_TIMEOUT_S),
        )
        async for item in gen:
            if cancel_event and cancel_event.is_set():
                await gen.aclose()
                return None
            if isinstance(item, tuple):
                # Tuple signals end-of-stream from generate_stream; the first element
                # is the final token (or '__timeout__'). Append if valid, then stop.
                token, _ = item
                if isinstance(token, str) and token and token != '__timeout__':
                    parts.append(token)
                break
            elif isinstance(item, str):
                parts.append(item)

        content = ''.join(parts).strip()
        terms = []
        for line in content.splitlines():
            line = line.strip().lstrip('•-*0123456789. ')
            if '=' in line:
                halves = line.split('=', 1)
                source = halves[0].strip().strip('"\'')
                translation = halves[1].strip().strip('"\'')
                if source and translation and len(source) < 80:
                    terms.append({'source': source, 'translation': translation})
        if terms:
            log.debug('translate_glossary_extracted', file_id=file_id, term_count=len(terms))
            return json.dumps(terms)
    except Exception as exc:
        log.warning('translate_glossary_failed', file_id=file_id, error=str(exc))
    return None


async def _build_sections(db: aiosqlite.Connection, file_id: int) -> list[dict]:
    # Group chunks by section_path to create natural sections.
    cursor = await db.execute(
        '''
        SELECT content, section_path, chunk_index
        FROM chunks
        WHERE file_id = ? AND parent_id IS NULL
        ORDER BY chunk_index ASC
        ''',
        (file_id,),
    )
    rows = await cursor.fetchall()

    if not rows:
        return []

    sections: list[dict] = []
    current_path: str | None = None
    current_chunks: list[str] = []
    current_title: str | None = None

    def _path_to_title(path: str | None) -> str | None:
        if not path:
            return None
        parts = str(path).split('/')
        return parts[-1].strip() or None

    def _first_sentence_label(text: str) -> str:
        """
        Derive a short title from the first sentence of a headerless section.
        Takes up to 60 chars, trimming at the last word boundary, with an ellipsis.
        Used as a fallback when Docling detected no section header (e.g. academic
        PDFs, EPUBs, plain-text files).
        """
        first_line = text.strip().split('\n')[0].strip()
        if len(first_line) <= 60:
            return first_line
        trimmed = first_line[:60].rsplit(' ', 1)[0]
        return f'{trimmed}…'

    def _flush():
        if current_chunks:
            text = '\n\n'.join(current_chunks)
            # Fall back to first-sentence label when Docling found no section header.
            title = current_title or _first_sentence_label(text)
            # If section text exceeds budget, split it using progressively
            # smaller structural boundaries so flat documents still translate
            # in multiple chunks.
            text_parts = _split_text_for_translation(text, TRANSLATE_BATCH_TARGET_TOKENS)
            if not text_parts:
                return
            if len(text_parts) == 1:
                sections.append({'title': title, 'text': text_parts[0]})
                return
            for part, section_text in enumerate(text_parts, start=1):
                sections.append(
                    {
                        'title': f'{title} ({part})',
                        'text': section_text,
                    }
                )

    for row in rows:
        path = str(row['section_path'] or '')
        content = str(row['content'] or '')
        if path != current_path:
            _flush()
            current_path = path
            current_chunks = [content]
            current_title = _path_to_title(path)
        else:
            current_chunks.append(content)

    _flush()
    return sections if sections else [{'title': None, 'text': '\n\n'.join(str(r['content']) for r in rows)}]


async def _translate_section(
    source_text: str,
    target_language: str,
    tone: str,
    glossary_block: str,
    max_tokens: int,
    cancel_event: asyncio.Event | None = None,
) -> tuple[str | None, str | None]:
    """
    Async translation using generate_stream.

    generate_stream uses an internal wall_clock + cancel_event mechanism that
    actually stops the C++ generation thread on timeout — unlike the previous
    asyncio.wait_for(asyncio.to_thread(chat_complete)) pattern which left zombie
    threads blocking subsequent calls.
    """
    tone_instr = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS['natural']).format(language=target_language)
    system = (
        f'You are a professional translator. Translate the following text to {target_language}. '
        f'{tone_instr} '
        'Preserve all Markdown formatting: headers (#, ##, ###), bold (**text**), italic (*text*), '
        'bullet lists (- item), numbered lists (1. item), and code blocks (```). '
        'For tables: reproduce them as valid GFM Markdown tables with a header row, a separator row '
        '(|---|---|), and data rows. If the source table is ambiguous or malformed, render its '
        'content as a bulleted list instead. Never output a standalone separator line (---|---) '
        'without surrounding table rows — an orphaned separator line is a formatting error. '
        'Output ONLY the translated text. No commentary, no preamble, no explanations.'
    )
    if glossary_block:
        system += glossary_block

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': source_text},
    ]

    parts: list[str] = []
    finish_reason: str | None = None
    gen = llm_engine.generate_stream(
        messages,
        max_tokens=max_tokens,
        temperature=TRANSLATE_TEMPERATURE,
        timeout_seconds=float(TRANSLATE_SECTION_TIMEOUT_S),
    )
    try:
        async for item in gen:
            # Honour immediate cancellation — closes the generator which
            # sets the engine's internal cancel_event, releasing the LLM.
            if cancel_event and cancel_event.is_set():
                await gen.aclose()
                return None, 'cancelled'
            if isinstance(item, tuple):
                # Tuple signals end-of-stream; first element is final token or '__timeout__'
                token, meta = item
                if token == '__timeout__':
                    finish_reason = 'timeout'
                else:
                    finish_reason = (meta or {}).get('finish_reason', 'stop') if isinstance(meta, dict) else 'stop'
                    if isinstance(token, str) and token:
                        parts.append(token)
                break
            elif isinstance(item, str):
                parts.append(item)
    except Exception as exc:
        raise RuntimeError(f'Translation stream error: {exc}') from exc

    text = ''.join(parts).strip() or None
    return text, finish_reason
