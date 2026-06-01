# ==============================================================================
# Informity AI — Translation Routes (v3)
# Upload, job management, SSE streaming, result export.
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

import aiosqlite
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from informity.config import settings
from informity.db.sqlite import (
    create_translate_job,
    create_translate_section,
    get_db,
    get_file_by_id,
    get_file_by_path,
    get_translate_job,
    get_translate_job_result,
    get_translate_sections,
    update_translate_job,
    update_translate_section,
)
from informity.indexer.pipeline import index_file, remove_file
from informity.llm.engine import llm_engine
from informity.scanner.crawler import scanned_file_for_path
from informity.translate_policy import (
    TONE_INSTRUCTIONS,
    TRANSLATE_BATCH_TARGET_TOKENS,
    TRANSLATE_CALL_MAX_TOKENS,
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
    TRANSLATE_STORAGE_DIRNAME,
    TRANSLATE_TEMPERATURE,
    TRANSLATE_AVG_SECTION_SECONDS,
    TRANSLATE_AVG_TOKENS_PER_PAGE,
)

log = structlog.get_logger(__name__)
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
    except Exception:
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
    except Exception:
        return ''


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

    result = await index_file(db, scanned, source_provider=TRANSLATE_PROVIDER, entity_type=TRANSLATE_ENTITY_TYPE)
    if not result.success:
        raise HTTPException(status_code=422, detail=f'Indexing failed: {result.error or "unknown"}')

    # IndexResult has no file_id field — fetch the record by path after indexing.
    indexed = await get_file_by_path(db, str(file_path))
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
    await remove_file(db, indexed)
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
    if _translate_lock.locked():
        raise HTTPException(status_code=409, detail='A translation is already in progress.')

    file_id = int(body.get('file_id') or 0)
    target_language = str(body.get('target_language') or 'Spanish').strip()
    tone = str(body.get('tone') or 'natural').strip()
    output_mode = str(body.get('output_mode') or 'markdown').strip()

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
        target_language=target_language, tone=tone, output_mode=output_mode,
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
    await update_translate_job(db, job_id, status='stalled', error='Cancelled by user')
    # Immediately cancel the active generate_stream (releases LLM lock now)
    cancel_ev = _job_cancel_events.get(job_id)
    if cancel_ev:
        cancel_ev.set()
    # Signal the SSE stream to close
    q = _job_queues.get(job_id)
    if q:
        await q.put(_SENTINEL)
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
            yield {'event': job_status == 'done' and 'job_done' or f'job_{job_status}', 'data': json.dumps({})}
            return

        # Wait for new events from the worker.
        q = _job_queues.get(job_id)
        if q is None:
            return

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
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

    # Soft limit: also flag by token count for files without reliable page_count
    exceeds_soft_limit = exceeds_soft_limit or section_count_estimate > 25

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
    import time
    db = None
    try:
        from informity.db.sqlite import get_connection
        db = await get_connection()

        cancel_event = _job_cancel_events.get(job_id)

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
                # Stall deadline resets at each section START (not just section done).
                # This means: "no section has even begun in TRANSLATE_JOB_STALL_S seconds".
                # An active section translating for up to TRANSLATE_SECTION_TIMEOUT_S
                # does NOT trigger a stall — the deadline advances when the section starts.
                stall_deadline = time.monotonic() + TRANSLATE_JOB_STALL_S
                glossary_block = _build_glossary_block(glossary_json)

                for s_idx, (section, row) in enumerate(zip(sections, section_rows)):
                    # Check for user cancellation between sections
                    job_row = await get_translate_job(db, job_id)
                    if job_row and str(job_row['status']) == 'stalled':
                        log.info('translate_job_cancelled', job_id=job_id, section=s_idx)
                        return

                    section_id = str(row['section_id'])
                    # Reset stall deadline at section start — active work is not a stall
                    stall_deadline = time.monotonic() + TRANSLATE_JOB_STALL_S
                    await update_translate_section(db, section_id, status='running')
                    await _emit(job_id, 'section_started', {
                        'section_index': s_idx,
                        'section_title': section.get('title'),
                    })

                    translated = None
                    last_error = None
                    for attempt in range(TRANSLATE_SECTION_RETRY_MAX + 1):
                        token_cap = TRANSLATE_RETRY_TOKEN_CAP if attempt > 0 else TRANSLATE_BATCH_TARGET_TOKENS
                        max_out   = int(token_cap * 1.35) + 100  # proportional output cap
                        source = section['text']
                        if _count_tokens(source) > token_cap:
                            words = source.split()
                            cap_words = max(1, int(token_cap / 1.3))
                            source = ' '.join(words[:cap_words])
                        try:
                            # _translate_section uses generate_stream internally, so it has
                            # a real cancel_event-backed timeout — no zombie threads.
                            translated = await _translate_section(
                                source, target_language, tone, glossary_block, max_out,
                                cancel_event=cancel_event,
                            )
                            if translated:
                                break
                        except Exception as exc:
                            last_error = str(exc)
                            log.warning('translate_section_attempt_failed',
                                        job_id=job_id, section=s_idx, attempt=attempt, error=last_error)
                            await _emit(job_id, 'section_retry', {
                                'section_index': s_idx, 'attempt': attempt + 1, 'error': last_error,
                            })

                    await update_translate_section(db, section_id, attempt_count=attempt + 1)

                    if translated:
                        await update_translate_section(db, section_id, status='done', result_text=translated)
                        completed += 1
                        await update_translate_job(db, job_id, completed_sections=completed)
                        await db.commit()
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
                        await _emit(job_id, 'section_failed', {
                            'section_index': s_idx, 'error': last_error or 'Translation failed',
                        })

                    # Stall check
                    if time.monotonic() > stall_deadline:
                        await update_translate_job(db, job_id, status='stalled',
                                                   error='No progress within stall window.')
                        await _emit(job_id, 'job_stalled', {})
                        return

                final_status = 'done' if completed > 0 else 'failed'
                await update_translate_job(db, job_id, status=final_status)
                event = 'job_done' if final_status == 'done' else 'job_failed'
                await _emit(job_id, event, {'completed_sections': completed, 'failed_sections': failed})

    except asyncio.TimeoutError:
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

    def _flush():
        if current_chunks:
            text = '\n\n'.join(current_chunks)
            # If section text exceeds budget, split at paragraph boundaries
            token_count = _count_tokens(text)
            if token_count <= TRANSLATE_BATCH_TARGET_TOKENS:
                sections.append({'title': current_title, 'text': text})
            else:
                # Split by paragraphs into sub-sections
                paragraphs = text.split('\n\n')
                sub: list[str] = []
                sub_tokens = 0
                part = 0
                for para in paragraphs:
                    pt = _count_tokens(para)
                    if sub_tokens + pt > TRANSLATE_BATCH_TARGET_TOKENS and sub:
                        sections.append({'title': f'{current_title} ({part + 1})' if current_title else None,
                                         'text': '\n\n'.join(sub)})
                        sub = []
                        sub_tokens = 0
                        part += 1
                    sub.append(para)
                    sub_tokens += pt
                if sub:
                    sections.append({'title': f'{current_title} ({part + 1})' if current_title and part > 0 else current_title,
                                     'text': '\n\n'.join(sub)})

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
) -> str | None:
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
        'Preserve all Markdown formatting: headers (#, ##, ###), bold (**), italic (*), lists, and tables. '
        'Output ONLY the translated text. No commentary, no preamble, no explanations.'
    )
    if glossary_block:
        system += glossary_block

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': source_text},
    ]

    parts: list[str] = []
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
                return None
            if isinstance(item, tuple):
                token, _ = item
                if isinstance(token, str) and token and token != '__timeout__':
                    parts.append(token)
                break
            elif isinstance(item, str):
                parts.append(item)
    except Exception as exc:
        raise RuntimeError(f'Translation stream error: {exc}') from exc

    return ''.join(parts).strip() or None
