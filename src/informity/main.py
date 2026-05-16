# ==============================================================================
# Informity AI — FastAPI Application Entry Point
# Creates the app, registers routers, configures middleware, and manages
# the application lifespan (DB init, directory creation, clean shutdown).
# ==============================================================================
# ruff: noqa: E402

# CPU thread limits are applied by config._apply_thread_limits_early() at import time.
# Do not duplicate them here — config.py is the single source of truth.
import os as _os

# Suppress SyntaxWarnings from third-party libraries (e.g., pysbd) before any imports
# These warnings are emitted at import time and are harmless
# Python 3.13 is stricter about escape sequences, causing warnings in third-party code
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

import asyncio
import atexit
import multiprocessing
import signal
import time
import types
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from informity.api.routes_chat import router as chat_router
from informity.api.routes_index import router as index_router
from informity.api.routes_scan import router as scan_router
from informity.api.routes_search import router as search_router
from informity.api.routes_settings import router as settings_router
from informity.api.routes_system import router as system_router
from informity.api.schemas import HealthResponse
from informity.api.security import (
    TAURI_SESSION_HEADER,
    get_cors_allow_origins,
    get_tauri_session_token_from_env,
    is_loopback_host,
    is_tauri_desktop_mode,
    is_tauri_session_authorized,
)
from informity.config import APP_DISPLAY_NAME, configure_hf_environment, settings
from informity.version import APP_VERSION

# Set Hugging Face cache paths and offline flags before importing models.
# Allow boot into first-run setup when local models are not cached yet.
configure_hf_environment(fail_on_missing_full_privacy_models=False)

from informity.db.sqlite import (
    clear_stale_running_scans,
    get_connection,
    init_db,
    prune_continuation_artifacts,
)
from informity.indexer.adaptive_tuning import update_tuning_cache
from informity.indexer.embedder import embedder
from informity.indexer.reranker import reranker
from informity.llm.engine import llm_engine, remove_models_dir_cache
from informity.logging_config import configure_logging
from informity.mcp.lifecycle import mcp_lifecycle
from informity.scanner.watcher import start_watcher, stop_watcher

# ==============================================================================
# Initialize Logging
# ==============================================================================
# Configure logging BEFORE creating any loggers. This ensures all logs
# (including from imported modules) go to both console and files.

configure_logging()

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_STARTUP_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, TimeoutError)
_REQUEST_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, TimeoutError)
_WARMUP_TIMEOUT_SECONDS = 300.0
_TAURI_SESSION_TOKEN = get_tauri_session_token_from_env()
_DESKTOP_SESSION_MODE = is_tauri_desktop_mode(_TAURI_SESSION_TOKEN)
_MANAGED_PID_FILE_ENV = 'INFORMITY_MANAGED_PID_FILE'
_MANAGED_PID_FILE_RAW = _os.environ.get(_MANAGED_PID_FILE_ENV, '').strip()
_MANAGED_PID_FILE_PATH: Path | None = (
    Path(_MANAGED_PID_FILE_RAW).expanduser() if _MANAGED_PID_FILE_RAW else None
)

# ==============================================================================
# Process Cleanup
# ==============================================================================

def _cleanup_models() -> None:
    # Unload models to release resources.
    # Used on normal shutdown (lifespan/atexit) and on SIGTERM/SIGINT (reload child).
    embedder.unload()
    reranker.unload()


def _write_managed_pid_file() -> None:
    if _MANAGED_PID_FILE_PATH is None:
        return
    try:
        _MANAGED_PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MANAGED_PID_FILE_PATH.write_text(f'{_os.getpid()}\n', encoding='utf-8')
    except OSError as exc:
        log.warning(
            'managed_pid_file_write_failed',
            path=str(_MANAGED_PID_FILE_PATH),
            error=str(exc),
        )


def _remove_managed_pid_file() -> None:
    if _MANAGED_PID_FILE_PATH is None:
        return
    with suppress(OSError):
        _MANAGED_PID_FILE_PATH.unlink(missing_ok=True)


def _kill_child_processes() -> None:
    # Kill child processes known to multiprocessing (tokenizers, embedder workers).
    # We do not use killpg(process_group, SIGTERM) because we are in that group;
    # that would signal ourselves and trigger _signal_cleanup during lifespan.
    active_children = multiprocessing.active_children()
    for child in active_children:
        log.debug('terminating_child_process', pid=child.pid, name=child.name)
        with suppress(OSError):
            child.terminate()

    for child in active_children:
        try:
            child.join(timeout=2)
            if child.is_alive():
                log.warning('force_killing_child', pid=child.pid, name=child.name)
                child.kill()
        except (OSError, ValueError):
            pass


def _signal_cleanup(signum: int, _frame: types.FrameType | None) -> None:
    # On SIGTERM/SIGINT (e.g. Ctrl+C when running under uvicorn --reload), unload
    # models so joblib/loky release semaphores. Use _exit() so we don't raise
    # SystemExit into asyncio/uvicorn (which would produce a traceback).
    _remove_managed_pid_file()
    _cleanup_models()
    # Return conventional signal exit status (128 + signal number).
    code = 128 + signum
    _os._exit(code)


# atexit is a backup for normal exit paths; safe to register at module level.
atexit.register(_cleanup_models)
atexit.register(_remove_managed_pid_file)


def _register_signal_handlers() -> None:
    # Register signal handlers for clean shutdown.  Called from lifespan
    # (server startup) instead of module level so that importing main.py in
    # tests does not override the test runner's signal handlers.
    signal.signal(signal.SIGTERM, _signal_cleanup)
    # Only register SIGINT when not using reload, so the reloader can handle
    # Ctrl+C and send SIGTERM to the child; in non-reload mode we need SIGINT
    # to cleanup.
    if not settings.dev_reload:
        signal.signal(signal.SIGINT, _signal_cleanup)


# ==============================================================================
# LLM Warmup
# ==============================================================================

async def _run_llm_warmup() -> None:
    """
    Warm up the generation LLM by running a minimal production-path call.

    This initializes Metal GPU shaders and allocates generation runtime state so
    the first real user query avoids a cold-start penalty.

    Skipped if the model file is missing or larger than 20 GB (very large models
    are slow enough to load that warmup would block startup for too long even in
    the background). Models over 20 GB load lazily on first query.
    """
    try:
        model_path = llm_engine._get_model_path()
        if not model_path.exists():
            log.info(
                'llm_warmup_skipped_model_not_found',
                model_path=str(model_path),
                msg='Model file not found — will load on first query',
            )
            return
        model_size_gb = model_path.stat().st_size / (1024 ** 3)
        if model_size_gb > 20:
            log.info(
                'llm_warmup_skipped_large_model',
                model_size_gb=round(model_size_gb, 1),
                model=model_path.name,
                msg='Skipping warmup for large model — will load on first query',
            )
            return
        log.info('llm_warmup_starting', model=model_path.name, model_size_gb=round(model_size_gb, 1))
        from informity.llm.model_adapter import get_profile
        profile = get_profile()
        from informity.llm.prompt_builder import build_messages as _build_gen_messages
        messages = _build_gen_messages('warmup', context_chunks=[])
        stops = profile.get_stop_sequences(reasoning_enabled=False)
        await asyncio.wait_for(
            asyncio.to_thread(
                llm_engine.chat_complete,
                messages=messages,
                max_tokens=1,
                temperature=0.1,
                stop=stops,
            ),
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
        log.info('llm_warmup_completed')
    except asyncio.CancelledError:
        log.info('llm_warmup_cancelled')
        raise
    except TimeoutError:
        log.warning(
            'llm_warmup_timeout',
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg='LLM warmup timed out — model will respond on first query',
        )
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning('llm_warmup_failed', error=str(exc))


async def _run_embedder_warmup() -> None:
    """
    Warm up the embedding model by running a minimal encode call.

    This loads the SentenceTransformer model into memory and initializes MPS
    (Metal) kernels so the first real user query incurs no cold-start latency.
    """
    try:
        log.info('embedder_warmup_starting', model=settings.embedding_model)
        await asyncio.wait_for(
            asyncio.to_thread(embedder.embed_query, 'warmup'),
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
        log.info('embedder_warmup_completed')
    except asyncio.CancelledError:
        log.info('embedder_warmup_cancelled')
        raise
    except TimeoutError:
        log.warning(
            'embedder_warmup_timeout',
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg='Embedder warmup timed out — model will load on first query',
        )
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning('embedder_warmup_failed', error=str(exc))


async def _run_intent_router_warmup() -> None:
    """
    Warm up intent-router embeddings so first classification is not cold.
    """
    try:
        from informity.llm.intent_router import get_intent_router

        log.info('intent_router_warmup_starting')
        await asyncio.wait_for(
            asyncio.to_thread(get_intent_router().classify_intent, 'List indexed files.'),
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
        log.info('intent_router_warmup_completed')
    except asyncio.CancelledError:
        log.info('intent_router_warmup_cancelled')
        raise
    except TimeoutError:
        log.warning(
            'intent_router_warmup_timeout',
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg='Intent router warmup timed out — router will initialize on first classification',
        )
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning('intent_router_warmup_failed', error=str(exc))


# ==============================================================================
# Lifespan — startup and shutdown logic
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # -- Startup --------------------------------------------------------------
    _register_signal_handlers()
    _write_managed_pid_file()

    # Lower process priority so scans/indexing yield CPU time to foreground apps.
    # 0 disables priority changes.
    if settings.cpu_priority_nice > 0:
        try:
            if _os.name != 'nt':
                _os.nice(settings.cpu_priority_nice)
            else:
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                    0x4000,  # BELOW_NORMAL_PRIORITY_CLASS
                )
            log.info('process_priority_lowered', cpu_priority_nice=settings.cpu_priority_nice)
        except (OSError, AttributeError, ValueError) as exc:
            log.warning('process_priority_lower_failed', error=str(exc), cpu_priority_nice=settings.cpu_priority_nice)

    log.info('application_starting', host=settings.host, port=settings.port)

    # Create required directories
    settings.ensure_directories()

    # Remove any huggingface_hub .cache under models_dir (leftover from downloads)
    remove_models_dir_cache()

    # Initialize the database (create tables if needed)
    await init_db()

    # Clear any RUNNING scan records left from a previous process (crash/restart)
    await clear_stale_running_scans()

    # Prune expired continuation pass artifacts (TTL-based cleanup even during low-write periods).
    try:
        conn = await get_connection()
        try:
            await prune_continuation_artifacts(conn)
        finally:
            await conn.close()
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning('continuation_artifact_startup_prune_failed', error=str(exc))

    # Populate adaptive top-k cache from corpus stats (if enabled).
    # Startup is an explicit lifecycle event, so force recompute now.
    try:
        conn = await get_connection()
        try:
            await update_tuning_cache(conn, force_recompute=True)
        finally:
            await conn.close()
    except (ImportError, _STARTUP_RUNTIME_EXCEPTIONS) as exc:
        log.warning('adaptive_tuning_startup_failed', error=str(exc))

    # Warm up generation, embeddings, and intent-router index.
    # Server mode: blocking warmup before the server accepts requests.
    # Desktop mode: skip startup warmup to avoid blocking app launch.
    # Skipped in dev mode (reload) to avoid double-warmup on code changes.
    if not settings.dev_reload and not _DESKTOP_SESSION_MODE:
        await asyncio.gather(_run_llm_warmup(), _run_embedder_warmup())
        await _run_intent_router_warmup()

    # Start file watcher for incremental indexing (if watched_directories configured)
    loop = asyncio.get_running_loop()
    start_watcher(loop)

    if settings.mcp_enabled and settings.mcp_auto_start:
        await mcp_lifecycle.start_from_settings()

    log.info('application_started')

    yield

    # -- Shutdown -------------------------------------------------------------
    log.info('application_shutting_down')
    _remove_managed_pid_file()

    stop_watcher()
    await mcp_lifecycle.stop()
    _cleanup_models()

    # Kill any lingering child processes (tokenizers, embedder workers)
    _kill_child_processes()

    log.info('application_shutdown_complete')


# ==============================================================================
# Application
# ==============================================================================

def _resolve_api_docs_enabled() -> bool:
    # Desktop-shell mode always disables docs/OpenAPI routes.
    if _DESKTOP_SESSION_MODE:
        return False
    # Explicit setting wins; otherwise expose docs only in dev_reload sessions.
    if settings.api_docs_enabled is not None:
        return bool(settings.api_docs_enabled)
    return bool(settings.dev_reload)


_api_docs_enabled = _resolve_api_docs_enabled()

app = FastAPI(
    title=APP_DISPLAY_NAME,
    description='Privacy-first local document intelligence for macOS',
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url='/docs' if _api_docs_enabled else None,
    redoc_url='/redoc' if _api_docs_enabled else None,
    openapi_url='/openapi.json' if _api_docs_enabled else None,
)

if not _api_docs_enabled:
    # Prevent SPA static fallback from serving index.html on docs/OpenAPI paths.
    @app.get('/docs')
    async def docs_disabled() -> Response:
        return Response(status_code=404)

    @app.get('/redoc')
    async def redoc_disabled() -> Response:
        return Response(status_code=404)

    @app.get('/openapi.json')
    async def openapi_disabled() -> Response:
        return Response(status_code=404)

# ==============================================================================
# Middleware
# ==============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all HTTP requests: method, path, status code, duration.
    Skips static file requests (e.g., /, /index.html) to reduce noise.
    """
    async def dispatch(self, request: Request, call_next):
        clear_contextvars()
        request_id = uuid.uuid4().hex
        bind_contextvars(
            request_id=request_id,
            request_method=request.method,
            request_path=request.url.path,
        )

        # Skip logging for static files (frontend assets)
        if not request.url.path.startswith('/api'):
            # Check if it's likely a static file request (has file extension or is root)
            path = request.url.path
            if path == '/' or '.' in path.split('/')[-1]:
                # Likely a static file, skip detailed logging
                try:
                    response = await call_next(request)
                    response.headers['X-Request-ID'] = request_id
                    return response
                finally:
                    clear_contextvars()

        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = str(request.url.query) if request.url.query else ''

        # Log request start
        log.debug(
            'http_request_start',
            method=method,
            path=path,
            query=query_params,
        )

        try:
            response = await call_next(request)
            status_code = response.status_code
            duration_ms = (time.time() - start_time) * 1000

            # High-frequency polling endpoints (GET requests to status endpoints or file listings)
            # are logged at debug to avoid console noise during scans.
            # Pattern-based: any /api/*/status or /api/files GET request.
            is_polling_endpoint = (
                method == 'GET' and
                (path.endswith('/status') or path == '/api/files' or path == '/api/health')
            )
            log_fn = log.debug if is_polling_endpoint else log.info

            log_fn(
                'http_request',
                method=method,
                path=path,
                query=query_params,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers['X-Request-ID'] = request_id
            return response
        except _REQUEST_RUNTIME_EXCEPTIONS as exc:
            duration_ms = (time.time() - start_time) * 1000
            log.error(
                'http_request_error',
                method=method,
                path=path,
                query=query_params,
                error=str(exc),
                duration_ms=round(duration_ms, 2),
                exc_info=True,
            )
            raise
        finally:
            clear_contextvars()


class DesktopSessionMiddleware(BaseHTTPMiddleware):
    """
    Enforce per-launch desktop session token for API routes when running under Tauri.
    """

    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else None
        if (
            not _DESKTOP_SESSION_MODE
            and request.url.path.startswith('/api')
            and request.method != 'OPTIONS'
            and not is_loopback_host(client_host)
        ):
            return JSONResponse(
                status_code=403,
                content={'detail': 'API is only accessible from localhost in non-desktop mode.'},
            )
        if (
            _DESKTOP_SESSION_MODE
            and request.url.path.startswith('/api')
            and request.method != 'OPTIONS'
            and not is_tauri_session_authorized(request.headers, _TAURI_SESSION_TOKEN)
        ):
            return JSONResponse(
                status_code=401,
                content={
                    'detail': (
                        f'Missing or invalid desktop session token. '
                        f'Provide {TAURI_SESSION_HEADER}.'
                    ),
                },
            )
        return await call_next(request)


# Request logging — log all API requests
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(DesktopSessionMiddleware)

# CORS — allow the frontend to talk to the API from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(settings.port, desktop_mode=_DESKTOP_SESSION_MODE),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ==============================================================================
# Health Check
# ==============================================================================

@app.get('/api/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    # Simple health check endpoint.
    return HealthResponse(app_display_name=APP_DISPLAY_NAME)


# ==============================================================================
# Routers — will be wired up as modules are implemented
# ==============================================================================

app.include_router(scan_router)
app.include_router(index_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(settings_router)
app.include_router(system_router)


# ==============================================================================
# Static Frontend
# ==============================================================================
# Serve Vite build output (frontend/dist). Run `make frontend-build` before `make run`.
# Vanilla backup archived at .archive/frontend-bak/.
# SPAStaticFiles serves index.html for unknown paths so client-side routing works
# (e.g. reload on /chat or /files).
_SRC_DIR = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _SRC_DIR / 'frontend' / 'dist'


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown paths (SPA client-side routing)."""

    async def get_response(self, path: str, scope: dict) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404 and self.html:
                try:
                    return await super().get_response('index.html', scope)
                except HTTPException as index_exc:
                    raise exc from index_exc
            raise


if _FRONTEND_DIST.exists():
    app.mount('/', SPAStaticFiles(directory=str(_FRONTEND_DIST), html=True), name='frontend')
    log.info('static_files_mounted', directory=str(_FRONTEND_DIST), source='vite_dist')
else:
    log.warning('frontend_directory_not_found', expected=str(_FRONTEND_DIST), hint='Run make frontend-build')


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    # Run the application with uvicorn.
    # reload=True only when dev_reload is set (e.g. make dev); never in production.
    # access_log=False because we have custom RequestLoggingMiddleware that provides structured logging.
    if not is_loopback_host(settings.host):
        log.warning(
            'non_loopback_host_configured',
            host=settings.host,
            msg='API is bound to a non-loopback host; this increases local-network exposure risk.',
        )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.dev_reload,
        log_level=settings.log_level,
        access_log=False,
    )


if __name__ == '__main__':
    # Required for frozen executables (PyInstaller) that use multiprocessing.
    # Without freeze_support, spawned worker processes can re-enter the main
    # application entrypoint and become orphaned long-lived backend processes.
    multiprocessing.freeze_support()
    main()
