# ==============================================================================
# Informity AI — FastAPI Application Entry Point
# Creates the app, registers routers, configures middleware, and manages
# the application lifespan (DB init, directory creation, clean shutdown).
# ==============================================================================
# ruff: noqa: E402

# CPU thread limits are applied by config._apply_thread_limits_early() at import time.
# Do not duplicate them here — config.py is the single source of truth.

"""FastAPI application entry point for Informity."""

import os as _os
import sys as _sys

# Suppress SyntaxWarnings from third-party libraries (e.g., pysbd) before any imports
# These warnings are emitted at import time and are harmless
# Python 3.13 is stricter about escape sequences, causing warnings in third-party code
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

# IMPORTANT: MCP stdio mode must short-circuit before normal app imports/startup,
# otherwise structured/colorized logs can leak into stdout and corrupt JSON-RPC framing.
_PROGRAM_NAME = _os.path.basename(str(_sys.argv[0] or "")).lower()
_MCP_STDIO_MODE = (
    "--mcp-stdio" in {str(arg) for arg in _sys.argv[1:]} or _PROGRAM_NAME == "informity-mcp"
)
if _MCP_STDIO_MODE:
    from informity.mcp.stdio_server import (
        main as _mcp_stdio_main,  # pylint: disable=ungrouped-imports
    )

    _mcp_stdio_main()
    raise SystemExit(0)

import asyncio
import atexit
import multiprocessing
import signal
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

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
from informity.api.routes_debug import router as debug_router
from informity.api.routes_index import router as index_router
from informity.api.routes_logs import router as logs_router
from informity.api.routes_plugins import router as plugins_router
from informity.api.routes_scan import router as scan_router
from informity.api.routes_search import router as search_router
from informity.api.routes_settings import router as settings_router
from informity.api.routes_system import router as system_router
from informity.api.routes_translate import router as translate_router
from informity.api.schemas import HealthResponse
from informity.api.security import (
    TAURI_SESSION_HEADER,
    get_cors_allow_origins,
    get_tauri_session_token_from_env,
    is_loopback_host,
    is_tauri_desktop_mode,
    is_tauri_session_authorized,
)
from informity.config import (
    APP_DISPLAY_NAME,
    are_required_models_cached,
    configure_hf_environment,
    settings,
)
from informity.exceptions import LLMError
from informity.version import APP_VERSION

# Set Hugging Face cache paths and offline flags before importing models.
# Allow boot into first-run setup when local models are not cached yet.
configure_hf_environment(fail_on_missing_full_privacy_models=False)

from informity.db.sqlite import (
    LOG_EVENTS_MAX_ROWS_DEFAULT,
    LOG_EVENTS_RETENTION_DAYS_DEFAULT,
    clear_stale_running_scans,
    get_connection,
    init_db,
    prune_continuation_artifacts,
    prune_log_events,
)
from informity.indexer.adaptive_tuning import update_tuning_cache
from informity.indexer.embedder import embedder
from informity.indexer.reranker import reranker
from informity.llm.engine import llm_engine, remove_models_dir_cache
from informity.llm.five_q_classifier import resolve_classifier_model_path
from informity.llm.model_adapter import get_profile
from informity.llm.model_bootstrap import CLASSIFIER_GGUF_SPEC, download_gguf_model
from informity.llm.prompt_builder import BuildMessagesRequest
from informity.llm.prompt_builder import build_messages as _build_gen_messages
from informity.logging_config import configure_logging
from informity.mcp.lifecycle import mcp_lifecycle
from informity.scanner.watcher import start_watcher, stop_watcher
from informity.storage_migrations import migrate_legacy_upload_storage_layout

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for minimal environments
    psutil = None

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
_STARTUP_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, TimeoutError, LLMError)
_REQUEST_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, TimeoutError)
_WARMUP_TIMEOUT_SECONDS = 300.0
_TAURI_SESSION_TOKEN = get_tauri_session_token_from_env()
_DESKTOP_SESSION_MODE = is_tauri_desktop_mode(_TAURI_SESSION_TOKEN)
_MANAGED_PID_FILE_ENV = "INFORMITY_MANAGED_PID_FILE"
_MANAGED_PID_FILE_RAW = _os.environ.get(_MANAGED_PID_FILE_ENV, "").strip()
_MANAGED_PID_FILE_PATH: Path | None = (
    Path(_MANAGED_PID_FILE_RAW).expanduser() if _MANAGED_PID_FILE_RAW else None
)
_STARTUP_RAM_HEADROOM_RATIO = 0.85
_STARTUP_STATE_UNSET = object()


def _llm_engine_get_model_path() -> Path:
    """Get the active LLM model path with backward-compatible fallback."""
    model_path_getter = getattr(llm_engine, "get_model_path", None)
    if callable(model_path_getter):
        return model_path_getter()
    try:
        return llm_engine._get_model_path()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise AttributeError("llm_engine does not expose a model path getter") from exc


def _enable_startup_bootstrap_mode_if_needed() -> bool:
    """
    Temporarily relax privacy flags so startup can proceed when models are missing.

    The persisted privacy preference is not changed here. This just lets the app
    boot and fetch required models on demand instead of failing startup outright.
    """
    if are_required_models_cached():
        return False

    if settings.full_privacy or settings.llm_local_only or settings.embedding_offline:
        log.info(
            "startup_bootstrap_mode_enabled",
            full_privacy=settings.full_privacy,
            llm_local_only=settings.llm_local_only,
            embedding_offline=settings.embedding_offline,
            message=(
                "Required models are not cached yet; starting in temporary bootstrap mode"
                " so the app can launch and fetch them on demand."
            ),
        )

    settings.full_privacy = False
    settings.llm_local_only = False
    settings.embedding_offline = False
    configure_hf_environment(fail_on_missing_full_privacy_models=False)
    return True


@dataclass
class StartupHealthState:
    """StartupHealthState model."""
    status: str = "initializing"
    reason: str | None = "starting"
    detail: str | None = None
    progress_done: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def update(
        self,
        *,
        status: str | object = _STARTUP_STATE_UNSET,
        reason: str | object = _STARTUP_STATE_UNSET,
        detail: str | object = _STARTUP_STATE_UNSET,
        progress_done: int | None | object = _STARTUP_STATE_UNSET,
        progress_total: int | None | object = _STARTUP_STATE_UNSET,
        progress_percent: float | None | object = _STARTUP_STATE_UNSET,
    ) -> None:
        """update."""
        with self._lock:
            if status is not _STARTUP_STATE_UNSET:
                self.status = status
            if reason is not _STARTUP_STATE_UNSET:
                self.reason = reason
            if detail is not _STARTUP_STATE_UNSET:
                self.detail = detail
            if progress_done is not _STARTUP_STATE_UNSET:
                self.progress_done = progress_done
            if progress_total is not _STARTUP_STATE_UNSET:
                self.progress_total = progress_total
            if progress_percent is not _STARTUP_STATE_UNSET:
                self.progress_percent = progress_percent

    def snapshot(self) -> dict[str, object | None]:
        """snapshot."""
        with self._lock:
            return {
                "status": self.status,
                "reason": self.reason,
                "detail": self.detail,
                "progress_done": self.progress_done,
                "progress_total": self.progress_total,
                "progress_percent": self.progress_percent,
            }


def _set_startup_state(
    startup_app: FastAPI,
    *,
    status: str | object = _STARTUP_STATE_UNSET,
    reason: str | object = _STARTUP_STATE_UNSET,
    detail: str | object = _STARTUP_STATE_UNSET,
    progress_done: int | None | object = _STARTUP_STATE_UNSET,
    progress_total: int | None | object = _STARTUP_STATE_UNSET,
    progress_percent: float | None | object = _STARTUP_STATE_UNSET,
) -> None:
    """ set startup state."""
    tracker = getattr(startup_app.state, "startup_health_state", None)
    if isinstance(tracker, StartupHealthState):
        tracker.update(
            status=status,
            reason=reason,
            detail=detail,
            progress_done=progress_done,
            progress_total=progress_total,
            progress_percent=progress_percent,
        )


def _get_startup_state_snapshot(startup_app: FastAPI) -> dict[str, object | None]:
    """ get startup state snapshot."""
    tracker = getattr(startup_app.state, "startup_health_state", None)
    if isinstance(tracker, StartupHealthState):
        return tracker.snapshot()
    return {
        "status": "ok",
        "reason": None,
        "detail": None,
        "progress_done": None,
        "progress_total": None,
        "progress_percent": None,
    }


# ==============================================================================
# Process Cleanup
# ==============================================================================


def _cleanup_models() -> None:
    # Unload models to release resources.
    # Used on normal shutdown (lifespan/atexit) and on SIGTERM/SIGINT (reload child).
    """ cleanup models."""
    embedder.unload()
    reranker.unload()


def _write_managed_pid_file() -> None:
    """ write managed pid file."""
    if _MANAGED_PID_FILE_PATH is None:
        return
    try:
        _MANAGED_PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MANAGED_PID_FILE_PATH.write_text(f"{_os.getpid()}\n", encoding="utf-8")
    except OSError as exc:
        log.warning(
            "managed_pid_file_write_failed",
            path=str(_MANAGED_PID_FILE_PATH),
            error=str(exc),
        )


def _remove_managed_pid_file() -> None:
    """ remove managed pid file."""
    if _MANAGED_PID_FILE_PATH is None:
        return
    with suppress(OSError):
        _MANAGED_PID_FILE_PATH.unlink(missing_ok=True)


def _kill_child_processes() -> None:
    # Kill child processes known to multiprocessing (tokenizers, embedder workers).
    # We do not use killpg(process_group, SIGTERM) because we are in that group;
    # that would signal ourselves and trigger _signal_cleanup during lifespan.
    """ kill child processes."""
    active_children = multiprocessing.active_children()
    for child in active_children:
        log.debug("terminating_child_process", pid=child.pid, name=child.name)
        with suppress(OSError):
            child.terminate()

    for child in active_children:
        try:
            child.join(timeout=2)
            if child.is_alive():
                log.warning("force_killing_child", pid=child.pid, name=child.name)
                child.kill()
        except (OSError, ValueError):
            pass


def _signal_cleanup(signum: int, _frame: object | None) -> None:
    # On SIGTERM/SIGINT (e.g. Ctrl+C when running under uvicorn --reload), unload
    # models so joblib/loky release semaphores. Use _exit() so we don't raise
    # SystemExit into asyncio/uvicorn (which would produce a traceback).
    """ signal cleanup."""
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
    """ register signal handlers."""
    signal.signal(signal.SIGTERM, _signal_cleanup)
    # Only register SIGINT when not using reload, so the reloader can handle
    # Ctrl+C and send SIGTERM to the child; in non-reload mode we need SIGINT
    # to cleanup.
    if not settings.dev_reload:
        signal.signal(signal.SIGINT, _signal_cleanup)


# ==============================================================================
# LLM Warmup
# ==============================================================================


async def _run_llm_warmup() -> bool:
    """
    Warm up the generation LLM by running a minimal production-path call.

    This initializes Metal GPU shaders and allocates generation runtime state so
    the first real user query avoids a cold-start penalty.

    Skipped if the model file is missing or larger than 20 GB (very large models
    are slow enough to load that warmup would block startup for too long even in
    the background). Models over 20 GB load lazily on first query.
    """
    try:
        model_path = _llm_engine_get_model_path()
        if not model_path.exists():
            log.info(
                "llm_warmup_skipped_model_not_found",
                model_path=str(model_path),
                msg="Model file not found — will load on first query",
            )
            return False
        model_size_gb = model_path.stat().st_size / (1024**3)
        available_ram_gb = _get_available_ram_gb()
        will_warm = model_size_gb <= available_ram_gb * _STARTUP_RAM_HEADROOM_RATIO
        log.info(
            "startup_warmup_check",
            model=model_path.name,
            model_size_gb=round(model_size_gb, 1),
            available_ram_gb=round(available_ram_gb, 1),
            will_warm=will_warm,
        )
        if not will_warm:
            log.info(
                "llm_warmup_skipped_insufficient_ram",
                model=model_path.name,
                model_size_gb=round(model_size_gb, 1),
                available_ram_gb=round(available_ram_gb, 1),
                headroom_ratio=_STARTUP_RAM_HEADROOM_RATIO,
                msg="Skipping warmup for current model — will load on first query",
            )
            return False
        log.info(
            "llm_warmup_starting", model=model_path.name, model_size_gb=round(model_size_gb, 1)
        )

        profile = get_profile()

        messages = _build_gen_messages(
            BuildMessagesRequest(
                question="warmup",
                context_chunks=[],
            )
        )
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
        log.info("llm_warmup_completed")
        return True
    except asyncio.CancelledError:
        log.info("llm_warmup_cancelled")
        raise
    except TimeoutError:
        log.warning(
            "llm_warmup_timeout",
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg="LLM warmup timed out — model will respond on first query",
        )
        return False
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("llm_warmup_failed", error=str(exc))
        return False


def _get_available_ram_gb() -> float:
    """
    Return available RAM in GiB when possible, otherwise fall back to total RAM.
    """
    if psutil is not None:
        try:
            return float(psutil.virtual_memory().available) / (1024**3)
        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    try:
        pages = int(_os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(_os.sysconf("SC_PAGE_SIZE"))
        available_bytes = pages * page_size
        return float(available_bytes) / (1024**3)
    except (AttributeError, ValueError, OSError):
        pass

    try:
        pages = int(_os.sysconf("SC_PHYS_PAGES"))
        page_size = int(_os.sysconf("SC_PAGE_SIZE"))
        total_bytes = pages * page_size
        return float(total_bytes) / (1024**3)
    except (AttributeError, ValueError, OSError):
        return 0.0


async def _run_embedder_warmup() -> bool:
    """
    Warm up the embedding model by running a minimal encode call.

    This loads the SentenceTransformer model into memory and initializes MPS
    (Metal) kernels so the first real user query incurs no cold-start latency.
    """
    try:
        log.info("embedder_warmup_starting", model=settings.embedding_model)
        await asyncio.wait_for(
            asyncio.to_thread(embedder.embed_query, "warmup"),
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
        log.info("embedder_warmup_completed")
        return True
    except asyncio.CancelledError:
        log.info("embedder_warmup_cancelled")
        raise
    except TimeoutError:
        log.warning(
            "embedder_warmup_timeout",
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg="Embedder warmup timed out — model will load on first query",
        )
        return False
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("embedder_warmup_failed", error=str(exc))
        return False


async def _run_five_q_classifier_warmup() -> bool:
    """
    Warm up the 5Q classifier so first classification is not cold.
    """
    try:
        from informity.llm.classifier_service import get_classifier
        from informity.llm.five_q_classifier import ClassifierContext

        log.info("five_q_classifier_warmup_starting")
        await asyncio.wait_for(
            asyncio.to_thread(
                get_classifier().classify,
                "what documents do I have",
                ClassifierContext(
                    chat_mode="researcher", scope_kind="indexed_corpus", has_prior_turns=False
                ),
            ),
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
        log.info("five_q_classifier_warmup_completed")
        return True
    except asyncio.CancelledError:
        log.info("five_q_classifier_warmup_cancelled")
        raise
    except TimeoutError:
        log.warning(
            "five_q_classifier_warmup_timeout",
            timeout_seconds=int(_WARMUP_TIMEOUT_SECONDS),
            msg=(
                "5Q classifier warmup timed out — classifier will initialize on first"
                "classification"
            ),
        )
        return False
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("five_q_classifier_warmup_failed", error=str(exc))
        return False


async def _ensure_classifier_model_present(startup_app: FastAPI) -> Path:
    """ ensure classifier model present."""
    model_path = resolve_classifier_model_path()
    if model_path is not None and model_path.exists():
        return model_path

    classifier_target_dir = settings.classifier_models_dir
    classifier_target_path = classifier_target_dir / CLASSIFIER_GGUF_SPEC.filename
    _set_startup_state(
        startup_app,
        status="initializing",
        reason="downloading_classifier_model",
        detail="Downloading classifier model before startup.",
        progress_done=0,
        progress_total=None,
        progress_percent=None,
    )
    log.info(
        "classifier_model_missing_starting_download",
        model_path=str(classifier_target_path),
        repo=CLASSIFIER_GGUF_SPEC.repo_id,
        filename=CLASSIFIER_GGUF_SPEC.filename,
    )

    def _progress(bytes_done: int, total_bytes: int | None, _speed_bps: float) -> None:
        """ progress."""
        percent: float | None = None
        if total_bytes and total_bytes > 0:
            percent = min(100.0, max(0.0, (bytes_done / total_bytes) * 100.0))
        _set_startup_state(
            startup_app,
            status="initializing",
            reason="downloading_classifier_model",
            progress_done=bytes_done,
            progress_total=total_bytes,
            progress_percent=percent,
        )

    try:
        await asyncio.to_thread(
            download_gguf_model,
            spec=CLASSIFIER_GGUF_SPEC,
            target_path=classifier_target_path,
            progress_callback=_progress,
        )
    except Exception as exc:
        detail = f"classifier model download failed: {exc}"
        _set_startup_state(
            startup_app, status="error", reason="downloading_classifier_model", detail=detail
        )
        log.warning("classifier_model_download_failed", error=str(exc))
        raise

    model_path = resolve_classifier_model_path()
    if model_path is None or not model_path.exists():
        detail = (
            f"classifier model download completed but file is still missing:"
            f"{classifier_target_path}"
        )
        _set_startup_state(
            startup_app, status="error", reason="downloading_classifier_model", detail=detail
        )
        raise LLMError(detail)

    _set_startup_state(
        startup_app,
        status="initializing",
        reason="warming_models",
        detail="Classifier model ready. Warming startup models.",
        progress_done=None,
        progress_total=None,
        progress_percent=None,
    )
    return model_path


async def _run_startup_sequence(startup_app: FastAPI) -> None:
    """
    Run startup initialization after the server is already able to answer health checks.

    The sequence remains serialized so the classifier download, warmups, watcher,
    and MCP startup happen in a predictable order while /api/health reports progress.
    """
    startup_started_at = time.perf_counter()
    sidecar_startup_attempted = False
    try:
        _enable_startup_bootstrap_mode_if_needed()
        await _ensure_classifier_model_present(startup_app)

        warmups = (
            ("classifier", _run_five_q_classifier_warmup),
            ("llm", _run_llm_warmup),
            ("embedder", _run_embedder_warmup),
        )

        for component, warmup in warmups:
            component_started_at = time.perf_counter()
            warmed = await warmup()
            component_duration_ms = round((time.perf_counter() - component_started_at) * 1000, 1)
            startup_elapsed_ms = round((time.perf_counter() - startup_started_at) * 1000, 1)
            if warmed:
                log.info(
                    f"{component}_model_loaded",
                    component=component,
                    duration_ms=component_duration_ms,
                    elapsed_ms=startup_elapsed_ms,
                    module="main",
                    operation=f"{component}_warmup_completed",
                    status="ok",
                )

        _set_startup_state(
            startup_app,
            status="ok",
            reason="ready",
            detail="Startup complete.",
            progress_done=1,
            progress_total=1,
            progress_percent=100.0,
        )
        log.info(
            "server_ready",
            component="main",
            message="all models loaded, accepting requests",
            module="main",
            operation="server_ready",
            startup_elapsed_ms=round((time.perf_counter() - startup_started_at) * 1000, 1),
            status="ok",
        )
    except asyncio.CancelledError:
        _set_startup_state(
            startup_app, status="error", reason="startup_cancelled", detail="Startup was cancelled."
        )
        raise
    except LLMError:
        raise
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        detail = f"startup sequence failed: {exc}"
        _set_startup_state(startup_app, status="error", reason="startup_sequence", detail=detail)
        log.warning("startup_sequence_failed", error=str(exc))
        raise
    except Exception as exc:
        detail = f"unexpected startup sequence failure: {exc}"
        _set_startup_state(startup_app, status="error", reason="startup_sequence", detail=detail)
        log.exception("startup_sequence_unexpected_failure", error=str(exc))
        raise
    finally:
        if not sidecar_startup_attempted:
            sidecar_startup_attempted = True
            try:
                loop = asyncio.get_running_loop()
                start_watcher(loop)
                if settings.mcp_enabled and settings.mcp_auto_start:
                    await mcp_lifecycle.start_from_settings()
            except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                log.warning("startup_sidecars_failed", error=str(exc))


async def _backfill_page_counts(conn: object) -> None:
    """
    Populate page_count for indexed files where it is NULL.

    Uses pypdfium2 for PDFs (fast — no Docling model loading required).
    For non-PDF formats, estimates from total chunk token count:
      page_count ≈ total_tokens / TRANSLATE_AVG_TOKENS_PER_PAGE.
    Runs at startup; errors per-file are logged and skipped.
    """
    import aiosqlite

    from informity.translate_policy import TRANSLATE_AVG_TOKENS_PER_PAGE

    db: aiosqlite.Connection = conn  # type: ignore[assignment]

    cursor = await db.execute(
        """
        SELECT f.id, f.path, f.extension,
               COALESCE(SUM(c.token_count), 0) AS total_tokens
        FROM files f
        LEFT JOIN chunks c ON c.file_id = f.id AND c.parent_id IS NULL
        WHERE f.page_count IS NULL AND f.source_provider = 'filesystem'
        GROUP BY f.id
        """,
    )
    rows = await cursor.fetchall()
    if not rows:
        return

    updated = 0
    for row in rows:
        file_id = int(row["id"])
        path = str(row["path"] or "")
        ext = str(row["extension"] or "").lower()
        tokens = int(row["total_tokens"] or 0)
        page_count: int | None = None

        if ext == ".pdf" and path:
            try:
                import pypdfium2 as pdfium  # already a dependency via docling

                pdf_doc = pdfium.PdfDocument(path)
                page_count = len(pdf_doc)
                if hasattr(pdf_doc, "close"):
                    pdf_doc.close()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass  # fall through to token estimate

        if page_count is None and tokens > 0:
            page_count = max(1, tokens // TRANSLATE_AVG_TOKENS_PER_PAGE)

        if page_count:
            await db.execute(
                "UPDATE files SET page_count = ? WHERE id = ?",
                (page_count, file_id),
            )
            updated += 1

    if updated > 0:
        await db.commit()
        log.info("page_count_backfill_completed", files_updated=updated, files_checked=len(rows))


# ==============================================================================
# Lifespan — startup and shutdown logic
# ==============================================================================


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None]:
    # -- Startup --------------------------------------------------------------
    """lifespan."""
    _register_signal_handlers()
    _write_managed_pid_file()
    fastapi_app.state.startup_health_state = StartupHealthState()
    fastapi_app.state.startup_task = None

    # Lower process priority so scans/indexing yield CPU time to foreground apps.
    # 0 disables priority changes.
    if settings.cpu_priority_nice > 0:
        try:
            if _os.name != "nt":
                _os.nice(settings.cpu_priority_nice)
            else:
                import ctypes

                ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                    0x4000,  # BELOW_NORMAL_PRIORITY_CLASS
                )
            log.info("process_priority_lowered", cpu_priority_nice=settings.cpu_priority_nice)
        except (OSError, AttributeError, ValueError) as exc:
            log.warning(
                "process_priority_lower_failed",
                error=str(exc),
                cpu_priority_nice=settings.cpu_priority_nice,
            )

    log.info("application_starting", host=settings.host, port=settings.port)

    # Create required directories
    settings.ensure_directories()

    # Remove any huggingface_hub .cache under models_dir (leftover from downloads)
    remove_models_dir_cache()

    # Initialize the database (create tables if needed)
    await init_db()

    # Best-effort compatibility migration for older upload layouts.
    try:
        conn = await get_connection()
        try:
            await migrate_legacy_upload_storage_layout(conn, settings.app_data_dir)
        finally:
            await conn.close()
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("legacy_upload_storage_migration_failed", error=str(exc))

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
        log.warning("continuation_artifact_startup_prune_failed", error=str(exc))

    # Prune log_events retention window and row cap.
    try:
        conn = await get_connection()
        try:
            prune_result = await prune_log_events(
                conn,
                retention_days=LOG_EVENTS_RETENTION_DAYS_DEFAULT,
                max_rows=LOG_EVENTS_MAX_ROWS_DEFAULT,
            )
            if (
                int(prune_result.get("deleted_by_age", 0)) > 0
                or int(prune_result.get("deleted_by_count", 0)) > 0
            ):
                log.info("log_events_pruned", **prune_result)
        finally:
            await conn.close()
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("log_events_startup_prune_failed", error=str(exc))

    # Sweep stale translate.local uploads older than TRANSLATE_CLEANUP_AGE_HOURS.
    try:
        from datetime import UTC, datetime, timedelta

        from informity.db.sqlite import delete_translate_jobs_older_than
        from informity.translate_policy import TRANSLATE_CLEANUP_AGE_HOURS

        conn = await get_connection()
        try:
            cutoff = (datetime.now(UTC) - timedelta(hours=TRANSLATE_CLEANUP_AGE_HOURS)).isoformat()
            deleted = await delete_translate_jobs_older_than(conn, cutoff)
            if deleted > 0:
                log.info("translate_jobs_startup_pruned", count=deleted)
        finally:
            await conn.close()
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("translate_jobs_startup_prune_failed", error=str(exc))

    # Backfill page_count for indexed files that have NULL.
    # Uses pypdfium2 for PDFs (fast, no Docling model needed) and token-count
    # estimation for other formats. Runs quietly; errors are non-fatal.
    try:
        conn = await get_connection()
        try:
            await _backfill_page_counts(conn)
        finally:
            await conn.close()
    except _STARTUP_RUNTIME_EXCEPTIONS as exc:
        log.warning("page_count_backfill_failed", error=str(exc))

    # Populate adaptive top-k cache from corpus stats (if enabled).
    # Startup is an explicit lifecycle event, so force recompute now.
    try:
        conn = await get_connection()
        try:
            await update_tuning_cache(conn, force_recompute=True)
        finally:
            await conn.close()
    except (ImportError, _STARTUP_RUNTIME_EXCEPTIONS) as exc:
        log.warning("adaptive_tuning_startup_failed", error=str(exc))

    # Start the remaining startup sequence in the background so /api/health can
    # report model download and warmup progress while the backend boots.
    fastapi_app.state.startup_task = asyncio.create_task(
        _run_startup_sequence(fastapi_app), name="informity-startup-sequence"
    )

    log.info("application_started")

    yield

    # -- Shutdown -------------------------------------------------------------
    log.info("application_shutting_down")
    _remove_managed_pid_file()

    startup_task = getattr(fastapi_app.state, "startup_task", None)
    if isinstance(startup_task, asyncio.Task) and not startup_task.done():
        startup_task.cancel()
        with suppress(asyncio.CancelledError):
            await startup_task

    stop_watcher()
    await mcp_lifecycle.stop()
    _cleanup_models()

    # Kill any lingering child processes (tokenizers, embedder workers)
    _kill_child_processes()

    log.info("application_shutdown_complete")


# ==============================================================================
# Application
# ==============================================================================


def _resolve_api_docs_enabled() -> bool:
    # Desktop-shell mode always disables docs/OpenAPI routes.
    """ resolve api docs enabled."""
    if _DESKTOP_SESSION_MODE:
        return False
    # Explicit setting wins; otherwise expose docs only in dev_reload sessions.
    if settings.api_docs_enabled is not None:
        return bool(settings.api_docs_enabled)
    return bool(settings.dev_reload)


_api_docs_enabled = _resolve_api_docs_enabled()

app = FastAPI(
    title=APP_DISPLAY_NAME,
    description="Privacy-first local document intelligence for macOS",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _api_docs_enabled else None,
    redoc_url="/redoc" if _api_docs_enabled else None,
    openapi_url="/openapi.json" if _api_docs_enabled else None,
)

if not _api_docs_enabled:
    # Prevent SPA static fallback from serving index.html on docs/OpenAPI paths.
    @app.get("/docs")
    async def docs_disabled() -> Response:
        """docs disabled."""
        return Response(status_code=404)

    @app.get("/redoc")
    async def redoc_disabled() -> Response:
        """redoc disabled."""
        return Response(status_code=404)

    @app.get("/openapi.json")
    async def openapi_disabled() -> Response:
        """openapi disabled."""
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
        """dispatch."""
        clear_contextvars()
        request_id = uuid.uuid4().hex
        bind_contextvars(
            request_id=request_id,
            request_method=request.method,
            request_path=request.url.path,
        )

        # Skip logging for static files (frontend assets)
        if not request.url.path.startswith("/api"):
            # Check if it's likely a static file request (has file extension or is root)
            path = request.url.path
            if path == "/" or "." in path.split("/")[-1]:
                # Likely a static file, skip detailed logging
                try:
                    response = await call_next(request)
                    response.headers["X-Request-ID"] = request_id
                    return response
                finally:
                    clear_contextvars()

        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = str(request.url.query) if request.url.query else ""

        # Log request start
        log.debug(
            "http_request_start",
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
            is_polling_endpoint = method == "GET" and (
                path.endswith("/status") or path == "/api/files" or path == "/api/health"
            )
            log_fn = log.debug if is_polling_endpoint else log.info

            log_fn(
                "http_request",
                method=method,
                path=path,
                query=query_params,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers["X-Request-ID"] = request_id
            return response
        except _REQUEST_RUNTIME_EXCEPTIONS as exc:
            duration_ms = (time.time() - start_time) * 1000
            log.error(
                "http_request_error",
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
        """dispatch."""
        client_host = request.client.host if request.client else None
        if (
            not _DESKTOP_SESSION_MODE
            and request.url.path.startswith("/api")
            and request.method != "OPTIONS"
            and not is_loopback_host(client_host)
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "API is only accessible from localhost in non-desktop mode."},
            )
        if (
            _DESKTOP_SESSION_MODE
            and request.url.path.startswith("/api")
            and request.method != "OPTIONS"
            and not is_tauri_session_authorized(request.headers, _TAURI_SESSION_TOKEN)
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        f"Missing or invalid desktop session token. Provide {TAURI_SESSION_HEADER}."
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Health Check
# ==============================================================================


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    # Simple health check endpoint.
    """health check."""
    startup_state = _get_startup_state_snapshot(app)
    return HealthResponse(
        app_display_name=APP_DISPLAY_NAME,
        status=str(startup_state.get("status") or "ok"),
        reason=startup_state.get("reason"),
        detail=startup_state.get("detail"),
        progress_done=startup_state.get("progress_done"),
        progress_total=startup_state.get("progress_total"),
        progress_percent=startup_state.get("progress_percent"),
    )


# ==============================================================================
# Routers — wired up below
# ==============================================================================

app.include_router(scan_router)
app.include_router(index_router)
app.include_router(chat_router)
app.include_router(debug_router)
app.include_router(translate_router)
app.include_router(search_router)
app.include_router(settings_router)
app.include_router(plugins_router)
app.include_router(system_router)
app.include_router(logs_router)


# ==============================================================================
# Static Frontend
# ==============================================================================
# Serve Vite build output (frontend/dist). Run `make frontend-build` before `make run`.
# Vanilla backup archived at .archive/frontend-bak/.
# SPAStaticFiles serves index.html for unknown paths so client-side routing works
# (e.g. reload on /chat or /files).
_SRC_DIR = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _SRC_DIR / "frontend" / "dist"


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown paths (SPA client-side routing)."""

    async def get_response(self, path: str, scope: dict) -> Response:
        """get response."""
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404 and self.html:
                try:
                    return await super().get_response("index.html", scope)
                except HTTPException as index_exc:
                    raise exc from index_exc
            raise


if _FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    log.info("static_files_mounted", directory=str(_FRONTEND_DIST), source="vite_dist")
else:
    log.warning(
        "frontend_directory_not_found", expected=str(_FRONTEND_DIST), hint="Run make frontend-build"
    )


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main() -> None:
    # Run the application with uvicorn.
    # reload=True only when dev_reload is set (e.g. make dev); never in production.
    # access_log=False because we have custom RequestLoggingMiddleware that provides structured
    # logging.
    """main."""
    if not is_loopback_host(settings.host):
        log.warning(
            "non_loopback_host_configured",
            host=settings.host,
            msg="API is bound to a non-loopback host; this increases local-network exposure risk.",
        )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.dev_reload,
        log_level=settings.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    # Required for frozen executables (PyInstaller) that use multiprocessing.
    # Without freeze_support, spawned worker processes can re-enter the main
    # application entrypoint and become orphaned long-lived backend processes.
    multiprocessing.freeze_support()
    main()
