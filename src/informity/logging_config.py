# ==============================================================================
# Informity AI — Logging Configuration
# Configures structlog to write structured logs to both console and files:
#   - Console: INFO and above (or app log_level if higher)
#   - app.log: JSON logs at app log level (default INFO; set log_level=debug for more)
#   - app.error.log: JSON logs for only ERROR and CRITICAL level logs
#
# Third-party loggers (aiosqlite, urllib3, etc.) are set to WARNING so they
# do not flood logs even when app log_level is debug.
# Logs are rotated daily and kept for 7 days.
# ==============================================================================

import logging
import os
import sys
import warnings
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog
from structlog.types import EventDict, Processor

from informity.config import settings
from informity.utils.directory_utils import ensure_directory

_STATUS_BY_LEVEL = {
    'critical': 'error',
    'error': 'error',
    'warning': 'warning',
    'info': 'ok',
    'debug': 'ok',
}
_CONSOLE_SUPPRESS_ENV_VAR = 'INFORMITY_SUPPRESS_CONSOLE_LOGS'
_LOG_ROTATION_WHEN = 'midnight'
_LOG_ROTATION_INTERVAL_DAYS = 1
_LOG_RETENTION_DAYS = 7
_LOG_RETENTION_LABEL = f'{_LOG_RETENTION_DAYS}_days'

# ==============================================================================
# Log File Paths
# ==============================================================================

def _get_log_paths() -> tuple[Path, Path, Path]:
    # Returns (app_log_path, error_log_path, mcp_log_path) based on settings.
    # Ensures logs_dir exists before returning paths.
    logs_dir = settings.logs_dir
    if logs_dir is None:
        raise RuntimeError('logs_dir is not configured')

    ensure_directory(logs_dir)

    app_log   = logs_dir / 'app.log'
    error_log = logs_dir / 'app.error.log'

    mcp_log = logs_dir / 'app.mcp.log'
    return app_log, error_log, mcp_log


# ==============================================================================
# Structlog Processors
# ==============================================================================

def _add_timestamp(logger: logging.Logger | None, method_name: str, event_dict: EventDict) -> EventDict:
    # Add ISO-8601 timestamp to each log entry.
    event_dict['timestamp'] = datetime.now(UTC).isoformat()
    return event_dict


def _add_operation_context(logger: logging.Logger | None, method_name: str, event_dict: EventDict) -> EventDict:
    # Add operation context based on logger name (module path).
    # This helps identify which component generated the log (scanner, indexer, llm, etc.)
    # Logger can be None when ProcessorFormatter processes foreign stdlib log records (e.g. aiosqlite).
    if logger is None:
        return event_dict
    logger_name = logger.name
    if logger_name.startswith('informity.'):
        parts = logger_name.split('.')
        if len(parts) >= 2:
            # Extract module name (e.g., 'scanner', 'indexer', 'llm')
            module = parts[1]
            event_dict['module'] = module
        if len(parts) >= 3:
            # Extract submodule (e.g., 'extractors', 'embedder')
            submodule = parts[2]
            event_dict['submodule'] = submodule

    return event_dict


def _normalize_event_contract(logger: logging.Logger | None, method_name: str, event_dict: EventDict) -> EventDict:
    # Normalize structured log fields to a stable contract for dashboards/queries.
    level = str(event_dict.get('level') or '').lower()
    event = str(event_dict.get('event') or method_name)

    # Stable operation identifier: defaults to event name if missing.
    operation = event_dict.get('operation')
    if not operation:
        event_dict['operation'] = event

    # Stable component: use module first, then logger prefix fallback.
    component = event_dict.get('component')
    if not component:
        module_name = event_dict.get('module')
        if isinstance(module_name, str) and module_name:
            event_dict['component'] = module_name
        elif logger is not None and logger.name.startswith('informity.'):
            parts = logger.name.split('.')
            if len(parts) >= 2:
                event_dict['component'] = parts[1]

    # Stable status for easy filtering/alerting.
    if not event_dict.get('status'):
        event_dict['status'] = _STATUS_BY_LEVEL.get(level, 'ok')

    # Normalize duration fields into duration_ms.
    if 'duration_ms' not in event_dict:
        elapsed_ms = event_dict.get('elapsed_ms')
        if isinstance(elapsed_ms, int | float):
            event_dict['duration_ms'] = round(float(elapsed_ms), 2)
        else:
            elapsed_s = event_dict.get('elapsed_s')
            generation_seconds = event_dict.get('generation_seconds')
            if isinstance(elapsed_s, int | float):
                event_dict['duration_ms'] = round(float(elapsed_s) * 1000.0, 2)
            elif isinstance(generation_seconds, int | float):
                event_dict['duration_ms'] = round(float(generation_seconds) * 1000.0, 2)

    # Replace ad-hoc "msg" with explicit "message" for consistency.
    if 'msg' in event_dict and 'message' not in event_dict:
        event_dict['message'] = event_dict.pop('msg')

    # Ensure event key is always present and stringified.
    event_dict['event'] = event
    return event_dict


class _SuppressDoclingWarningsFilter(logging.Filter):
    """Filter to suppress deprecation warnings from docling_core about strict_text parameter."""
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress warnings from docling_core about strict_text deprecation
        if record.name.startswith('docling_core') and record.levelno == logging.WARNING:
            message = record.getMessage().lower()
            if 'strict_text' in message and 'deprecated' in message:
                return False
        return True


class _McpOnlyFilter(logging.Filter):
    """Allow only MCP namespace records into dedicated MCP log file."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = str(getattr(record, 'name', '') or '')
        return name.startswith('informity.mcp')


# ==============================================================================
# Log level mapping
# ==============================================================================

_LEVEL_MAP = {
    'debug':    logging.DEBUG,
    'info':     logging.INFO,
    'warning':  logging.WARNING,
    'error':    logging.ERROR,
    'critical': logging.CRITICAL,
}

# Third-party loggers that are noisy at DEBUG; we always set them to WARNING.
_NOISY_LOGGERS = ('aiosqlite', 'asyncio', 'urllib3', 'httpx', 'httpcore')


# ==============================================================================
# Third-Party Logger Suppression Utilities
# ==============================================================================

class _TemporaryLoggerSuppression:
    """
    Context manager for temporarily suppressing a logger during a specific operation.

    Use this when you need to suppress noisy third-party library logs during model loading
    or other operations, but want to restore the original level afterward.

    Example:
        with suppress_logger_temporarily('transformers.modeling_utils', logging.ERROR):
            model = load_model()
    """
    def __init__(self, logger_name: str, temporary_level: int):
        self.logger_name = logger_name
        self.temporary_level = temporary_level
        self.logger: logging.Logger | None = None
        self.original_level: int | None = None

    def __enter__(self) -> '_TemporaryLoggerSuppression':
        self.logger = logging.getLogger(self.logger_name)
        # Store original level (use NOTSET if not explicitly set)
        self.original_level = self.logger.level if self.logger.level != logging.NOTSET else logging.NOTSET
        self.logger.setLevel(self.temporary_level)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self.logger is not None and self.original_level is not None:
            # Restore original level. If it was NOTSET, restore to NOTSET so logger
            # inherits from parent logger's level (consistent with original behavior).
            self.logger.setLevel(self.original_level)


def suppress_logger_temporarily(logger_name: str, temporary_level: int) -> _TemporaryLoggerSuppression:
    """
    Context manager for temporarily suppressing a logger during a specific operation.

    Args:
        logger_name: Name of the logger to suppress (e.g., 'transformers.modeling_utils')
        temporary_level: Log level to set during suppression (e.g., logging.ERROR)

    Returns:
        Context manager that restores original logger level on exit.

    Example:
        with suppress_logger_temporarily('transformers.modeling_utils', logging.ERROR):
            # Example: suppress warnings during model loading
            pass
    """
    return _TemporaryLoggerSuppression(logger_name, temporary_level)


# ==============================================================================
# Logging Setup
# ==============================================================================

_logging_configured = False


def configure_logging() -> None:
    # Configure structlog to write to console and files.
    # This should be called once during application startup, before any other
    # modules log anything.  Idempotent — repeated calls are no-ops.
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    # Check if console logging should be suppressed (for CLI tools)
    suppress_console = os.environ.get(_CONSOLE_SUPPRESS_ENV_VAR) == '1'

    # ==========================================================================
    # Third-Party Warning Suppression (Python warnings module)
    # ==========================================================================
    # Suppress harmless warnings from third-party libraries that we cannot control.
    # These are applied globally via Python's warnings.filterwarnings().

    # Suppress SyntaxWarnings from third-party libraries (e.g., pysbd with Python 3.13)
    # These are harmless warnings about invalid escape sequences in third-party code
    # Note: This is also applied in main.py before imports to catch early warnings
    # Suppress all SyntaxWarnings (they're all from third-party code we can't fix)
    warnings.filterwarnings('ignore', category=SyntaxWarning)

    # Suppress deprecation warning from docling_core about strict_text parameter
    # This is an internal deprecation in docling that we can't control
    # Filter by both message pattern and module to be more specific
    warnings.filterwarnings(
        'ignore',
        message='.*strict_text.*deprecated.*',
        category=DeprecationWarning,
        module='docling_core',
    )

    app_log_path, error_log_path, mcp_log_path = _get_log_paths()

    # Resolve application log level from config (default: INFO to reduce noise).
    level_name = (settings.log_level or 'info').strip().lower()
    app_level  = _LEVEL_MAP.get(level_name, logging.INFO)

    # -- Standard library logging setup ----------------------------------------
    # Configure the root logger and handlers for file output

    # General log handler (app level; no DEBUG from third-party libs in file)
    general_handler = TimedRotatingFileHandler(
        filename     = str(app_log_path),
        when         = _LOG_ROTATION_WHEN,
        interval     = _LOG_ROTATION_INTERVAL_DAYS,
        backupCount  = _LOG_RETENTION_DAYS,
        encoding     = 'utf-8',
    )
    general_handler.setLevel(app_level)
    # Formatter will be set after structlog configuration

    # Error log handler (ERROR and CRITICAL only)
    error_handler = TimedRotatingFileHandler(
        filename     = str(error_log_path),
        when         = _LOG_ROTATION_WHEN,
        interval     = _LOG_ROTATION_INTERVAL_DAYS,
        backupCount  = _LOG_RETENTION_DAYS,
        encoding     = 'utf-8',
    )
    error_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL
    # Formatter will be set after structlog configuration

    # MCP log handler (INFO and above, MCP namespace only)
    mcp_handler = TimedRotatingFileHandler(
        filename=str(mcp_log_path),
        when=_LOG_ROTATION_WHEN,
        interval=_LOG_ROTATION_INTERVAL_DAYS,
        backupCount=_LOG_RETENTION_DAYS,
        encoding='utf-8',
    )
    mcp_handler.setLevel(logging.INFO)
    mcp_handler.addFilter(_McpOnlyFilter())

    # Console handler (INFO and above for readability) - skip if suppressed
    console_handler = None
    if not suppress_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(max(app_level, logging.INFO))
        # Formatter will be set after structlog configuration

    # Configure root logger (app level so third-party DEBUG is not propagated)
    root_logger = logging.getLogger()
    root_logger.setLevel(app_level)
    # Add filter to suppress docling_core deprecation warnings
    root_logger.addFilter(_SuppressDoclingWarningsFilter())
    # Duplication policy: ERROR/CRITICAL are intentionally present in both streams:
    # - app.log for chronological full-app narrative
    # - app.error.log for fast error-only triage
    root_logger.addHandler(general_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(mcp_handler)
    if console_handler:
        root_logger.addHandler(console_handler)

    # ==========================================================================
    # Third-Party Logger Suppression (Permanent)
    # ==========================================================================
    # Set log levels for third-party loggers that are noisy at DEBUG.
    # These are permanent settings applied at startup.

    # Standard noisy loggers: set to WARNING to prevent DEBUG spam
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Docling-specific logger configurations:
    # - docling_core.types.doc.document: Suppress deprecation warnings (strict_text)
    #   These are logged through Python's warnings system which structlog captures
    logging.getLogger('docling_core.types.doc.document').setLevel(logging.ERROR)

    # - docling.pipeline: Non-fatal validation errors (malformed hyperlinks, unknown fonts)
    #   These don't prevent extraction - docling continues processing and returns results.
    #   These are informational logs about PDF quality issues, not extraction failures.
    #   Set to WARNING to reduce noise while preserving actual extraction failures
    #   (which would be logged at CRITICAL or as exceptions).
    logging.getLogger('docling.pipeline').setLevel(logging.WARNING)

    # -- Structlog configuration -----------------------------------------------
    # Configure structlog to integrate with Python's standard logging system.
    # This allows structlog logs to go through our file handlers.

    # Shared processors for all loggers (before formatting)
    # These add metadata but don't format the output
    pre_processors: list[Processor] = [
        # Merge request/task-bound contextvars (request_id, scan_id, etc.) first.
        structlog.contextvars.merge_contextvars,
        # Add timestamp and context
        _add_timestamp,
        _add_operation_context,
        _normalize_event_contract,
        # Add stack info for exceptions
        structlog.processors.format_exc_info,
        # Add logger name (from stdlib module)
        structlog.stdlib.add_logger_name,
        # Add log level (from stdlib module)
        structlog.stdlib.add_log_level,
    ]

    # Full processor chain for structlog.configure
    # wrap_for_formatter converts event_dict to (args, kw) tuple for standard logging
    configure_processors: list[Processor] = (
        pre_processors +
        [structlog.stdlib.ProcessorFormatter.wrap_for_formatter]
    )

    # Configure structlog to use standard library logging
    structlog.configure(
        processors    = configure_processors,
        wrapper_class = structlog.stdlib.BoundLogger,
        context_class = dict,
        logger_factory = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True,
    )

    # Configure structlog's formatter for file handlers (JSON format)
    # This produces machine-parseable structured logs for ops/perf tooling.
    # foreign_pre_chain should be pre_processors (without wrap_for_formatter)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor         = structlog.processors.JSONRenderer(sort_keys=True),
        foreign_pre_chain = pre_processors,
    )

    # Configure structlog's formatter for console (pretty colors)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor         = structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain = pre_processors,
    )

    # Apply formatters to handlers
    general_handler.setFormatter(file_formatter)
    error_handler.setFormatter(file_formatter)
    mcp_handler.setFormatter(file_formatter)
    if console_handler:
        console_handler.setFormatter(console_formatter)

    # Log that logging is configured
    log = structlog.get_logger(__name__)
    log.info(
        'logging_configured',
        app_log     = str(app_log_path),
        error_log   = str(error_log_path),
        log_level   = level_name,
        rotation    = 'daily',
        retention   = _LOG_RETENTION_LABEL,
    )
