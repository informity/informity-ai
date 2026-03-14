# ==============================================================================
# Informity AI — Configuration Reference Metadata
# Defines groups, descriptions, and default values for application constants
# and defaults that are not configurable via environment variables.
# Used by GET /api/config/reference for the Configuration page.
# ==============================================================================

from informity.api.schemas import ConfigReferenceResponse, ConstantGroup, ConstantItem
from informity.config import (
    EXCLUDE_DEVELOPER_PATTERNS,
    EXCLUDE_MACOS_SYSTEM_PATTERNS,
    settings,
)
from informity.scanner.extractors.text_utils import MAX_FILE_SIZE_BYTES
from informity.scanner.watcher import DEBOUNCE_SECONDS

# Local constants for Configuration page reference (RAG coverage retrieval parameters).
_COVERAGE_MAX_FILES           = 10
_COVERAGE_SCORE_GAP_THRESHOLD = 2.0
_COVERAGE_MIN_FILES           = 2


# ------------------------------------------------------------------------------
# Group definitions: title, description, and list of (name, default, description).
# ------------------------------------------------------------------------------

_GROUPS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        'Preset Exclusion Patterns',
        'Patterns used when "Exclude common macOS system and application data" or "Exclude common developer data" are enabled in Settings. These are applied automatically based on the corresponding checkboxes.',
        [
            (
                'EXCLUDE_MACOS_SYSTEM_PATTERNS',
                ', '.join(EXCLUDE_MACOS_SYSTEM_PATTERNS),
                'Patterns for macOS system files and directories (.DS_Store, Library, *.app, etc.). Applied when exclude_macos_system is true.',
            ),
            (
                'EXCLUDE_DEVELOPER_PATTERNS',
                ', '.join(EXCLUDE_DEVELOPER_PATTERNS),
                'Patterns for developer directories and files (.git, node_modules, __pycache__, etc.). Applied when exclude_developer_data is true.',
            ),
        ],
    ),
    (
        'File Processing Limits',
        'Maximum file sizes enforced during scanning and extraction. These prevent memory exhaustion on very large files.',
        [
            (
                'MAX_FILE_SIZE_BYTES',
                f'{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB',
                'Maximum file size for text, PDF, and HTML files. Files larger than this are skipped during scanning.',
            ),
        ],
    ),
    (
        'File Watcher',
        'Settings for the filesystem watcher that monitors watched directories for changes.',
        [
            (
                'DEBOUNCE_SECONDS',
                str(DEBOUNCE_SECONDS),
                'Debounce delay in seconds before processing file change events. Prevents excessive CPU usage when many files change rapidly.',
            ),
        ],
    ),
    (
        'Operation State',
        'Thresholds for detecting and handling long-running operations.',
        [
            (
                'SCAN_STALE_THRESHOLD_SECONDS',
                str(settings.scan_stale_threshold_seconds),
                'Age in seconds after which a running scan is considered stuck and automatically marked as failed. Prevents scans from appearing stuck indefinitely.',
            ),
        ],
    ),
    (
        'RAG Coverage Retrieval',
        'Parameters for coverage-mode queries (e.g., "all years", "every document"). These control how many files and chunks are retrieved for comprehensive answers.',
        [
            (
                'COVERAGE_MAX_FILES',
                str(_COVERAGE_MAX_FILES),
                'Maximum number of files to include in coverage round-robin retrieval. Hard ceiling — score-gap detection usually cuts earlier.',
            ),
            (
                'COVERAGE_SCORE_GAP_THRESHOLD',
                str(_COVERAGE_SCORE_GAP_THRESHOLD),
                'L2 distance gap threshold for coverage file selection. When the gap between consecutive files exceeds this value, the file list is cut there to remove irrelevant files.',
            ),
            (
                'COVERAGE_MIN_FILES',
                str(_COVERAGE_MIN_FILES),
                'Minimum number of files to always keep in coverage results, regardless of score gaps.',
            ),
        ],
    ),
]


def get_config_reference_response() -> ConfigReferenceResponse:
    # Build the config reference response with constant groups.
    groups: list[ConstantGroup] = []
    for title, description, constants in _GROUPS:
        items = [
            ConstantItem(name=name, default=default, description=desc)
            for name, default, desc in constants
        ]
        groups.append(ConstantGroup(title=title, description=description, constants=items))
    return ConfigReferenceResponse(groups=groups)
