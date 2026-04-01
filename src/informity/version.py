# ==============================================================================
# Informity AI — Application Version
# Single source of truth for runtime app version.
# ==============================================================================

from importlib import metadata

_PACKAGE_NAME = 'informity'
_LOCAL_FALLBACK_VERSION = '0.8.2-local'


def _resolve_app_version() -> str:
    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return _LOCAL_FALLBACK_VERSION


APP_VERSION = _resolve_app_version()
