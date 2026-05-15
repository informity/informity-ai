# ==============================================================================
# Informity AI — Application Version
# Single source of truth for runtime app version.
# ==============================================================================

import os
import tomllib
from importlib import metadata
from pathlib import Path

_PACKAGE_NAME = 'informity'
_ENV_VERSION_KEY = 'INFORMITY_APP_VERSION'
_DEFAULT_FALLBACK_VERSION = '0.12.1'


def _resolve_pyproject_version() -> str | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        pyproject_path = parent / 'pyproject.toml'
        if not pyproject_path.exists():
            continue
        try:
            with pyproject_path.open('rb') as file_obj:
                data = tomllib.load(file_obj)
            version = data.get('project', {}).get('version')
            if isinstance(version, str) and version.strip():
                return version.strip()
        except (OSError, tomllib.TOMLDecodeError):
            return None
    return None


def _resolve_app_version() -> str:
    env_version = str(os.getenv(_ENV_VERSION_KEY, '')).strip()
    if env_version:
        return env_version

    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        pyproject_version = _resolve_pyproject_version()
        if pyproject_version:
            return pyproject_version
        return _DEFAULT_FALLBACK_VERSION


APP_VERSION = _resolve_app_version()
