# ==============================================================================
# Informity AI — Path Utilities (v2)
# Standardized path resolution and normalization
# ==============================================================================

from pathlib import Path


def normalize_path(path: Path | str, expand_user: bool = True) -> Path:
    """
    Normalize a path: expand user home directory and resolve to absolute.

    Args:
        path: Path string or Path object
        expand_user: Whether to expand ~ to user home directory (default: True)

    Returns:
        Resolved absolute Path object
    """
    path_obj = Path(path)
    if expand_user:
        path_obj = path_obj.expanduser()
    return path_obj.resolve()


def normalize_paths(paths: list[Path | str], expand_user: bool = True) -> list[Path]:
    """
    Normalize a list of paths.

    Args:
        paths: List of path strings or Path objects
        expand_user: Whether to expand ~ to user home directory (default: True)

    Returns:
        List of resolved absolute Path objects
    """
    return [normalize_path(p, expand_user=expand_user) for p in paths]


def resolve_and_check_path(path: Path | str) -> tuple[Path, bool]:
    """
    Resolve a path (expand user home directory and resolve to absolute) and check if it exists.

    This is a convenience wrapper around normalize_path() that also checks existence.

    Args:
        path: Path string or Path object

    Returns:
        tuple[Path, bool]: (resolved_path, exists)
    """
    resolved = normalize_path(path, expand_user=True)
    return resolved, resolved.exists()
