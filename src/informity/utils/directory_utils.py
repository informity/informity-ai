# ==============================================================================
# Informity AI — Directory Utilities (v2)
# Standardized directory creation and management
# ==============================================================================

from pathlib import Path


def ensure_directory(path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure
        parents: Create parent directories if they don't exist (default: True)
        exist_ok: Don't raise error if directory already exists (default: True)

    Returns:
        Path object of the created/existing directory
    """
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path


def ensure_directories(paths: list[Path], parents: bool = True, exist_ok: bool = True) -> None:
    """
    Ensure multiple directories exist, creating them if necessary.

    Args:
        paths: List of directory paths to ensure
        parents: Create parent directories if they don't exist (default: True)
        exist_ok: Don't raise error if directories already exist (default: True)
    """
    for path in paths:
        ensure_directory(path, parents=parents, exist_ok=exist_ok)


def ensure_file_directory(file_path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Ensure the parent directory of a file path exists.

    Convenience function for ensuring a file's parent directory exists before writing.

    Args:
        file_path: File path whose parent directory should be ensured
        parents: Create parent directories if they don't exist (default: True)
        exist_ok: Don't raise error if directory already exists (default: True)

    Returns:
        Path object of the parent directory
    """
    return ensure_directory(file_path.parent, parents=parents, exist_ok=exist_ok)
