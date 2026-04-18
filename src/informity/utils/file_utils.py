"""
Informity AI — file helper utilities.
"""


def normalize_extension(extension: str | None) -> str:
    """
    Normalize extension-like strings to a leading-dot form.
    Returns empty string for empty input.
    """
    normalized = str(extension or '').strip()
    if not normalized:
        return ''
    return normalized if normalized.startswith('.') else f'.{normalized}'

