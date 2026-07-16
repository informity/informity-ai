# ==============================================================================
# Informity AI — numeric conversion helpers
# Shared coercion utilities for permissive int/float parsing.
# ==============================================================================

"""Small numeric coercion helpers."""


def safe_int(value: object, default: int = 0) -> int:
    """Coerce a value to int when possible, else return the default."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def safe_float(value: object, default: float = 0.0) -> float:
    """Coerce a value to float when possible, else return the default."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default
