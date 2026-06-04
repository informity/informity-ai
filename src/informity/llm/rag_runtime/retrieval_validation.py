import math


def _normalize_relevance_score(raw_score: float) -> float:
    """
    Convert CrossEncoder raw logit (-10..10) to 0-1 relevance for display.
    Uses sigmoid so 0 -> 0.5, positive -> higher, negative -> lower.
    """
    try:
        numeric_score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(-numeric_score))
    except OverflowError:
        return 0.0 if numeric_score < 0 else 1.0
