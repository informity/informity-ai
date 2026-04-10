# ==============================================================================
# Informity AI — NLP Heuristics
# Minimal retained heuristic pattern(s) used by production routing paths.
# ==============================================================================

import re

BY_PER_YEAR_PATTERN = re.compile(r'\b(?:by|per)\s+year\b', re.IGNORECASE)

__all__ = ['BY_PER_YEAR_PATTERN']
