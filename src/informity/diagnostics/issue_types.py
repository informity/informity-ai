# ==============================================================================
# Informity AI — Diagnostics Issue Types
# Enum defining types of issues detected during diagnostics evaluation
# ==============================================================================

from enum import StrEnum


class IssueType(StrEnum):
    """
    Types of issues that can be detected during diagnostics evaluation.

    Used for issue detection and reporting in diagnostics pipeline.
    """
    retrieval_failure = 'retrieval_failure'        # Zero chunks retrieved
    insufficient_retrieval = 'insufficient_retrieval'  # < 3 chunks for complex queries
    empty_answer = 'empty_answer'                  # Answer is empty/whitespace-only
    refusal_bias = 'refusal_bias'                 # Model refuses to answer (detected patterns)
    timeout = 'timeout'                           # Generation timeout occurred
    very_short_answer = 'very_short_answer'       # Answer length < 20 chars for non-simple queries
    unsupported_claims_detected = 'unsupported_claims_detected'  # Grounding verifier detected unsupported claims
