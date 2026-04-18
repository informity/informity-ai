# ==============================================================================
# Informity AI — Diagnostics Observer
# Metrics collection and issue detection for diagnostics evaluation
# ==============================================================================

import re
from dataclasses import dataclass

from informity.diagnostics.issue_types import IssueType
from informity.llm.types import QueryType

_RAG_QUERY_TYPES = (QueryType.FOCUSED, QueryType.COVERAGE)
_FILENAME_ANCHORED_QUERY_PATTERN = re.compile(
    r'\b(?:what|which|summari[sz]e|describe)\b.*\b[\w\-\s()]+\.[a-z0-9]{2,5}\b',
    re.IGNORECASE,
)
_INSUFFICIENT_RETRIEVAL_MIN_CHUNKS = 3
_COMPLEX_QUERY_MIN_WORDS = 10
_SIMPLE_QUERY_TYPE = QueryType.SIMPLE
_VERY_SHORT_ANSWER_MAX_CHARS = 20
_OBSERVER_HEURISTIC_PROFILE = 'diagnostics_observer_v1'
_EVIDENCE_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9]+')
_CLAIM_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
_BULLET_PREFIX_PATTERN = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+')
_EVIDENCE_MIN_CLAIM_CHARS = 20
_EVIDENCE_CLAIM_SIGNAL_PATTERN = re.compile(
    r'(?:\d|\$|%|\b(?:total|balance|amount|conflict|difference|delta|increase|decrease)\b)',
    re.IGNORECASE,
)
_LIKELY_REASON_PATTERN = re.compile(
    r'(?:\blikely\s+reason\b|\bprobable\s+reason\b|\bpossible\s+reason\b|\breason\s*:)',
    re.IGNORECASE,
)
_NOT_FOUND_PATTERN = re.compile(r'\bnot found\b', re.IGNORECASE)
_EVIDENCE_STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'onto', 'about',
    'were', 'was', 'are', 'have', 'has', 'had', 'their', 'there', 'which', 'while',
    'such', 'than', 'then', 'also', 'only', 'using', 'across', 'based', 'records',
    'record', 'document', 'documents', 'file', 'files', 'indexed',
}


@dataclass
class EvalMetrics:
    """
    Evaluation metrics for a single query response.

    Fields use OTel-style naming conventions (via openinference-semantic-conventions)
    for diagnostics consistency across observers.
    """
    chat_id: str
    question: str
    model_filename: str
    query_type: QueryType  # 'focused', 'coverage', 'metadata', or 'simple' (from _resolve_handler_for_classification in llm/rag.py)
    raw_chunks_count: int  # Candidates from vector search (0 for metadata/simple queries)
    sources_count: int  # Parent chunks used (0 for metadata queries)
    generation_seconds: float
    answer_length: int
    timeout_occurred: bool
    has_empty_answer: bool
    has_refusal_pattern: bool
    unsupported_claim_count: int = 0
    evidence_coverage_rate: float = 0.0
    not_found_count: int = 0


def detect_issues(answer: str, metrics: EvalMetrics) -> list[IssueType]:
    """
    Detect issues in a query response based on metrics.

    Uses diagnostics-only heuristic detection patterns (non-routing, non-blocking).
    Returns list of IssueType enum values.

    Args:
        answer: The generated answer text
        metrics: EvalMetrics dataclass with response metrics

    Returns:
        List of IssueType enum values for detected issues
    """
    issues: list[IssueType] = []

    # Retrieval failure: zero chunks retrieved (only for RAG queries)
    if metrics.query_type in _RAG_QUERY_TYPES and metrics.raw_chunks_count == 0:
        issues.append(IssueType.retrieval_failure)

    filename_anchored_question = bool(
        _FILENAME_ANCHORED_QUERY_PATTERN.search(metrics.question)
    )

    # Insufficient retrieval: < 3 chunks for complex queries.
    # Do not flag focused single-file lookups when at least one source exists.
    if (
        metrics.query_type in _RAG_QUERY_TYPES
        and 0 < metrics.raw_chunks_count < _INSUFFICIENT_RETRIEVAL_MIN_CHUNKS
        and (len(metrics.question.split()) > _COMPLEX_QUERY_MIN_WORDS or metrics.query_type == QueryType.COVERAGE)
        and not (metrics.query_type == QueryType.FOCUSED and filename_anchored_question and metrics.sources_count > 0)
    ):
        # Only flag if query seems complex (long question or coverage type)
        issues.append(IssueType.insufficient_retrieval)

    # Empty answer: answer is empty or whitespace-only
    if metrics.has_empty_answer or (not answer or not answer.strip()):
        issues.append(IssueType.empty_answer)

    # Refusal bias: model refuses to answer (detected patterns)
    if metrics.has_refusal_pattern:
        issues.append(IssueType.refusal_bias)

    # Timeout: generation timeout occurred
    if metrics.timeout_occurred:
        issues.append(IssueType.timeout)

    # Very short answer: < 20 chars for non-simple queries
    if (
        metrics.query_type != _SIMPLE_QUERY_TYPE
        and metrics.answer_length > 0
        and metrics.answer_length < _VERY_SHORT_ANSWER_MAX_CHARS
    ):
        issues.append(IssueType.very_short_answer)
    if metrics.unsupported_claim_count > 0:
        issues.append(IssueType.unsupported_claims_detected)

    return issues


def populate_signals(answer: str, metrics: EvalMetrics) -> dict:
    """
    Extract quality signals from answer and metrics.

    Returns a dictionary of quality signals that can be used for analysis.

    Args:
        answer: The generated answer text
        metrics: EvalMetrics dataclass with response metrics

    Returns:
        Dictionary of quality signals
    """
    signals: dict = {
        'heuristic_profile': _OBSERVER_HEURISTIC_PROFILE,
        'has_retrieval': metrics.raw_chunks_count > 0,
        'has_sources': metrics.sources_count > 0,
        'answer_length': metrics.answer_length,
        'generation_time': metrics.generation_seconds,
        'query_type': metrics.query_type,
    }

    # Add answer quality signals
    if answer:
        signals['has_markdown'] = bool(re.search(r'[#*\[\]`]', answer))
        signals['has_list'] = bool(re.search(r'^\s*[-*•]|\d+\.', answer, re.MULTILINE))
        signals['has_table'] = bool(re.search(r'\|.*\|', answer))
        signals['word_count'] = len(answer.split())
    else:
        signals['has_markdown'] = False
        signals['has_list'] = False
        signals['has_table'] = False
        signals['word_count'] = 0

    return signals


def estimate_evidence_metrics(
    *,
    answer: str,
    source_texts: list[str],
) -> tuple[int, float, int]:
    """
    Estimate unsupported claim count and evidence coverage using lexical overlap.

    Returns:
        tuple[unsupported_claim_count, evidence_coverage_rate, not_found_count]
    """

    answer_text = str(answer or '').strip()
    if not answer_text:
        return 0, 0.0, 0

    not_found_count = len(_NOT_FOUND_PATTERN.findall(answer_text))
    claims = _extract_claim_units(answer_text)
    if not claims:
        return 0, (1.0 if source_texts else 0.0), not_found_count

    source_token_sets = [_tokenize_evidence_text(text) for text in source_texts if str(text or '').strip()]
    source_token_sets = [tokens for tokens in source_token_sets if tokens]
    if not source_token_sets:
        return len(claims), 0.0, not_found_count

    evaluated_claims = 0
    supported_claims = 0
    for claim in claims:
        claim_tokens = _tokenize_evidence_text(claim)
        if len(claim_tokens) < 3:
            continue
        if not _EVIDENCE_CLAIM_SIGNAL_PATTERN.search(claim):
            continue
        has_numeric_signal = any(any(ch.isdigit() for ch in token) for token in claim_tokens)
        # Explanatory "likely reason" text is useful but often inferential.
        # Keep unsupported-claim counting focused on factual extractive claims.
        if not has_numeric_signal and _LIKELY_REASON_PATTERN.search(claim):
            continue
        evaluated_claims += 1
        threshold = 1 if has_numeric_signal else 2
        max_overlap = 0
        for source_tokens in source_token_sets:
            overlap = len(claim_tokens.intersection(source_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
        if max_overlap >= threshold:
            supported_claims += 1

    if evaluated_claims <= 0:
        return 0, 1.0, not_found_count

    unsupported_claim_count = max(evaluated_claims - supported_claims, 0)
    evidence_coverage_rate = float(supported_claims) / float(evaluated_claims)
    return unsupported_claim_count, round(evidence_coverage_rate, 3), not_found_count


def _extract_claim_units(answer: str) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        normalized = re.sub(r'\s+', ' ', str(candidate or '').strip())
        if len(normalized) < _EVIDENCE_MIN_CLAIM_CHARS:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        claims.append(normalized)

    for line in answer.splitlines():
        if _BULLET_PREFIX_PATTERN.match(line):
            _add(_BULLET_PREFIX_PATTERN.sub('', line))

    for segment in _CLAIM_SPLIT_PATTERN.split(re.sub(r'\s+', ' ', answer)):
        _add(segment)

    return claims


def _tokenize_evidence_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _EVIDENCE_TOKEN_PATTERN.findall(str(text or '')):
        lowered = token.casefold()
        if len(lowered) < 3:
            continue
        if lowered in _EVIDENCE_STOPWORDS:
            continue
        tokens.add(lowered)
    return tokens
