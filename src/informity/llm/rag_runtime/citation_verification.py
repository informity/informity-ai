from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from informity.api.schemas import ChatSourceReference

_EVIDENCE_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9]+')
_EVIDENCE_STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'were', 'been', 'have', 'has',
    'are', 'was', 'but', 'not', 'out', 'you', 'your', 'their', 'they', 'them', 'then', 'when',
    'what', 'which', 'who', 'where', 'why', 'how', 'does', 'did', 'can', 'could', 'should',
}
_MEANINGFUL_NUMERIC_PATTERN = re.compile(r'(?<!\w)\d+(?:\.\d+)?(?!\w)')


@dataclass(frozen=True)
class CitationVerificationResult:
    evaluated_claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    evidence_coverage_rate: float
    verified_source_count: int

    @property
    def has_thin_evidence(self) -> bool:
        return self.supported_claim_count <= 0 < self.evaluated_claim_count

    @property
    def should_fail_closed(self) -> bool:
        return self.has_thin_evidence


def filter_verified_sources(
    sources: Sequence[ChatSourceReference],
    *,
    answer_text: str,
) -> list[ChatSourceReference]:
    text = str(answer_text or '').strip()
    if not text:
        return list(sources)
    if len(text.split()) <= 3:
        return list(sources)

    claims = _extract_claim_units(text)
    if not claims:
        return []

    verified_sources: list[ChatSourceReference] = []
    for source in sources:
        chunk_text = str(source.chunk_preview or '').strip()
        if not chunk_text:
            continue
        if _source_supports_any_claim(chunk_text, claims):
            verified_sources.append(source)
    return verified_sources


def assess_answer_support(
    *,
    answer_text: str,
    source_texts: Sequence[str],
) -> CitationVerificationResult:
    text = str(answer_text or '').strip()
    if not text:
        return CitationVerificationResult(
            evaluated_claim_count=0,
            supported_claim_count=0,
            unsupported_claim_count=0,
            evidence_coverage_rate=0.0,
            verified_source_count=0,
        )
    if len(text.split()) <= 3:
        return CitationVerificationResult(
            evaluated_claim_count=0,
            supported_claim_count=0,
            unsupported_claim_count=0,
            evidence_coverage_rate=1.0,
            verified_source_count=len([source_text for source_text in source_texts if str(source_text or '').strip()]),
        )

    claims = _extract_claim_units(text)
    if not claims:
        return CitationVerificationResult(
            evaluated_claim_count=0,
            supported_claim_count=0,
            unsupported_claim_count=0,
            evidence_coverage_rate=1.0,
            verified_source_count=0,
        )

    source_token_sets = [
        _tokenize_evidence_text(source_text)
        for source_text in source_texts
        if str(source_text or '').strip()
    ]
    source_token_sets = [tokens for tokens in source_token_sets if tokens]
    if not source_token_sets:
        evaluated_claim_count = len(claims)
        return CitationVerificationResult(
            evaluated_claim_count=evaluated_claim_count,
            supported_claim_count=0,
            unsupported_claim_count=evaluated_claim_count,
            evidence_coverage_rate=0.0,
            verified_source_count=0,
        )

    evaluated_claims = 0
    supported_claims = 0
    for claim in claims:
        claim_tokens = _tokenize_evidence_text(claim)
        if len(claim_tokens) < 3:
            continue
        evaluated_claims += 1
        has_numeric_signal = bool(_MEANINGFUL_NUMERIC_PATTERN.search(claim))
        threshold = 1 if has_numeric_signal else 2
        max_overlap = 0
        for source_tokens in source_token_sets:
            overlap = len(claim_tokens.intersection(source_tokens))
            max_overlap = max(max_overlap, overlap)
        if max_overlap >= threshold:
            supported_claims += 1

    if evaluated_claims <= 0:
        return CitationVerificationResult(
            evaluated_claim_count=0,
            supported_claim_count=0,
            unsupported_claim_count=0,
            evidence_coverage_rate=1.0,
            verified_source_count=0,
        )

    unsupported_claim_count = max(evaluated_claims - supported_claims, 0)
    evidence_coverage_rate = float(supported_claims) / float(evaluated_claims)
    verified_source_count = len(filter_verified_sources_from_claims(sources=source_texts, claims=claims))
    return CitationVerificationResult(
        evaluated_claim_count=evaluated_claims,
        supported_claim_count=supported_claims,
        unsupported_claim_count=unsupported_claim_count,
        evidence_coverage_rate=round(evidence_coverage_rate, 3),
        verified_source_count=verified_source_count,
    )


def filter_verified_sources_from_claims(
    *,
    sources: Sequence[str],
    claims: Sequence[str],
) -> list[str]:
    verified_sources: list[str] = []
    for source_text in sources:
        normalized_source = str(source_text or '').strip()
        if not normalized_source:
            continue
        if _source_supports_any_claim(normalized_source, claims):
            verified_sources.append(normalized_source)
    return verified_sources


def _source_supports_any_claim(source_text: str, claims: Sequence[str]) -> bool:
    source_tokens = _tokenize_evidence_text(source_text)
    if not source_tokens:
        return False
    for claim in claims:
        claim_tokens = _tokenize_evidence_text(claim)
        if len(claim_tokens) < 3:
            continue
        has_numeric_signal = bool(_MEANINGFUL_NUMERIC_PATTERN.search(claim))
        threshold = 1 if has_numeric_signal else 2
        if len(claim_tokens.intersection(source_tokens)) >= threshold:
            return True
    return False


def _extract_claim_units(answer: str) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        normalized = re.sub(r'\s+', ' ', str(candidate or '').strip())
        if len(normalized) < 12:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        claims.append(normalized)

    bullet_pattern = re.compile(r'(?m)^\s*[-*+]\s+(.*\S)\s*$')
    numbered_pattern = re.compile(r'(?m)^\s*\d+\.\s+(.*\S)\s*$')

    for match in bullet_pattern.finditer(answer):
        _add(match.group(1))
    for match in numbered_pattern.finditer(answer):
        _add(match.group(1))
    for segment in re.split(r'[.!?]\s+|\n{2,}', answer):
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
