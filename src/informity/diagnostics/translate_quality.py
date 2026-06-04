from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'has', 'are', 'was', 'were',
    'you', 'your', 'our', 'their', 'them', 'they', 'into', 'onto', 'about', 'after', 'before',
    'also', 'than', 'then', 'when', 'what', 'which', 'will', 'would', 'could', 'should',
    'there', 'here', 'such', 'each', 'more', 'most', 'less', 'many', 'much', 'some', 'only',
    'over', 'under', 'between', 'across', 'within', 'without', 'during', 'while',
}


@dataclass(frozen=True)
class TranslateQualityMetrics:
    completion_rate: float
    truncation_rate: float
    term_consistency: float | None
    structural_fidelity: float
    quality_score: int
    recommendation: str


def _normalize_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _adjacent_overlap_score(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    scores = []
    previous_tokens = _normalize_tokens(texts[0])
    for text in texts[1:]:
        current_tokens = _normalize_tokens(text)
        scores.append(_jaccard(previous_tokens, current_tokens))
        previous_tokens = current_tokens
    return sum(scores) / len(scores) if scores else None


def _markdown_issue_count(text: str) -> int:
    issues = 0
    if not text.strip():
        issues += 1
    if text.count('```') % 2 != 0:
        issues += 1
    if any('|' in line for line in text.splitlines()) and '---' not in text:
        issues += 1
    return issues


def compute_translate_quality(
    sections: Sequence[object],
    *,
    section_count: int | None,
    truncated_sections: int,
) -> TranslateQualityMetrics:
    total_sections = section_count if section_count and section_count > 0 else len(sections)
    total_sections = max(total_sections, 1)
    completed_sections = [section for section in sections if bool(getattr(section, 'completed', False))]
    completed_count = len(completed_sections)
    completion_rate = completed_count / total_sections
    truncation_rate = max(0.0, truncated_sections / total_sections)

    completed_texts = [
        str(getattr(section, 'text', '') or '')
        for section in completed_sections
        if str(getattr(section, 'text', '') or '').strip()
    ]
    term_consistency = _adjacent_overlap_score(completed_texts)

    issue_count = 0
    for section in completed_sections:
        issue_count += _markdown_issue_count(str(getattr(section, 'text', '') or ''))
    issue_count += max(0, total_sections - completed_count)
    structural_fidelity = max(0.0, 1.0 - (issue_count / total_sections))

    term_consistency_component = 0.5 if term_consistency is None else term_consistency
    quality_score = round(
        100
        * (
            0.45 * completion_rate
            + 0.25 * structural_fidelity
            + 0.20 * term_consistency_component
            + 0.10 * (1.0 - truncation_rate)
        )
    )

    if (
        completion_rate >= 0.95
        and structural_fidelity >= 0.90
        and truncation_rate <= 0.10
        and (term_consistency is None or term_consistency >= 0.15)
    ):
        recommendation = 'pass'
    elif quality_score >= 65:
        recommendation = 'review'
    else:
        recommendation = 'hold'

    return TranslateQualityMetrics(
        completion_rate=completion_rate,
        truncation_rate=truncation_rate,
        term_consistency=term_consistency,
        structural_fidelity=structural_fidelity,
        quality_score=quality_score,
        recommendation=recommendation,
    )
