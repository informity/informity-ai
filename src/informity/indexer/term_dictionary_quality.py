# ==============================================================================
# Informity AI — Term Dictionary Quality Evaluation
# Deterministic quality metrics and gate checks for dictionary rebuilds.
# ==============================================================================

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(slots=True)
class TermDictionaryQualityMetrics:
    total_candidates: int
    kept_candidates: int
    rejected_candidates: int
    noise_rate: float
    keep_rate: float
    candidate_type_counts: dict[str, int]
    kept_type_counts: dict[str, int]


@dataclass(slots=True)
class TermDictionaryQualityGateResult:
    passed: bool
    reason: str
    metrics: TermDictionaryQualityMetrics


def evaluate_term_dictionary_quality(
    *,
    total_candidates: int,
    kept_candidates: int,
    noise_rate_threshold: float,
    min_candidates_for_gate: int,
    gate_enabled: bool,
    candidate_term_types: list[str] | None = None,
    kept_term_types: list[str] | None = None,
) -> TermDictionaryQualityGateResult:
    total = max(0, int(total_candidates))
    kept = max(0, int(kept_candidates))
    rejected = max(0, total - kept)
    noise_rate = (rejected / total) if total > 0 else 0.0
    keep_rate = (kept / total) if total > 0 else 0.0
    metrics = TermDictionaryQualityMetrics(
        total_candidates=total,
        kept_candidates=kept,
        rejected_candidates=rejected,
        noise_rate=noise_rate,
        keep_rate=keep_rate,
        candidate_type_counts=dict(Counter(candidate_term_types or [])),
        kept_type_counts=dict(Counter(kept_term_types or [])),
    )

    if not gate_enabled:
        return TermDictionaryQualityGateResult(passed=True, reason='gate_disabled', metrics=metrics)

    if total < max(1, int(min_candidates_for_gate)):
        return TermDictionaryQualityGateResult(
            passed=True,
            reason='below_min_candidates_for_gate',
            metrics=metrics,
        )

    threshold = max(0.0, min(1.0, float(noise_rate_threshold)))
    if noise_rate > threshold:
        return TermDictionaryQualityGateResult(
            passed=False,
            reason=f'noise_rate_exceeded:{noise_rate:.3f}>{threshold:.3f}',
            metrics=metrics,
        )

    return TermDictionaryQualityGateResult(passed=True, reason='ok', metrics=metrics)
