# ==============================================================================
# Informity AI — Role Evidence Fallback
# Shared role-scoped evidence fallback for runtime and diagnostics parity.
# ==============================================================================

from __future__ import annotations

from informity.diagnostics.observer import estimate_evidence_metrics
from informity.llm.roles import get_role_profile

_ROLE_EVIDENCE_COVERAGE_MIN = 0.45
_ROLE_UNSUPPORTED_CLAIMS_MAX = 2


def should_apply_role_evidence_fallback(
    *,
    chat_mode: str,
    role_id: str | None,
    unsupported_claim_count: int,
    evidence_coverage_rate: float,
    sources_count: int,
) -> bool:
    if chat_mode != 'researcher':
        return False
    if not str(role_id or '').strip():
        return False
    if sources_count <= 0:
        return False
    return (
        unsupported_claim_count > _ROLE_UNSUPPORTED_CLAIMS_MAX
        or evidence_coverage_rate < _ROLE_EVIDENCE_COVERAGE_MIN
    )


def build_role_evidence_fallback_answer(
    *,
    original_answer: str,
    source_items: list[dict[str, object]],
    role_id: str,
) -> str:
    role = get_role_profile(role_id)
    source_lines: list[str] = []
    for source in source_items[:3]:
        filename = str(source.get('filename', 'unknown') or 'unknown')
        preview = str(source.get('chunk_preview', '') or '').strip()
        if len(preview) > 220:
            preview = preview[:219].rstrip() + '...'
        if preview:
            source_lines.append(f'- {filename}: {preview}')
        else:
            source_lines.append(f'- {filename}')

    uncertainty_lines = [
        '- Retrieved evidence does not support a fully definitive domain conclusion.',
        '- Additional relevant excerpts may be needed to confirm role-specific risks or obligations.',
    ]
    if original_answer:
        uncertainty_lines.append('- The prior draft may include interpretation beyond directly supported evidence.')

    sections = [
        '## Evidence from Retrieved Context',
        '\n'.join(source_lines) if source_lines else '- No substantive excerpts were available.',
        '## Uncertainty / Missing Evidence',
        '\n'.join(uncertainty_lines),
    ]
    if role.disclaimer:
        sections.append(f'Disclaimer: {role.disclaimer}')
    return '\n\n'.join(sections).strip()


def apply_role_evidence_fallback(
    *,
    answer: str,
    source_items: list[dict[str, object]],
    chat_mode: str,
    role_id: str | None,
) -> tuple[str, int, float, int, bool]:
    unsupported_claim_count, evidence_coverage_rate, not_found_count = estimate_evidence_metrics(
        answer=answer,
        source_texts=[str(source.get('chunk_preview', '') or '') for source in source_items],
    )
    if not should_apply_role_evidence_fallback(
        chat_mode=chat_mode,
        role_id=role_id,
        unsupported_claim_count=unsupported_claim_count,
        evidence_coverage_rate=evidence_coverage_rate,
        sources_count=len(source_items),
    ):
        return answer, unsupported_claim_count, evidence_coverage_rate, not_found_count, False

    fallback_answer = build_role_evidence_fallback_answer(
        original_answer=answer,
        source_items=source_items,
        role_id=str(role_id),
    )
    fallback_unsupported, fallback_coverage, fallback_not_found = estimate_evidence_metrics(
        answer=fallback_answer,
        source_texts=[str(source.get('chunk_preview', '') or '') for source in source_items],
    )
    return fallback_answer, fallback_unsupported, fallback_coverage, fallback_not_found, True

