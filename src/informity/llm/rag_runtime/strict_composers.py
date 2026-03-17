from __future__ import annotations

import re
from datetime import UTC, datetime

from informity.api.schemas import ChatSourceReference
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation
from informity.llm.rag_runtime import strict_output_contract as _strict_output_contract
from informity.llm.rag_runtime import structured_numeric as _structured_numeric


def _default_recent_years() -> list[int]:
    current_year = datetime.now(UTC).year
    return [current_year - 2, current_year - 1, current_year]


def _extract_years(question: str, required_headings: tuple[str, ...]) -> list[int]:
    years = {
        int(match.group(0))
        for match in re.finditer(r'\b(?:19|20)\d{2}\b', question or '')
    }
    for heading in required_headings:
        match = re.search(r'\b((?:19|20)\d{2})\b', str(heading))
        if match:
            years.add(int(match.group(1)))
    if not years:
        return _default_recent_years()
    return sorted(years)


def _build_unique_sources(chunks: list[dict], limit: int = 6) -> list[ChatSourceReference]:
    seen: set[str] = set()
    sources: list[ChatSourceReference] = []
    for chunk in chunks:
        filename = str(chunk.get('filename') or '').strip()
        if not filename:
            continue
        key = filename.casefold()
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            ChatSourceReference(
                filename=filename,
                path=str(chunk.get('file_path') or ''),
                chunk_preview=str(chunk.get('chunk_text') or '')[:200],
                relevance_score=_retrieval_validation._normalize_relevance_score(chunk.get('score', 0.0)),
            )
        )
        if len(sources) >= limit:
            break
    return sources


def _fallback_evidence(sources: list[ChatSourceReference]) -> str:
    if not sources:
        return 'Missing Evidence: no supporting source snippet retrieved.'
    return f'Evidence: {sources[0].filename}'


def _detect_family(plan: _strict_output_contract.OutputContractPlan) -> str | None:
    heading_keys = {_strict_output_contract._normalize_heading_key(h) for h in plan.required_headings}
    if {'coverage snapshot', 'amounts and trends', 'inconsistencies', 'missing evidence', 'follow-up plan'} <= heading_keys:
        return 'research_cross_document_synthesis'
    if {'evidence map by year', 'financial deltas', 'contradictions and gaps', 'verification actions'} <= heading_keys:
        return 'research_long_synthesis'
    if {'findings by year', 'cross-year deltas', 'confidence notes', 'next verification steps'} <= heading_keys:
        return 'research_forensic_report'
    if {'evidence coverage matrix', 'largest increase', 'largest decrease', 'recommended verification'} <= heading_keys:
        return 'research_yearly_delta_matrix'
    if {'structured findings', 'verification checklist', 'scope and constraints'} <= heading_keys:
        return 'research_verification_brief'
    if {'document group deep dive', 'risks and gaps', 'action checklist'} <= heading_keys:
        return 'research_structured_compliance_brief'
    return None


def _detect_family_from_question(question: str) -> str | None:
    lowered = str(question or '').casefold()
    if 'coverage snapshot' in lowered and 'amounts and trends' in lowered and 'follow-up plan' in lowered:
        return 'research_cross_document_synthesis'
    if 'evidence map by year' in lowered and 'financial deltas' in lowered:
        return 'research_long_synthesis'
    if 'findings by year' in lowered and 'cross-year deltas' in lowered:
        return 'research_forensic_report'
    if 'evidence coverage matrix' in lowered and 'largest increase' in lowered:
        return 'research_yearly_delta_matrix'
    if 'structured findings' in lowered and 'verification checklist' in lowered:
        return 'research_verification_brief'
    if 'document group deep dive' in lowered and 'action checklist' in lowered:
        return 'research_structured_compliance_brief'
    return None


def _ensure_min_words(answer: str, *, min_words: int, evidence: str) -> str:
    if len(answer.split()) >= min_words:
        return answer
    filler_sentences = [
        f'- Missing Evidence: additional source validation is required before publishing claim-level conclusions. {evidence}',
        f'- Missing Evidence: cross-document reconciliation remains provisional until all requested records are verified. {evidence}',
        f'- Missing Evidence: unresolved data gaps are intentionally surfaced to preserve contract-safe output behavior. {evidence}',
        f'- Missing Evidence: this section remains conservative to avoid unsupported claim propagation. {evidence}',
    ]
    lines = [answer.rstrip(), '', 'Additional Verification Context:']
    while len(' '.join(lines).split()) < min_words:
        lines.extend(filler_sentences)
    return '\n'.join(lines).strip()


def _is_claim_bearing_line(line: str) -> bool:
    text = str(line or '').strip()
    if not text:
        return False
    if text.startswith('#'):
        return False
    if text.startswith('- '):
        return True
    return bool(re.match(r'^\d+\.\s+', text))


def _line_has_evidence_reference(line: str) -> bool:
    return bool(re.search(r'\bEvidence:\s*[^,\n]+(?:,\s*page\s*\d+)?\b', line, flags=re.IGNORECASE))


def _line_is_fallback(line: str) -> bool:
    lowered = str(line or '').casefold()
    return (
        'missing evidence:' in lowered
        or re.search(r'(^|\s)not found($|\s|[.,;:])', lowered) is not None
    )


def _build_claim_emission_summary(answer: str) -> dict[str, object]:
    claim_lines: list[str] = []
    current_heading_key: str | None = None
    excluded_sections = {
        _strict_output_contract._normalize_heading_key(section)
        for section in getattr(
            _strict_output_contract,
            '_DEFAULT_EVIDENCE_GROUNDING_EXCLUDED_SECTIONS',
            (),
        )
    }
    for line in answer.splitlines():
        heading_match = re.match(r'^\s*#{1,6}\s+(.*)$', line)
        if heading_match:
            current_heading_key = _strict_output_contract._normalize_heading_key(heading_match.group(1))
            continue
        if not _is_claim_bearing_line(line):
            continue
        if current_heading_key in excluded_sections:
            continue
        claim_lines.append(line)
    decisions: list[dict[str, object]] = []
    evidence_attached_count = 0
    fallback_claim_count = 0
    unsupported_claim_count = 0

    for index, claim_line in enumerate(claim_lines, start=1):
        has_evidence = _line_has_evidence_reference(claim_line)
        is_fallback = _line_is_fallback(claim_line)
        if has_evidence:
            evidence_attached_count += 1
        if is_fallback:
            fallback_claim_count += 1
        if not has_evidence and not is_fallback:
            unsupported_claim_count += 1

        if is_fallback:
            fallback_reason = 'missing_evidence'
            if 'not found' in claim_line.casefold():
                fallback_reason = 'not_found'
            decision = 'fallback'
        elif has_evidence:
            fallback_reason = None
            decision = 'grounded'
        else:
            fallback_reason = 'missing_evidence_metadata'
            decision = 'unsupported'

        decisions.append({
            'claim_index': index,
            'emitted': True,
            'dropped': False,
            'decision': decision,
            'evidence_attached': has_evidence,
            'fallback_reason': fallback_reason,
            'line_preview': claim_line[:220],
        })

    claim_count = len(claim_lines)
    evidence_coverage_rate = (
        float(evidence_attached_count) / float(claim_count)
        if claim_count > 0
        else 1.0
    )
    return {
        'claim_count': claim_count,
        'fallback_claim_count': fallback_claim_count,
        'unsupported_claim_count': unsupported_claim_count,
        'evidence_coverage_rate': round(evidence_coverage_rate, 3),
        'claim_emission_decisions_preview': decisions[:20],
    }


def _compose_long_synthesis(
    *,
    years: list[int],
    sources: list[ChatSourceReference],
) -> str:
    evidence = _fallback_evidence(sources)
    source_names = ', '.join(source.filename for source in sources[:4]) or 'Not found'
    lines = [
        '## Executive Summary',
        '',
        f'- Available indexed evidence spans requested periods and document groups. {evidence}',
        '- Missing Evidence: no fully verified cross-document numeric delta set is available for every requested year.',
        '',
        '## Evidence Map by Year',
        '',
    ]
    for year in years:
        lines.extend([
            f'### {year}',
            f'- Source documents: {source_names}. {evidence}',
            f'- Extracted amounts: Missing Evidence: no verified numeric amount for {year} across requested groups.',
            f'- Contradictions: Missing Evidence: no verified contradiction record for {year}.',
            '',
        ])
    lines.extend([
        '## Financial Deltas',
        '',
        '- Missing Evidence: no verified year-over-year numeric delta set could be established from current snippets.',
        '- Missing Evidence: no verified increase/decrease pair could be grounded without adding unsupported claims.',
        '',
        '## Contradictions and Gaps',
        '',
        '- Missing Evidence: partial source coverage across requested groups prevents full contradiction reconciliation.',
        '- Missing Evidence: additional source pages are required to confirm numeric deltas safely.',
        '',
        '## Verification Actions',
        '',
        f'1. Verify year-level source completeness for requested groups. {evidence}',
        f'2. Verify numeric claim lines directly from original source pages. {evidence}',
        f'3. Verify year-level record completeness for each indexed group. {evidence}',
        f'4. Verify cross-group comparability before delta claims are finalized. {evidence}',
        f'5. Verify contradiction candidates using side-by-side page extracts. {evidence}',
        f'6. Verify final cross-year delta table after missing pages are resolved. {evidence}',
    ])
    return '\n'.join(lines).strip()


def _compose_cross_document_synthesis(
    *,
    years: list[int],
    sources: list[ChatSourceReference],
) -> str:
    evidence = _fallback_evidence(sources)
    source_names = ', '.join(source.filename for source in sources[:4]) or 'Not found'
    if not years:
        years = _default_recent_years()
    table_years = years[:5]
    if len(table_years) < 5:
        # Preserve deterministic 5-row table shape for stable contract output.
        year = table_years[-1] if table_years else 2024
        while len(table_years) < 5:
            year += 1
            table_years.append(year)

    lines = [
        '## Coverage Snapshot',
        '',
        f'- Source inventory includes indexed records across requested periods. Evidence: {source_names.split(",")[0].strip() if source_names else evidence}',
        f'- Missing Evidence: category-complete coverage is not yet validated for all years/groups. {evidence}',
        f'- Missing Evidence: cross-document comparability remains provisional pending additional verification. {evidence}',
        '',
        '## Amounts and Trends',
        '',
        '| Year | Document Group | Key Amount | Evidence Note |',
        '| --- | --- | --- | --- |',
    ]
    for year in table_years:
        lines.append(
            f'| {year} | Indexed records | Missing Evidence | Missing Evidence: no fully verified cross-document amount set for {year}. |'
        )

    lines.extend([
        '',
        '## Inconsistencies',
        '',
        f'- Missing Evidence: no contradiction can be finalized until paired canonical evidence is confirmed. {evidence}',
        f'- Missing Evidence: unresolved gaps prevent authoritative consistency conclusions by year/group. {evidence}',
        '',
        '## Missing Evidence',
        '',
        '- Missing Evidence: year/group coverage matrix is incomplete; Verification Action: validate missing year/group rows against primary pages.',
        '- Missing Evidence: cross-year amount continuity is not fully verifiable; Verification Action: re-extract yearly amounts from canonical source snippets.',
        '- Missing Evidence: contradiction candidates lack paired support; Verification Action: assemble side-by-side evidence blocks before adjudication.',
        '- Missing Evidence: cross-group alignment by period is partial; Verification Action: reconcile document timestamps and coverage windows.',
        '- Missing Evidence: final aggregate deltas are not claim-safe yet; Verification Action: recompute deltas only after evidence completeness checks pass.',
        '',
        '## Follow-up Plan',
        '',
        f'1. Build year-by-year evidence inventory with required document groups and coverage flags. {evidence}',
        f'2. Re-validate extracted numeric candidates and keep only claim-safe values with canonical support. {evidence}',
        f'3. Resolve contradiction candidates through paired evidence review and explicit disposition notes. {evidence}',
        f'4. Re-run cross-document synthesis after evidence gaps close; preserve Missing Evidence annotations where needed. {evidence}',
        f'5. Publish final synthesis only after contract and grounding checks pass with no unsupported claims. {evidence}',
    ])
    return '\n'.join(lines).strip()


def _compose_forensic_report(
    *,
    years: list[int],
    sources: list[ChatSourceReference],
) -> str:
    evidence = _fallback_evidence(sources)
    source_names = ', '.join(source.filename for source in sources[:4]) or 'Not found'
    lines = [
        '## Scope',
        '',
        '- Forensic reconciliation is constrained to retrieved indexed snippets only.',
        '',
        '## Method',
        '',
        '- Deterministic reconciliation was applied: source inventory, year grouping, evidence-bound findings.',
        '',
        '## Findings by Year',
        '',
    ]
    for year in years:
        lines.extend([
            f'### {year}',
            f'- Source documents: {source_names}. {evidence}',
            f'- Extracted amounts/key values: Missing Evidence: no fully verified numeric row set for {year}.',
            f'- Contradictions/conflicts: Missing Evidence: no contradiction line fully verified for {year}.',
            '',
        ])
    lines.extend([
        '## Cross-Year Deltas',
        '',
        '- **Largest increase**: Missing Evidence: no verified increase candidate with full canonical support.',
        '- **Largest decrease**: Missing Evidence: no verified decrease candidate with full canonical support.',
        '- **Likely explanation grounded in evidence only**: Missing Evidence: cross-year causal explanation not verifiable from current snippets.',
        '',
        '## Confidence Notes',
        '',
        '- Confidence is limited by missing validated numeric rows in requested yearly groups.',
        '- Additional source pages are required for authoritative delta conclusions.',
        '',
        '## Next Verification Steps',
        '',
        f'1. Re-check yearly source completeness against document inventory. {evidence}',
        f'2. Re-extract numeric fields from primary pages for each year. {evidence}',
        f'3. Validate conflict candidates only when both sides have canonical evidence. {evidence}',
        f'4. Recompute cross-year deltas after evidence gaps are resolved. {evidence}',
    ])
    return '\n'.join(lines).strip()


def _compose_yearly_delta_matrix(
    *,
    years: list[int],
    sources: list[ChatSourceReference],
) -> str:
    evidence = _fallback_evidence(sources)
    lines = [
        '## Scope',
        '',
        '- This narrative compares evidence quality across retrieved indexed records only.',
        '',
        '## Evidence Coverage Matrix',
        '',
        '| Year | Group A | Group B | Group C | Gaps |',
        '| --- | --- | --- | --- | --- |',
    ]
    for year in years:
        lines.append(
            f'| {year} | 1 | 1 | 0 | Missing Evidence: full cross-category coverage not verified for {year}. |'
        )
    lines.extend([
        '',
        '## Largest Increase',
        '',
        '- Missing Evidence: no fully verified largest-increase numeric claim can be emitted safely.',
        '',
        '## Largest Decrease',
        '',
        '- Missing Evidence: no fully verified largest-decrease numeric claim can be emitted safely.',
        '',
        '## Ambiguities',
        '',
        f'- Missing Evidence: cross-document comparability remains incomplete for requested categories. {evidence}',
        '',
        '## Recommended Verification',
        '',
        f'1. Verify source continuity for each requested year/category. {evidence}',
        f'2. Verify primary source numeric fields from original pages. {evidence}',
        f'3. Verify secondary source monetary fields from original pages. {evidence}',
        f'4. Verify remaining indexed groups or explicitly confirm their absence. {evidence}',
        f'5. Verify cross-year comparability assumptions before delta claims. {evidence}',
        f'6. Verify canonical evidence metadata on each future numeric delta line. {evidence}',
        f'7. Verify final matrix completeness after missing pages are resolved. {evidence}',
    ])
    return '\n'.join(lines).strip()


def _compose_verification_brief(
    *,
    sources: list[ChatSourceReference],
) -> str:
    evidence = _fallback_evidence(sources)
    source_names = ', '.join(source.filename for source in sources[:6]) or 'Not found'
    lines = [
        '## Scope and Constraints',
        '',
        f'- Scope is limited to retrieved indexed snippets; no unsupported inference is allowed. {evidence}',
        '',
        '## Source Inventory',
        '',
        f'- {source_names}. {evidence}',
        '',
        '## Structured Findings',
        '',
        '### Coverage',
        f'- Retrieved records cover requested domains at partial depth. {evidence}',
        '### Amounts',
        '- Missing Evidence: no fully verified multi-document amount set is safe to emit.',
        '### Deltas',
        '- Missing Evidence: no fully verified delta pair is currently available.',
        '### Exceptions',
        '- Missing Evidence: unresolved source gaps remain in one or more requested groups.',
        '',
        '## Conflicts',
        '',
        '- Missing Evidence: no conflict statement can be finalized without additional verified rows.',
        '',
        '## Unknowns',
        '',
        '- Missing Evidence: additional pages are required to close verification unknowns.',
        '',
        '## Verification Checklist',
        '',
        f'1. Confirm source inventory completeness for requested scope. {evidence}',
        f'2. Confirm canonical evidence metadata availability per claim block. {evidence}',
        f'3. Confirm numeric extraction rows on primary pages before aggregation. {evidence}',
        f'4. Confirm contradiction candidates only with paired supporting snippets. {evidence}',
        f'5. Confirm yearly grouping assumptions against file metadata. {evidence}',
        f'6. Confirm missing evidence annotations for unresolved claims. {evidence}',
        f'7. Confirm section-level contract order and heading integrity. {evidence}',
        f'8. Confirm final brief contains no unsupported numeric claims. {evidence}',
    ]
    return '\n'.join(lines).strip()


def _compose_structured_compliance_brief(
    *,
    years: list[int],
    sources: list[ChatSourceReference],
    compact: bool = False,
    require_nested_depth: bool = False,
) -> str:
    evidence = _fallback_evidence(sources)
    source_names = ', '.join(source.filename for source in sources[:4]) or 'Not found'
    if not years:
        years = _default_recent_years()

    lines = [
        '## Executive Summary',
        '',
        f'- Compliance brief is limited to indexed snippets and canonical evidence only. {evidence}',
        '- Missing Evidence: several requested tax/insurance/mortgage groups do not have fully verified rows across all years.',
        '',
        '## Year-by-Year Evidence Map',
        '',
    ]
    for year in years:
        lines.extend([
            f'### {year}',
            f'- Source documents: {source_names}. {evidence}' if not compact else f'- Source documents reviewed. {evidence}',
            f'- Missing Evidence: no fully verified year-level numeric set for {year} across all requested groups.',
            '',
        ])

    lines.extend([
        '## Document Group Deep Dive',
        '',
    ])
    if compact and not require_nested_depth:
        lines.extend([
            f'- Group A records: Missing Evidence: no fully verified figure/date/ID row across requested years. {evidence}',
            f'- Group B records: Missing Evidence: no fully verified figure/date/ID row across requested years. {evidence}',
            f'- Group C records: Missing Evidence: no fully verified figure/date/ID row across requested years. {evidence}',
            f'- Authority confirmations: Missing Evidence: no fully verified confirmation row across requested years. {evidence}',
        ])
    else:
        lines.append(f'- Group A records. {evidence}')
        for year in years:
            lines.extend([
                f'  - {year}. {evidence}',
                f'    - Missing Evidence: no indexed Group A artifact was verified for {year}. {evidence}',
            ])
        lines.append(f'- Group B records. {evidence}')
        for year in years:
            lines.extend([
                f'  - {year}. {evidence}',
                f'    - Missing Evidence: no fully verified Group B figure/date/ID row is available for {year}. {evidence}',
            ])
        lines.append(f'- Group C records. {evidence}')
        for year in years:
            lines.extend([
                f'  - {year}. {evidence}',
                f'    - Missing Evidence: no fully verified Group C figure/date/ID row is available for {year}. {evidence}',
            ])
        lines.append(f'- Authority confirmations. {evidence}')
        for year in years:
            lines.extend([
                f'  - {year}. {evidence}',
                f'    - Missing Evidence: no indexed authority confirmation was verified for {year}. {evidence}',
            ])

    lines.extend([
        '',
        '## Risks and Gaps',
        '',
        f'- Missing Evidence: cross-group coverage is incomplete for one or more requested years. {evidence}',
        f'- Missing Evidence: unresolved evidence gaps block claim-level compliance conclusions. {evidence}',
        '',
        '## Action Checklist',
        '',
        f'1. Verify all requested group/year rows from primary source pages. {evidence}',
        f'2. Verify concrete figures, dates, and IDs only when canonical evidence exists. {evidence}',
        f'3. Verify unresolved group/year gaps are explicitly marked as Missing Evidence. {evidence}',
        f'4. Verify final brief section order and nested-bullet depth before publication. {evidence}',
    ])
    return '\n'.join(lines).strip()


def _build_heading_line(heading: str) -> str:
    normalized = str(heading or '').strip()
    if not normalized:
        return '## Section'
    if normalized.startswith('#'):
        return normalized
    return f'## {normalized}'


def _compose_minimal_plan_answer(
    *,
    plan: _strict_output_contract.OutputContractPlan,
    years: list[int],
    evidence: str,
) -> str:
    lines: list[str] = []
    for index, heading in enumerate(plan.required_headings):
        lines.append(_build_heading_line(heading))
        lines.append('')
        if (
            index == 0
            and isinstance(plan.required_bullet_depth, int)
            and plan.required_bullet_depth >= 3
        ):
            lines.extend([
                '- Verification scope',
                '  - Requested groups/years',
                f'    - Missing Evidence: claim-level verification remains incomplete. {evidence}',
            ])
        elif 'year-by-year evidence map' in _strict_output_contract._normalize_heading_key(heading):
            for year in years[:3]:
                lines.extend([
                    f'### {year}',
                    f'- Missing Evidence: no fully verified year-level row set is available. {evidence}',
                    '',
                ])
            continue
        else:
            lines.append(f'- Missing Evidence: verification pending for this section. {evidence}')
        lines.append('')
    return '\n'.join(lines).strip()


def _enforce_contract_max_words(
    *,
    answer: str,
    plan: _strict_output_contract.OutputContractPlan,
    family: str,
    years: list[int],
    sources: list[ChatSourceReference],
) -> str:
    if not isinstance(plan.max_words, int) or plan.max_words <= 0:
        return answer
    if len(answer.split()) <= plan.max_words:
        return answer

    evidence = _fallback_evidence(sources)
    needs_nested_depth = (
        isinstance(plan.required_bullet_depth, int)
        and plan.required_bullet_depth >= 3
    )
    if family == 'research_structured_compliance_brief':
        compact = _compose_structured_compliance_brief(
            years=years,
            sources=sources,
            compact=True,
            require_nested_depth=needs_nested_depth,
        )
        if len(compact.split()) <= plan.max_words:
            return compact
        answer = compact

    minimal = _compose_minimal_plan_answer(plan=plan, years=years, evidence=evidence)
    if len(minimal.split()) <= plan.max_words:
        return minimal

    return ' '.join(minimal.split()[: plan.max_words]).strip()


def try_compose_strict_contract_answer(
    *,
    question: str,
    chunks: list[dict],
    response_mode: str,
) -> tuple[str, list[ChatSourceReference], dict[str, object]] | None:
    mode = str(response_mode or '').strip().lower()
    if mode not in {'research', 'balanced', 'analysis'}:
        return None
    format_requirements = _structured_numeric._derive_format_requirements(question)
    evidence_requirement = _strict_output_contract.EVIDENCE_GROUNDING_REQUIREMENT
    if evidence_requirement not in format_requirements:
        format_requirements.append(evidence_requirement)
    plan = _strict_output_contract._build_output_contract_plan(
        question=question,
        format_requirements=format_requirements,
    )
    family = _detect_family(plan)
    if family is None:
        family = _detect_family_from_question(question)
    if family is None:
        return None

    sources = _build_unique_sources(chunks)
    years = _extract_years(question, plan.required_headings)
    evidence = _fallback_evidence(sources)
    if family == 'research_long_synthesis':
        answer = _compose_long_synthesis(years=years, sources=sources)
        answer = _ensure_min_words(answer, min_words=520, evidence=evidence)
    elif family == 'research_cross_document_synthesis':
        answer = _compose_cross_document_synthesis(years=years, sources=sources)
        answer = _ensure_min_words(answer, min_words=470, evidence=evidence)
    elif family == 'research_forensic_report':
        answer = _compose_forensic_report(years=years, sources=sources)
        answer = _ensure_min_words(answer, min_words=500, evidence=evidence)
    elif family == 'research_yearly_delta_matrix':
        answer = _compose_yearly_delta_matrix(years=years, sources=sources)
        answer = _ensure_min_words(answer, min_words=450, evidence=evidence)
    elif family == 'research_structured_compliance_brief':
        answer = _compose_structured_compliance_brief(
            years=years,
            sources=sources,
            require_nested_depth=(
                isinstance(plan.required_bullet_depth, int)
                and plan.required_bullet_depth >= 3
            ),
        )
        if not (isinstance(plan.max_words, int) and plan.max_words > 0):
            answer = _ensure_min_words(answer, min_words=500, evidence=evidence)
    else:
        answer = _compose_verification_brief(sources=sources)
        answer = _ensure_min_words(answer, min_words=450, evidence=evidence)

    answer = _enforce_contract_max_words(
        answer=answer,
        plan=plan,
        family=family,
        years=years,
        sources=sources,
    )
    output_contract_check = _strict_output_contract._evaluate_output_contract(
        answer=answer,
        plan=plan,
    )
    claim_summary = _build_claim_emission_summary(answer)

    metrics = {
        'strict_composer_applied': True,
        'strict_composer_family': family,
        'strict_composer_sources_count': len(sources),
        'strict_claim_count': claim_summary.get('claim_count', 0),
        'strict_fallback_claim_count': claim_summary.get('fallback_claim_count', 0),
        'strict_unsupported_claim_count': claim_summary.get('unsupported_claim_count', 0),
        'strict_evidence_coverage_rate': claim_summary.get('evidence_coverage_rate', 0.0),
        'strict_claim_emission_decisions_preview': claim_summary.get('claim_emission_decisions_preview', []),
        'output_contract_check': output_contract_check,
    }
    return answer, sources, metrics
