# ==============================================================================
# Informity AI — Contract Gate
# Single contract validation utilities for closeout-time enforcement.
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass

from informity.llm import contract_prompt_parser as _contract_prompt_parser
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import QuerySubtype, QueryType
from informity.llm.user_messages import MISSING_EVIDENCE_CALLOUT_MESSAGE

_MARKDOWN_HEADING_PATTERN = re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$', re.MULTILINE)
_YEAR_TOKEN_PATTERN = re.compile(r'\b[12]\d{3}\b')
_MISSING_EVIDENCE_REQUEST_PATTERN = re.compile(r'\bmissing\s+evidence\b|\bmissing\s+records?\b|\bgaps?\b', re.IGNORECASE)
_MISSING_EVIDENCE_LINE_PATTERN = re.compile(r'(?im)^\s*(?:[-*]\s*)?missing\s+evidence\s*:')
_SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_REDACTED_SSN_TOKEN = '[REDACTED-SSN]'
_SECTION_INSUFFICIENT_EVIDENCE_LINE = '- Insufficient evidence in retrieved context to complete this section.'


@dataclass(frozen=True)
class ContractSpec:
    required_headings: list[str]
    min_year_count: int
    enforce_heading_order: bool = False
    requires_missing_evidence_callout: bool = False
    enforce_forbidden_redaction: bool = True


@dataclass(frozen=True)
class ContractValidationResult:
    missing_required_headings: list[str]
    required_year_count: int
    observed_year_count: int

    @property
    def has_gap(self) -> bool:
        if self.missing_required_headings:
            return True
        return self.required_year_count > 0 and self.observed_year_count < self.required_year_count


def build_contract_spec(*, question: str, classification: QueryClassification | None) -> ContractSpec:
    required_headings = _contract_prompt_parser.extract_required_headings(question)
    min_year_count = 0
    if (
        classification is not None
        and str(classification.intent).strip().lower() == QueryType.COVERAGE
        and str(classification.subtype or '').strip().lower() == QuerySubtype.AGGREGATE_BY_PERIOD
        and _contract_prompt_parser.has_year_subsection_cue(question)
    ):
        min_year_count = 2
    enforce_heading_order = bool(required_headings) and _contract_prompt_parser.has_ordered_heading_cue(question)
    requires_missing_evidence_callout = bool(_MISSING_EVIDENCE_REQUEST_PATTERN.search(str(question or '')))
    return ContractSpec(
        required_headings=required_headings,
        min_year_count=min_year_count,
        enforce_heading_order=enforce_heading_order,
        requires_missing_evidence_callout=requires_missing_evidence_callout,
        enforce_forbidden_redaction=True,
    )


def validate_contract(*, answer: str, spec: ContractSpec) -> ContractValidationResult:
    normalized_answer = str(answer or '')
    missing_required_headings = _find_missing_required_headings(normalized_answer, spec.required_headings)
    observed_year_count = _count_distinct_years(normalized_answer)
    return ContractValidationResult(
        missing_required_headings=missing_required_headings,
        required_year_count=max(0, int(spec.min_year_count)),
        observed_year_count=observed_year_count,
    )


def build_repair_guidance(result: ContractValidationResult) -> str | None:
    lines: list[str] = []
    if result.missing_required_headings:
        headings_list = '; '.join(f'## {heading}' for heading in result.missing_required_headings)
        lines.append(f'Complete only the missing required sections using these exact headings: {headings_list}.')
        lines.append('Do not repeat sections that are already complete.')
    if result.required_year_count > 0 and result.observed_year_count < result.required_year_count:
        lines.append(f'Include at least {result.required_year_count} distinct years using only retrieved evidence.')
    guidance = ' '.join(lines).strip()
    return guidance or None


def enforce_required_sections(answer: str, spec: ContractSpec) -> tuple[str, list[str]]:
    # Closeout-time fallback: append missing required headings in contract order.
    # This is model-agnostic and only fills structural gaps when generation ends
    # before all explicitly requested sections are present.
    normalized_answer = str(answer or '').strip()
    if spec.enforce_forbidden_redaction:
        normalized_answer = _SSN_PATTERN.sub(_REDACTED_SSN_TOKEN, normalized_answer)

    if spec.enforce_heading_order and spec.required_headings:
        normalized_answer = _normalize_required_heading_order(
            answer=normalized_answer,
            required_headings=spec.required_headings,
        )

    result = validate_contract(answer=normalized_answer, spec=spec)
    if result.missing_required_headings:
        appended_blocks: list[str] = []
        for heading in result.missing_required_headings:
            appended_blocks.append(
                '\n'.join(
                    [
                        f'## {heading}',
                        _SECTION_INSUFFICIENT_EVIDENCE_LINE,
                    ]
                )
            )
        appended = '\n\n'.join(appended_blocks).strip()
        normalized_answer = f'{normalized_answer}\n\n{appended}'.strip() if normalized_answer else appended

    if spec.requires_missing_evidence_callout and _MISSING_EVIDENCE_LINE_PATTERN.search(normalized_answer) is None:
        callout = MISSING_EVIDENCE_CALLOUT_MESSAGE
        normalized_answer = f'{normalized_answer}\n\n{callout}'.strip() if normalized_answer else callout

    return normalized_answer, list(result.missing_required_headings)


def _normalize_heading_text(value: str) -> str:
    normalized = re.sub(r'^\s*#{1,6}\s*', '', str(value or '').strip())
    normalized = re.sub(r'\s+', ' ', normalized).strip().strip(' .:')
    return normalized.casefold()


def _extract_headings_from_answer(answer: str) -> set[str]:
    headings: set[str] = set()
    for raw in _MARKDOWN_HEADING_PATTERN.findall(str(answer or '')):
        normalized = _normalize_heading_text(raw)
        if normalized:
            headings.add(normalized)
    return headings


def _find_missing_required_headings(answer: str, required_headings: list[str]) -> list[str]:
    if not required_headings:
        return []
    present = _extract_headings_from_answer(answer)
    missing: list[str] = []
    for heading in required_headings:
        if _normalize_heading_text(heading) not in present:
            missing.append(heading)
    return missing


def _count_distinct_years(answer: str) -> int:
    return len(set(_YEAR_TOKEN_PATTERN.findall(str(answer or ''))))


def _normalize_required_heading_order(*, answer: str, required_headings: list[str]) -> str:
    text = str(answer or '').strip()
    if not text or not required_headings:
        return text

    heading_matches = list(_MARKDOWN_HEADING_PATTERN.finditer(text))
    if not heading_matches:
        return text

    sections: list[tuple[str, str, str]] = []
    for idx, match in enumerate(heading_matches):
        heading_raw = str(match.group(1) or '').strip()
        normalized_heading = _normalize_heading_text(heading_raw)
        start = match.start()
        end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(text)
        block = text[start:end].strip()
        sections.append((normalized_heading, heading_raw, block))

    required_norm = [_normalize_heading_text(item) for item in required_headings]
    required_set = set(required_norm)
    section_map: dict[str, tuple[str, str, str]] = {}
    for section in sections:
        key = section[0]
        if key and key not in section_map:
            section_map[key] = section

    if not any(key in section_map for key in required_set):
        return text

    rebuilt_blocks: list[str] = []
    for heading in required_headings:
        norm = _normalize_heading_text(heading)
        existing = section_map.get(norm)
        if existing is not None:
            rebuilt_blocks.append(existing[2])
        else:
            rebuilt_blocks.append(
                '\n'.join(
                    [
                        f'## {heading}',
                        _SECTION_INSUFFICIENT_EVIDENCE_LINE,
                    ]
                )
            )

    extras = [block for norm, _, block in sections if norm not in required_set]
    if extras:
        rebuilt_blocks.extend(extras)

    return '\n\n'.join(part.strip() for part in rebuilt_blocks if str(part).strip()).strip()
