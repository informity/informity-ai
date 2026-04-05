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

_MARKDOWN_HEADING_PATTERN = re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$', re.MULTILINE)
_YEAR_TOKEN_PATTERN = re.compile(r'\b[12]\d{3}\b')


@dataclass(frozen=True)
class ContractSpec:
    required_headings: list[str]
    min_year_count: int


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
    return ContractSpec(required_headings=required_headings, min_year_count=min_year_count)


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
