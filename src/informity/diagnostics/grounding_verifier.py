from __future__ import annotations

import re

_EVIDENCE_GROUNDED_QUERY_PATTERN = re.compile(r'\bevidence[-\s]*grounded\b', re.IGNORECASE)
_CURRENCY_TOKEN_PATTERN = re.compile(r'\$?\d[\d,]*(?:\.\d{1,2})?')
_PHRASE_VERIFICATION_PATTERNS = (
    'foreign account',
    'foreign accounts',
    'fatca',
)
_YEAR_TOKEN_MIN = 1000
_YEAR_TOKEN_MAX = 2999
_GROUNDING_VERIFIER_HEURISTIC_PROFILE = 'grounding_verifier_v1'
_EVIDENCE_MARKER = 'evidence:'
_NOT_FOUND_MARKER = 'not found'
_TABLE_HEADER_TOKENS = {'field', 'value', 'source snippet', 'line item', 'amount', 'box'}


def _extract_top_level_bullet_blocks(answer: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in answer.splitlines():
        if re.match(r'^[-*]\s+\S+', line):
            if current:
                blocks.append(current)
            current = [line]
            continue
        if current:
            current.append(line)
    if current:
        blocks.append(current)
    return ['\n'.join(lines).strip() for lines in blocks if lines]


def _normalize_numeric_token(raw_value: str) -> str:
    return re.sub(r'[^0-9.\-]', '', raw_value or '')


def _is_year_token(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and _YEAR_TOKEN_MIN <= int(token) <= _YEAR_TOKEN_MAX


def _is_actionable_numeric_token(raw_token: str, normalized_token: str) -> bool:
    if not normalized_token:
        return False
    digits_only = re.sub(r'[^0-9]', '', normalized_token)
    if len(digits_only) <= 1:
        return False
    if _is_year_token(normalized_token):
        return False
    has_currency_shape = (
        '$' in raw_token
        or ',' in raw_token
        or '.' in raw_token
        or raw_token.strip().startswith('(')
        or raw_token.strip().endswith(')')
    )
    if has_currency_shape:
        return True
    # Plain small integers are usually labels/page markers/IDs in OCR output.
    return not (normalized_token.isdigit() and len(normalized_token) < 5)


def _extract_numeric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _CURRENCY_TOKEN_PATTERN.finditer(text):
        raw_token = match.group(0)
        normalized = _normalize_numeric_token(raw_token)
        if not _is_actionable_numeric_token(raw_token, normalized):
            continue
        tokens.add(normalized)
    return tokens


def _is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith('|') and stripped.endswith('|')):
        return False
    body = stripped.strip('|').replace(' ', '')
    return bool(body) and all(char in '-:|' for char in body)


def _extract_claim_text(answer: str) -> str:
    claim_lines: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_markdown_table_separator(line):
            continue
        if line.startswith('|') and line.endswith('|'):
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if len(cells) >= 3:
                first = cells[0].casefold()
                second = cells[1].casefold()
                if first in _TABLE_HEADER_TOKENS and second in _TABLE_HEADER_TOKENS:
                    continue
                claim_lines.append(' | '.join(cells[:2]))
                continue
        bullet_match = re.match(r'^[-*]\s+(.*)$', line)
        if bullet_match:
            body = bullet_match.group(1).strip()
            segments = [segment.strip() for segment in body.split('|')]
            if len(segments) >= 3:
                claim_lines.append(' | '.join(segments[:2]))
                continue
        claim_lines.append(line)
    return '\n'.join(claim_lines)


def run_grounding_verifier(
    *,
    question: str,
    answer: str,
    sources: list[dict[str, object]],
) -> dict[str, object]:
    # Diagnostics-only heuristic verification: this annotates quality signals
    # and does not mutate answer text or runtime routing decisions.
    strict_required = bool(_EVIDENCE_GROUNDED_QUERY_PATTERN.search(question or ''))
    source_text = ' '.join(str(source.get('chunk_preview', '') or '') for source in sources).casefold()
    claim_text = _extract_claim_text(answer)
    source_numeric_tokens = _extract_numeric_tokens(source_text)
    answer_numeric_tokens = _extract_numeric_tokens(claim_text)
    unsupported_numbers = sorted(token for token in answer_numeric_tokens if token not in source_numeric_tokens)

    unsupported_phrases = sorted(
        phrase
        for phrase in _PHRASE_VERIFICATION_PATTERNS
        if phrase in claim_text.casefold() and phrase not in source_text
    )
    unsupported_claims = [*unsupported_numbers, *unsupported_phrases]

    bullet_blocks = _extract_top_level_bullet_blocks(answer)
    evidence_bullet_hits = sum(1 for block in bullet_blocks if _EVIDENCE_MARKER in block.casefold())
    evidence_coverage_rate = (
        round(evidence_bullet_hits / len(bullet_blocks), 3) if bullet_blocks else 0.0
    )
    not_found_count = answer.casefold().count(_NOT_FOUND_MARKER)
    verifier_passed = len(unsupported_claims) == 0
    return {
        'heuristic_profile': _GROUNDING_VERIFIER_HEURISTIC_PROFILE,
        'required': strict_required,
        'passed': verifier_passed,
        'unsupported_claims': unsupported_claims,
        'unsupported_claim_count': len(unsupported_claims),
        'evidence_coverage_rate': evidence_coverage_rate,
        'not_found_count': not_found_count,
    }
