from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt
from number_parser import parse_number

_MISSING_EVIDENCE_PATTERNS = (
    'missing evidence',
    'no evidence',
    'not found',
    'insufficient evidence',
    'evidence gap',
)
_CANONICAL_MISSING_EVIDENCE_PREFIX = 'Missing Evidence:'
_CANONICAL_EVIDENCE_PATTERN = re.compile(
    r'evidence:\s*[^,\n][^,\n]*(?:,\s*page\s*[0-9]+)?',
    flags=re.IGNORECASE,
)
_EVIDENCE_FILENAME_PAGE_PATTERN = re.compile(
    r'\b[^\n|]*?\.(?:pdf|docx?|xlsx?|csv|md|txt)\b[^\n|]*\('
    r'\s*(?:pages?\s*[0-9]+(?:\s*[-–]\s*[0-9]+)?|section\s*:[^)]+)\s*\)',
    flags=re.IGNORECASE,
)
ORDERED_HEADINGS_REQUIREMENT = 'use the required headings exactly and in the requested order'
INCLUDE_HEADING_PREFIX = 'include heading:'
NESTED_BULLETS_PREFIX = 'use nested bullet lists with exactly'
MISSING_EVIDENCE_REQUIREMENT = 'explicitly call out missing evidence by requested group and/or year'
EVIDENCE_GROUNDING_REQUIREMENT = (
    'require canonical evidence grounding for every claim-bearing bullet/list item and '
    'narrative claim paragraph'
)
_DEFAULT_EVIDENCE_GROUNDING_EXCLUDED_SECTIONS = (
    'executive summary',
    'scope',
    'method',
    'confidence notes',
)
_MARKDOWN_PARSER = MarkdownIt('commonmark')
_LEADING_HEADING_NUMBER_PATTERN = re.compile(r'^\s*\d+[\).]\s*')
_TRAILING_HEADING_QUALIFIER_PATTERN = re.compile(
    r'\s+(?:only|alone|solely|exclusively)\s*$',
    re.IGNORECASE,
)
_INLINE_HEADING_DIRECTIVE_TAIL_PATTERN = re.compile(
    r'^(?P<head>.*?)\.\s+'
    r'(?P<tail>'
    r'under\b|in\s+each\b|include\b|keep\b|constraints?\b|with\b|where\b|for\s+each\b'
    r').*$',
    re.IGNORECASE,
)
_CURRENCY_OR_LONG_NUMBER_PATTERN = re.compile(
    r'(?:\$\s?\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\b\d{6,}\b)'
)


def _is_ordered_headings_requirement(requirement: str) -> bool:
    return requirement.strip().casefold().startswith(ORDERED_HEADINGS_REQUIREMENT)


def _extract_heading_from_requirement(requirement: str) -> str | None:
    normalized = requirement.strip()
    if not normalized.casefold().startswith(INCLUDE_HEADING_PREFIX):
        return None
    heading = normalized.split(':', 1)[1].strip()
    return heading or None


def _extract_bullet_depth_requirement(requirement: str) -> int | None:
    normalized = requirement.strip()
    if not normalized.casefold().startswith(NESTED_BULLETS_PREFIX):
        return None
    match = re.search(r'exactly\s+([0-9]+)\s+levels?', normalized, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _is_missing_evidence_requirement(requirement: str) -> bool:
    return requirement.strip().casefold().startswith(MISSING_EVIDENCE_REQUIREMENT)


def _is_evidence_grounding_requirement(requirement: str) -> bool:
    return requirement.strip().casefold().startswith(EVIDENCE_GROUNDING_REQUIREMENT)


@dataclass(frozen=True)
class OutputContractPlan:
    required_headings: tuple[str, ...]
    enforce_order: bool
    required_bullet_depth: int | None
    requires_missing_evidence_callout: bool
    max_words: int | None
    exact_top_level_bullets: int | None
    exact_top_level_bullets_section: str | None
    requires_evidence_grounding: bool
    evidence_grounding_excluded_sections: tuple[str, ...]
    requires_not_found_fallback: bool


def _clean_heading_label(value: str) -> str:
    heading = value.strip().strip('"').strip("'").strip()
    if '. ' in heading:
        heading = heading.split('. ', 1)[0].strip()
    heading = _strip_inline_heading_directive_tail(heading)
    heading = _TRAILING_HEADING_QUALIFIER_PATTERN.sub('', heading)
    heading = heading.rstrip('.,;')
    return heading


def _strip_inline_heading_directive_tail(value: str) -> str:
    match = _INLINE_HEADING_DIRECTIVE_TAIL_PATTERN.match(value.strip())
    if match is None:
        return value.strip()
    return match.group('head').strip()


def _strip_trailing_heading_annotation(value: str) -> str:
    trimmed = value.strip()
    closed_paren = re.search(r'\s*\(([^)]*)\)\s*$', trimmed)
    if closed_paren:
        annotation = (closed_paren.group(1) or '').casefold()
        annotation_has_list_shape = (annotation.count(',') >= 1 and len(annotation) >= 18)
        if re.search(r'\b(max|word|year|years|section|sections)\b', annotation) or re.search(
            r'\b(?:19|20)\d{2}\b',
            annotation,
        ) or annotation_has_list_shape:
            trimmed = re.sub(r'\s*\([^)]*\)\s*$', '', trimmed).strip()
            return trimmed

    open_paren = re.search(r'\s*\(([^)]*)$', trimmed)
    if open_paren:
        annotation = (open_paren.group(1) or '').casefold()
        annotation_has_list_shape = (annotation.count(',') >= 1 and len(annotation) >= 18)
        if re.search(r'\b(max|word|year|years|section|sections)\b', annotation) or re.search(
            r'\b(?:19|20)\d{2}\b',
            annotation,
        ) or annotation_has_list_shape:
            trimmed = re.sub(r'\s*\([^)]*$', '', trimmed).strip()
    return trimmed


def _normalize_heading_key(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r'^(?:\s*#{1,6}\s*)+', '', normalized)
    normalized = _LEADING_HEADING_NUMBER_PATTERN.sub('', normalized)
    normalized = _strip_trailing_heading_annotation(normalized)
    inline_directive = _INLINE_HEADING_DIRECTIVE_TAIL_PATTERN.match(normalized)
    if inline_directive is not None:
        normalized = inline_directive.group('head').strip()
    normalized = _TRAILING_HEADING_QUALIFIER_PATTERN.sub('', normalized)
    normalized = normalized.rstrip(' .')
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.casefold().strip()


def _line_start_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r'\n', text):
        offsets.append(match.end())
    return offsets


def _line_index_to_char_offset(
    *,
    line_index: int,
    offsets: list[int],
    text_length: int,
) -> int:
    if line_index < 0:
        return 0
    if line_index >= len(offsets):
        return text_length
    return offsets[line_index]


def _normalize_markdown_heading_fragment(text: str) -> str:
    # Queries commonly list multiple headings inline as "## A, ## B".
    return re.sub(r'\s*,\s*(#{2,6}\s+)', r'\n\1', text)


def _extract_markdown_heading_entries(
    text: str,
    *,
    allow_inline_heading_list: bool = False,
) -> list[tuple[str, int]]:
    normalized_text = (
        _normalize_markdown_heading_fragment(text)
        if allow_inline_heading_list
        else text
    )
    tokens = _MARKDOWN_PARSER.parse(normalized_text)
    offsets = _line_start_offsets(normalized_text)
    entries: list[tuple[str, int]] = []
    for idx, token in enumerate(tokens):
        if token.type != 'heading_open':
            continue
        level = 2
        if token.tag.startswith('h') and token.tag[1:].isdigit():
            level = int(token.tag[1:])
        inline_content = ''
        if idx + 1 < len(tokens) and tokens[idx + 1].type == 'inline':
            inline_content = tokens[idx + 1].content.strip()
        if not inline_content:
            continue
        heading = _clean_heading_label(f"{'#' * level} {inline_content}")
        if not heading:
            continue
        position = 0
        if token.map and len(token.map) == 2:
            position = _line_index_to_char_offset(
                line_index=token.map[0],
                offsets=offsets,
                text_length=len(normalized_text),
            )
        entries.append((heading, position))
    return entries


def _extract_markdown_headings(
    text: str,
    *,
    allow_inline_heading_list: bool = False,
) -> list[str]:
    headings: list[str] = []
    seen: set[str] = set()
    for heading, _ in _extract_markdown_heading_entries(
        text,
        allow_inline_heading_list=allow_inline_heading_list,
    ):
        if not heading:
            continue
        key = heading.casefold()
        if key in seen:
            continue
        seen.add(key)
        headings.append(heading)
    return headings


def _extract_explicit_order_headings(question: str) -> list[str]:
    patterns = (
        r'(?:headings?|sections?)\s+(?:in\s+this\s+)?exact\s+order\s*:\s*(.+?)(?:\.\s|$)',
        r'(?:required\s+)?headings?\s+in\s+order\s*:\s*(.+?)(?:\.\s|$)',
        r'output\s+must\s+contain\s*:\s*(.+?)(?:\.\s|$)',
        r'required\s+headings?\s+in\s+exact\s+order\s*:\s*(.+?)(?:\.\s|$)',
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        fragment = match.group(1).strip()
        parsed = _extract_markdown_headings(fragment, allow_inline_heading_list=True)
        if parsed:
            return parsed
    return []


def _extract_max_words(question: str) -> int | None:
    patterns = (
        r'<=\s*([0-9]{2,5})\s*words?',
        r'total\s*<=\s*([0-9]{2,5})\s*words?',
        r'(?:keep|limit)\s+(?:the\s+)?(?:total|overall)\s+(?:response\s+)?(?:under|<=|at\s+most)\s*([0-9]{2,5})\s*words?',
        r'(?:total|overall)\s+(?:response\s+)?(?:length\s+)?(?:must\s+be\s+)?(?:under|<=|at\s+most|max(?:imum)?(?:\s+of)?)\s*([0-9]{2,5})\s*words?',
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _parse_numeric_token(token: str) -> int | None:
    normalized = token.strip().casefold()
    if normalized.isdigit():
        return int(normalized)
    parsed = parse_number(normalized)
    if isinstance(parsed, int) and parsed > 0:
        return parsed
    return None


def _extract_exact_top_level_bullets(question: str) -> int | None:
    patterns = (
        r'(?:output|include|return)\s+exactly\s+([a-z0-9]+)\s+bullets?\b',
        r'exactly\s+([a-z0-9]+)\s+bullets?\b(?:\s+in\s+total)?',
        r'exactly\s+([a-z0-9]+)\s+top[-\s]*level\s+bullets?\b(?:\s+in\s+total)?',
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        if re.search(r'each\s+with\s+exactly', question, flags=re.IGNORECASE):
            continue
        parsed = _parse_numeric_token(match.group(1))
        if isinstance(parsed, int) and parsed > 0:
            return parsed
    return None


def _extract_exact_top_level_bullets_section(question: str) -> str | None:
    match = re.search(
        r'under\s+(##\s*[^,\n\.]+?)\s+include\s+exactly\s+[a-z0-9]+\s+bullets?\b',
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    section = _clean_heading_label(match.group(1))
    section = _strip_inline_heading_directive_tail(section)
    return section or None


def _requires_evidence_grounding(question: str) -> bool:
    return bool(
        re.search(
            r'\bevidence[-\s]*grounded\b|'
            r'\bcite\s+evidence\b|'
            r'\bwith\s+evidence\b|'
            r'\bevidence\s+only\b|'
            r'\bevidence\s+quality\b',
            question,
            flags=re.IGNORECASE,
        ),
    )


def _requires_not_found_fallback(question: str) -> bool:
    return bool(
        re.search(
            r'\bif (?:evidence|data|information) (?:is|are) (?:missing|absent)\b|\boutput\s+not\s+found\b|\buse\s+not\s+found\b',
            question,
            flags=re.IGNORECASE,
        ),
    )


def _build_output_contract_plan(
    *,
    question: str,
    format_requirements: list[str],
) -> OutputContractPlan:
    heading_items: list[str] = []
    seen: set[str] = set()
    enforce_order = False
    required_bullet_depth: int | None = None
    requires_missing_evidence_callout = False
    max_words: int | None = _extract_max_words(question)
    exact_top_level_bullets: int | None = _extract_exact_top_level_bullets(question)
    exact_top_level_bullets_section: str | None = _extract_exact_top_level_bullets_section(question)
    requires_evidence_grounding = _requires_evidence_grounding(question)
    requires_not_found_fallback = _requires_not_found_fallback(question)
    evidence_grounding_excluded_sections = tuple(
        _normalize_heading_key(section)
        for section in _DEFAULT_EVIDENCE_GROUNDING_EXCLUDED_SECTIONS
    )

    for requirement in format_requirements:
        normalized = requirement.strip()
        heading = _extract_heading_from_requirement(normalized)
        if heading is not None:
            key = heading.casefold()
            if key not in seen:
                heading_items.append(heading)
                seen.add(key)
            continue

        if _is_ordered_headings_requirement(normalized):
            enforce_order = True
            continue

        bullet_depth = _extract_bullet_depth_requirement(normalized)
        if bullet_depth is not None:
            required_bullet_depth = bullet_depth
            continue

        if _is_missing_evidence_requirement(normalized):
            requires_missing_evidence_callout = True
            continue

        if _is_evidence_grounding_requirement(normalized):
            requires_evidence_grounding = True

    # Fallback guard: question explicitly asks "in order" with numbered sections.
    if not enforce_order and re.search(r'\bin order\b', question, re.IGNORECASE) and heading_items:
        enforce_order = True

    explicit_order_headings = _extract_explicit_order_headings(question)
    if explicit_order_headings:
        heading_items = explicit_order_headings
        enforce_order = True
    else:
        for heading in _extract_markdown_headings(question, allow_inline_heading_list=True):
            cleaned_heading = _strip_inline_heading_directive_tail(heading)
            key = cleaned_heading.casefold()
            if key not in seen:
                heading_items.append(cleaned_heading)
                seen.add(key)

    return OutputContractPlan(
        required_headings=tuple(heading_items),
        enforce_order=enforce_order,
        required_bullet_depth=required_bullet_depth,
        requires_missing_evidence_callout=requires_missing_evidence_callout,
        max_words=max_words,
        exact_top_level_bullets=exact_top_level_bullets,
        exact_top_level_bullets_section=exact_top_level_bullets_section,
        requires_evidence_grounding=requires_evidence_grounding,
        evidence_grounding_excluded_sections=evidence_grounding_excluded_sections,
        requires_not_found_fallback=requires_not_found_fallback,
    )


def _build_contract_prompt_requirements(plan: OutputContractPlan) -> list[str]:
    requirements: list[str] = []
    if plan.required_headings and plan.enforce_order:
        ordered = ' -> '.join(plan.required_headings)
        requirements.append(f'follow this section skeleton in order: {ordered}')
        requirements.append(
            'emit every required heading as its own markdown heading line using explicit "#" markers at the level shown; '
            'do not add extra "#" prefixes, and do not merge, paraphrase, or skip required headings'
        )
        heading_template = '\n'.join(plan.required_headings[:6])
        if heading_template:
            requirements.append(
                'use this heading template shape (one heading per line):\n'
                f'{heading_template}'
            )
    if isinstance(plan.required_bullet_depth, int) and plan.required_bullet_depth >= 3:
        requirements.append(
            'in sections where nested bullets are required, ensure at least one bullet path reaches '
            'depth 3 (Parent -> Child -> Grandchild); child-only depth 2 is not sufficient. '
            'Example structure:\n- Parent\n  - Child\n    - Grandchild'
        )
    if plan.requires_missing_evidence_callout:
        requirements.append(
            'include explicit missing-evidence callouts using the phrase "Missing Evidence:" '
            'for requested groups/years where evidence is absent'
        )
        requirements.append(
            'include at least one line that starts exactly with "Missing Evidence:" '
            'when any requested evidence is absent'
        )
    if isinstance(plan.max_words, int) and plan.max_words > 0:
        requirements.append(f'keep total response length to at most {plan.max_words} words')
    if isinstance(plan.exact_top_level_bullets, int) and plan.exact_top_level_bullets > 0:
        if isinstance(plan.exact_top_level_bullets_section, str) and plan.exact_top_level_bullets_section.strip():
            requirements.append(
                f'under heading "{plan.exact_top_level_bullets_section}", '
                f'output exactly {plan.exact_top_level_bullets} bullet list items using "- " prefix'
            )
        else:
            requirements.append(
                f'output exactly {plan.exact_top_level_bullets} top-level bullet items using "- " prefix; '
                'do not use sub-bullets'
            )
    if plan.requires_evidence_grounding:
        requirements.append(
            'for every claim-bearing bullet/list item, include evidence metadata in canonical form '
            '"Evidence: {filename}, page {N}" (or "Evidence: {filename}" when page is unavailable)'
        )
        requirements.append(
            'for narrative claim paragraphs, include at least one evidence line per paragraph block '
            'using the same canonical "Evidence:" format'
        )
        required_heading_keys = {_normalize_heading_key(heading) for heading in plan.required_headings}
        if 'next verification steps' in required_heading_keys:
            requirements.append(
                'under heading "## Next Verification Steps", every numbered or bulleted step must include canonical '
                '"Evidence: {filename}, page {N}" (or "Evidence: {filename}") metadata'
            )
        if {'largest increase', 'largest decrease'} & required_heading_keys:
            requirements.append(
                'under headings "## Largest Increase" and "## Largest Decrease", express numeric delta claims as '
                'bullet items and include canonical "Evidence:" metadata on each delta claim'
            )
            requirements.append(
                'do not emit uncited numeric delta lines in "Largest Increase/Decrease"; '
                'each numeric line must include canonical "Evidence:" or explicit "Missing Evidence:"'
            )
        if 'contradictions and gaps' in required_heading_keys:
            requirements.append(
                'under "## Contradictions and Gaps", do not use bare "No contradictions found" lines; '
                'include canonical "Evidence:" or explicit "Missing Evidence:" on each contradiction line'
            )
        if {'findings by year', 'cross-year deltas'} & required_heading_keys:
            requirements.append(
                'under "## Findings by Year" and "## Cross-Year Deltas", every source, amount, and contradiction bullet '
                'must include canonical "Evidence:" metadata; if support is absent, use "Missing Evidence:" instead'
            )
        requirements.append(
            'do not infer missing facts; when evidence is insufficient for a required claim, output "Not found"'
        )
    elif plan.requires_not_found_fallback:
        requirements.append('when evidence is insufficient for a required claim, output "Not found"')
    return requirements


def _extract_top_level_bullet_blocks(answer: str) -> list[tuple[str, str | None]]:
    blocks: list[tuple[str, int]] = []
    heading_sections = _extract_heading_sections(answer)
    heading_markers: list[tuple[int, str]] = [
        (start, _normalize_heading_key(heading))
        for heading, _level, start in heading_sections
    ]

    def _nearest_heading_key(position: int) -> str | None:
        current: str | None = None
        for start, heading_key in heading_markers:
            if start > position:
                break
            current = heading_key
        return current

    current_lines: list[str] = []
    current_start_offset: int | None = None
    cursor = 0
    for raw_line in answer.splitlines(keepends=True):
        line = raw_line.rstrip('\n')
        if re.match(r'^(?:[-*]|\d+[.)])\s+\S+', line):
            if current_lines and isinstance(current_start_offset, int):
                blocks.append(('\n'.join(current_lines).strip(), current_start_offset))
            current_lines = [line]
            current_start_offset = cursor
            cursor += len(raw_line)
            continue
        if current_lines:
            current_lines.append(line)
        cursor += len(raw_line)
    if current_lines and isinstance(current_start_offset, int):
        blocks.append(('\n'.join(current_lines).strip(), current_start_offset))

    return [
        (block, _nearest_heading_key(start_offset))
        for block, start_offset in blocks
        if block
    ]


def _extract_narrative_claim_blocks(answer: str) -> list[tuple[str, str | None]]:
    blocks: list[tuple[str, str | None]] = []
    heading_sections = _extract_heading_sections(answer)
    heading_markers: list[tuple[int, str]] = [
        (start, _normalize_heading_key(heading))
        for heading, _level, start in heading_sections
    ]

    def _nearest_heading_key(position: int) -> str | None:
        current: str | None = None
        for start, heading_key in heading_markers:
            if start > position:
                break
            current = heading_key
        return current

    block_slices: list[tuple[int, str]] = []
    cursor = 0
    for separator in re.finditer(r'\n\s*\n+', answer):
        block_slices.append((cursor, answer[cursor:separator.start()]))
        cursor = separator.end()
    if cursor <= len(answer):
        block_slices.append((cursor, answer[cursor:]))

    for start_offset, raw_block in block_slices:
        block = raw_block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = lines[0]
        if (
            first_line.startswith('#')
            or first_line.startswith('- ')
            or first_line.startswith('* ')
            or first_line.startswith('|')
            or first_line.startswith('```')
            or first_line.startswith(_CANONICAL_MISSING_EVIDENCE_PREFIX)
        ):
            continue
        if all(line.startswith('|') for line in lines):
            continue
        if sum(1 for _ in re.finditer(r'\b\w+\b', block)) < 8:
            continue
        blocks.append((block, _nearest_heading_key(start_offset)))
    return blocks


def _extract_heading_positions(answer: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for heading, position in _extract_markdown_heading_entries(answer):
        key = _normalize_heading_key(heading)
        if key not in positions:
            positions[key] = position
    return positions


def _has_evidence_metadata(block: str) -> bool:
    if _CANONICAL_EVIDENCE_PATTERN.search(block):
        return True
    # Accept common filename/page shorthand in source inventory bullets.
    return bool(_EVIDENCE_FILENAME_PAGE_PATTERN.search(block))


def _is_non_claim_placeholder_block(block: str) -> bool:
    normalized = block.strip()
    if not normalized:
        return True
    # Strip common list/item prefixes and lightweight markdown emphasis wrappers.
    normalized = normalized.splitlines()[0].strip()
    normalized = re.sub(r'^(?:[-*]|\d+[.)])\s+', '', normalized, count=1)
    normalized = re.sub(
        r'^\*\*Missing Evidence\*\*\s*:\s*',
        'Missing Evidence: ',
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r'^\*\*([^*]+)\*\*\s*:\s*', '', normalized, count=1)
    normalized = re.sub(r'^\*\*([^*]+):\*\*\s*', '', normalized, count=1)
    normalized = re.sub(r'^(?:[-*]|\d+[.)])\s+', '', normalized, count=1)
    if not normalized:
        return True
    folded = normalized.casefold()
    if folded.startswith('missing evidence:') or folded.startswith('not found'):
        return True
    if 'not found' in folded and not _CURRENCY_OR_LONG_NUMBER_PATTERN.search(normalized):
        return True
    if (
        normalized.endswith(':')
        and sum(1 for _ in re.finditer(r'\b\w+\b', normalized)) <= 4
        and not re.search(r'\d', normalized)
    ):
        return True
    if 'no contradictions found' in folded:
        return True
    if not folded and re.search(r'no contradictions (?:were )?found', block, flags=re.IGNORECASE):
        return True
    if folded in {'contradictions:', 'contradictions'} and re.search(
        r'no contradictions (?:were )?found',
        block,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(re.match(r'^no contradictions (?:were )?found\b', folded))


def _extract_list_metrics(answer: str) -> tuple[int, int]:
    tokens = _MARKDOWN_PARSER.parse(answer)
    list_stack: list[str] = []
    max_bullet_depth = 0
    top_level_bullet_count = 0
    for token in tokens:
        if token.type == 'bullet_list_open':
            list_stack.append('bullet')
            if len(list_stack) > max_bullet_depth:
                max_bullet_depth = len(list_stack)
            continue
        if token.type == 'ordered_list_open':
            list_stack.append('ordered')
            if len(list_stack) > max_bullet_depth:
                max_bullet_depth = len(list_stack)
            continue
        if token.type in {'bullet_list_close', 'ordered_list_close'}:
            if list_stack:
                list_stack.pop()
            continue
        if token.type == 'list_item_open' and len(list_stack) == 1 and list_stack[-1] == 'bullet':
            top_level_bullet_count += 1
    return max_bullet_depth, top_level_bullet_count


def _extract_heading_sections(answer: str) -> list[tuple[str, int, int]]:
    sections: list[tuple[str, int, int]] = []
    heading_pattern = re.compile(r'^\s*(#{1,6})\s+(.+?)\s*$', re.MULTILINE)
    for match in heading_pattern.finditer(answer):
        hashes = match.group(1)
        label = match.group(2)
        level = len(hashes)
        heading = _clean_heading_label(f"{hashes} {label}")
        sections.append((heading, level, match.start()))
    return sections


def _extract_section_slice(answer: str, target_heading: str) -> str | None:
    sections = _extract_heading_sections(answer)
    if not sections:
        return None
    target_key = _normalize_heading_key(target_heading)
    target_index = None
    target_level = None
    for idx, (heading, level, _start) in enumerate(sections):
        if _normalize_heading_key(heading) == target_key:
            target_index = idx
            target_level = level
            break
    if target_index is None or target_level is None:
        return None
    start = sections[target_index][2]
    end = len(answer)
    for heading, level, section_start in sections[target_index + 1:]:
        if level <= target_level and _normalize_heading_key(heading) != target_key:
            end = section_start
            break
    return answer[start:end]


def _evaluate_output_contract(
    *,
    answer: str,
    plan: OutputContractPlan,
) -> dict[str, object]:
    missing_headings: list[str] = []
    order_violations: list[str] = []
    last_position = -1
    heading_positions = _extract_heading_positions(answer)

    for heading in plan.required_headings:
        heading_key = _normalize_heading_key(heading)
        position = heading_positions.get(heading_key)
        if position is None:
            missing_headings.append(heading)
            continue
        if plan.enforce_order and position < last_position:
            order_violations.append(heading)
        last_position = position

    max_bullet_depth, top_level_bullet_count = _extract_list_metrics(answer)
    bullet_depth_ok: bool | None = None
    if isinstance(plan.required_bullet_depth, int) and plan.required_bullet_depth > 0:
        bullet_depth_ok = max_bullet_depth >= plan.required_bullet_depth

    missing_evidence_callout_ok: bool | None = None
    missing_evidence_callout_canonical: bool | None = None
    missing_evidence_callout_legacy_only: bool | None = None
    if plan.requires_missing_evidence_callout:
        canonical_hit = False
        canonical_cell_pattern = re.compile(
            r'(?:^|\|)\s*Missing Evidence:',
            re.IGNORECASE,
        )
        for raw_line in answer.splitlines():
            candidate = raw_line.lstrip()
            candidate = re.sub(r'^(?:[-*]|\d+[.)])\s+', '', candidate, count=1)
            candidate = re.sub(
                r'^\*\*Missing Evidence\*\*\s*:\s*',
                'Missing Evidence: ',
                candidate,
                flags=re.IGNORECASE,
            )
            if candidate.startswith(_CANONICAL_MISSING_EVIDENCE_PREFIX) or canonical_cell_pattern.search(raw_line):
                canonical_hit = True
                break
        answer_folded = answer.casefold()
        legacy_hit = any(pattern in answer_folded for pattern in _MISSING_EVIDENCE_PATTERNS)
        # Strict phase: canonical phrase is required for pass/fail.
        # Legacy variants are retained only for migration telemetry.
        missing_evidence_callout_ok = canonical_hit
        missing_evidence_callout_canonical = canonical_hit
        missing_evidence_callout_legacy_only = (not canonical_hit and legacy_hit)

    word_count = len(answer.split())
    word_count_ok: bool | None = None
    if isinstance(plan.max_words, int) and plan.max_words > 0:
        word_count_ok = word_count <= plan.max_words

    top_level_bullet_count_ok: bool | None = None
    if isinstance(plan.exact_top_level_bullets, int) and plan.exact_top_level_bullets > 0:
        scoped_top_level_bullet_count = top_level_bullet_count
        if isinstance(plan.exact_top_level_bullets_section, str) and plan.exact_top_level_bullets_section.strip():
            section_slice = _extract_section_slice(answer, plan.exact_top_level_bullets_section)
            if section_slice is not None:
                _depth, scoped_count = _extract_list_metrics(section_slice)
                scoped_top_level_bullet_count = scoped_count
        top_level_bullet_count_ok = scoped_top_level_bullet_count == plan.exact_top_level_bullets
        top_level_bullet_count = scoped_top_level_bullet_count

    evidence_grounding_ok: bool | None = None
    evidence_bullet_coverage_rate: float | None = None
    evidence_narrative_coverage_rate: float | None = None
    evidence_claim_block_coverage_rate: float | None = None
    evidence_format_canonical_rate: float | None = None
    evidence_claim_block_count = 0
    evidence_requires_canonical_format: bool | None = None
    evidence_scope_mode: str | None = None
    evidence_min_blocks_required = 0
    evidence_missing_blocks: list[str] = []
    evidence_missing_blocks_preview: list[str] = []
    evidence_requirement_summary: str | None = None
    evidence_note = 'not_applicable'
    contradiction_placeholder_ok: bool | None = None
    uncited_delta_numeric_ok: bool | None = None
    contradiction_placeholder_hits: list[str] = []
    uncited_delta_numeric_lines: list[str] = []
    required_heading_keys = {_normalize_heading_key(heading) for heading in plan.required_headings}
    if plan.requires_evidence_grounding:
        excluded_heading_keys = {
            _normalize_heading_key(section)
            for section in plan.evidence_grounding_excluded_sections
            if isinstance(section, str) and section.strip()
        }
        bullet_blocks_with_headings = _extract_top_level_bullet_blocks(answer)
        raw_bullet_blocks = [
            block
            for block, heading_key in bullet_blocks_with_headings
            if heading_key not in excluded_heading_keys
        ]
        bullet_blocks = list(raw_bullet_blocks)
        narrative_blocks_with_headings = _extract_narrative_claim_blocks(answer)
        raw_narrative_blocks = [
            block
            for block, heading_key in narrative_blocks_with_headings
            if heading_key not in excluded_heading_keys
        ]
        narrative_blocks = list(raw_narrative_blocks)
        bullet_blocks = [block for block in bullet_blocks if not _is_non_claim_placeholder_block(block)]
        narrative_blocks = [block for block in narrative_blocks if not _is_non_claim_placeholder_block(block)]
        evidence_scope_mode = (
            'bullet_and_narrative'
            if bullet_blocks and narrative_blocks
            else 'bullet_only'
            if bullet_blocks
            else 'narrative_only'
            if narrative_blocks
            else 'none'
        )
        evidence_requires_canonical_format = True
        evidence_min_blocks_required = len(bullet_blocks) + len(narrative_blocks)
        evidence_claim_block_count = evidence_min_blocks_required

        bullet_evidence_hits = sum(1 for block in bullet_blocks if _has_evidence_metadata(block))
        narrative_evidence_hits = sum(
            1 for block in narrative_blocks if _has_evidence_metadata(block)
        )
        total_hits = bullet_evidence_hits + narrative_evidence_hits

        if bullet_blocks:
            evidence_bullet_coverage_rate = round(bullet_evidence_hits / len(bullet_blocks), 3)
        if narrative_blocks:
            evidence_narrative_coverage_rate = round(narrative_evidence_hits / len(narrative_blocks), 3)

        if evidence_claim_block_count > 0:
            evidence_claim_block_coverage_rate = round(total_hits / evidence_claim_block_count, 3)
            evidence_format_canonical_rate = evidence_claim_block_coverage_rate
            evidence_grounding_ok = (total_hits == evidence_claim_block_count)
            evidence_note = 'pass' if evidence_grounding_ok else 'missing_evidence_in_claim_blocks'
        else:
            has_placeholder_only_fallback = bool(raw_bullet_blocks or raw_narrative_blocks) and all(
                _is_non_claim_placeholder_block(block)
                for block in [*raw_bullet_blocks, *raw_narrative_blocks]
            )
            if has_placeholder_only_fallback:
                evidence_claim_block_coverage_rate = 1.0
                evidence_format_canonical_rate = 1.0
                evidence_grounding_ok = True
                evidence_note = 'placeholder_only_fallback'
            else:
                evidence_claim_block_coverage_rate = 0.0
                evidence_format_canonical_rate = 0.0
                evidence_grounding_ok = False
                evidence_note = 'no_claim_blocks_found_for_evidence_enforcement'

        for block in bullet_blocks:
            if _has_evidence_metadata(block):
                continue
            evidence_missing_blocks.append('bullet')
            evidence_missing_blocks_preview.append(block.splitlines()[0][:120])
        for block in narrative_blocks:
            if _has_evidence_metadata(block):
                continue
            evidence_missing_blocks.append('narrative')
            evidence_missing_blocks_preview.append(re.sub(r'\s+', ' ', block)[:120])

        evidence_requirement_summary = (
            f'required canonical Evidence metadata in {evidence_claim_block_count} claim block(s); '
            f'satisfied {total_hits}'
        )

        if 'contradictions and gaps' in required_heading_keys:
            contradiction_lines = [
                line.strip()
                for line in answer.splitlines()
                if re.search(
                    r'^\s*\*\*Contradictions:\*\*\s*-\s*No contradictions (?:were )?found',
                    line,
                    flags=re.IGNORECASE,
                )
            ]
            for line in contradiction_lines:
                line_folded = line.casefold()
                if 'evidence:' in line_folded or 'missing evidence:' in line_folded:
                    continue
                contradiction_placeholder_hits.append(line[:160])
            contradiction_placeholder_ok = len(contradiction_placeholder_hits) == 0

        if {'largest increase', 'largest decrease'} & required_heading_keys:
            candidate_sections = ('## Largest Increase', '## Largest Decrease')
            for section_heading in candidate_sections:
                section_slice = _extract_section_slice(answer, section_heading)
                if not isinstance(section_slice, str) or not section_slice.strip():
                    continue
                for block, _heading_key in _extract_top_level_bullet_blocks(section_slice):
                    if not isinstance(block, str):
                        continue
                    if not _CURRENCY_OR_LONG_NUMBER_PATTERN.search(block):
                        continue
                    block_folded = block.casefold()
                    if 'evidence:' in block_folded or 'missing evidence:' in block_folded:
                        continue
                    uncited_delta_numeric_lines.append(block.splitlines()[0][:160])
            uncited_delta_numeric_ok = len(uncited_delta_numeric_lines) == 0

    not_found_fallback_ok: bool | None = None
    if plan.requires_not_found_fallback:
        # Presence check only: enforcement stays permissive unless request requires this fallback.
        not_found_fallback_ok = ('not found' in answer.casefold())

    passed = (
        not missing_headings
        and not order_violations
        and (bullet_depth_ok is not False)
        and (missing_evidence_callout_ok is not False)
        and (word_count_ok is not False)
        and (top_level_bullet_count_ok is not False)
        and (evidence_grounding_ok is not False)
        and (contradiction_placeholder_ok is not False)
        and (uncited_delta_numeric_ok is not False)
        and (not_found_fallback_ok is not False)
    )

    return {
        'required_headings': list(plan.required_headings),
        'enforce_order': plan.enforce_order,
        'required_bullet_depth': plan.required_bullet_depth,
        'requires_missing_evidence_callout': plan.requires_missing_evidence_callout,
        'max_words': plan.max_words,
        'exact_top_level_bullets': plan.exact_top_level_bullets,
        'exact_top_level_bullets_section': plan.exact_top_level_bullets_section,
        'requires_evidence_grounding': plan.requires_evidence_grounding,
        'evidence_grounding_excluded_sections': list(plan.evidence_grounding_excluded_sections),
        'requires_not_found_fallback': plan.requires_not_found_fallback,
        'missing_headings': missing_headings,
        'has_content_gap': bool(missing_headings),
        'order_violations': order_violations,
        'max_bullet_depth': max_bullet_depth,
        'bullet_depth_ok': bullet_depth_ok,
        'missing_evidence_callout_ok': missing_evidence_callout_ok,
        'missing_evidence_callout_canonical': missing_evidence_callout_canonical,
        'missing_evidence_callout_legacy_only': missing_evidence_callout_legacy_only,
        'word_count': word_count,
        'word_count_ok': word_count_ok,
        'top_level_bullet_count': top_level_bullet_count,
        'top_level_bullet_count_ok': top_level_bullet_count_ok,
        'evidence_grounding_ok': evidence_grounding_ok,
        'evidence_bullet_coverage_rate': evidence_bullet_coverage_rate,
        'evidence_narrative_coverage_rate': evidence_narrative_coverage_rate,
        'evidence_claim_block_coverage_rate': evidence_claim_block_coverage_rate,
        'evidence_format_canonical_rate': evidence_format_canonical_rate,
        'evidence_claim_block_count': evidence_claim_block_count,
        'evidence_requires_canonical_format': evidence_requires_canonical_format,
        'evidence_scope_mode': evidence_scope_mode,
        'evidence_min_blocks_required': evidence_min_blocks_required,
        'evidence_missing_blocks': evidence_missing_blocks,
        'evidence_missing_blocks_preview': evidence_missing_blocks_preview[:8],
        'evidence_requirement_summary': evidence_requirement_summary,
        'evidence_note': evidence_note,
        'contradiction_placeholder_ok': contradiction_placeholder_ok,
        'contradiction_placeholder_hits': contradiction_placeholder_hits[:8],
        'uncited_delta_numeric_ok': uncited_delta_numeric_ok,
        'uncited_delta_numeric_lines': uncited_delta_numeric_lines[:8],
        'not_found_fallback_ok': not_found_fallback_ok,
        'passed': passed,
    }
