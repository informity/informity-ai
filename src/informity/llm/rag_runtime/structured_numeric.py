import re

from informity.llm import contract_prompt_parser as _contract_prompt_parser

_NUMBER_PATTERN = re.compile(r'\(?\$?\d[\d,]*(?:\.\d{1,2})?\)?')
_FIELD_LABEL_NEAR_NUMBER_PATTERN = re.compile(r'([A-Za-z][A-Za-z0-9\s/_-]{1,36})$')
_REQUESTED_COLUMNS_PATTERN = re.compile(
    r'\bcolumns?\s*:\s*([^\n]+?)(?:\.\s|$)',
    re.IGNORECASE,
)
_PIPE_FORMAT_LABELS_PATTERN = re.compile(
    r'\bformat\s*:\s*([^\n]+?)(?:\.\s|$)',
    re.IGNORECASE,
)


def _normalize_hint_to_regex(field_hint: str) -> re.Pattern[str]:
    normalized = field_hint.replace('_', ' ').strip().lower()
    tokens = [re.escape(token) for token in normalized.split() if token]
    if not tokens:
        return re.compile(r'(?!)')
    pattern = r'\b' + r'[\s:._-]*'.join(tokens) + r'\b'
    return re.compile(pattern, re.IGNORECASE)


def _parse_numeric_token(raw_value: str) -> tuple[float, bool] | None:
    token = raw_value.strip()
    if not token:
        return None
    is_negative = token.startswith('(') and token.endswith(')')
    cleaned = token.strip('()').replace('$', '').replace(',', '').replace(' ', '')
    if cleaned.endswith('%'):
        cleaned = cleaned[:-1]
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if is_negative:
        value = -value
    has_currency_shape = ('$' in token) or (',' in token) or ('.' in token)
    is_year_like = cleaned.isdigit() and len(cleaned) == 4 and 1900 <= int(cleaned) <= 2099
    if is_year_like and not has_currency_shape:
        return None
    return value, has_currency_shape


def _extract_candidate_values(
    *,
    chunk_text: str,
    field_hint: str | None,
    base_score: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    hint_spans: list[tuple[int, int]] = []
    if field_hint:
        hint_pattern = _normalize_hint_to_regex(field_hint)
        hint_spans = [match.span() for match in hint_pattern.finditer(chunk_text)]
    for match in _NUMBER_PATTERN.finditer(chunk_text):
        parsed = _parse_numeric_token(match.group(0))
        if parsed is None:
            continue
        value, has_currency_shape = parsed
        if field_hint is None and not has_currency_shape:
            # Without a field hint, avoid plain integers (IDs, years, page numbers).
            continue
        confidence = float(base_score)
        if has_currency_shape:
            confidence += 0.45
        if hint_spans:
            candidate_center = (match.start() + match.end()) // 2
            nearest_distance = min(
                abs(candidate_center - ((hint_start + hint_end) // 2))
                for hint_start, hint_end in hint_spans
            )
            proximity_bonus = max(0.0, 1.3 - (min(nearest_distance, 220) / 180.0))
            confidence += proximity_bonus
        elif field_hint is not None:
            confidence -= 0.7
        candidates.append({
            'value': value,
            'raw_value': match.group(0),
            'confidence': confidence,
            'start': match.start(),
            'end': match.end(),
        })
    candidates.sort(key=lambda item: float(item['confidence']), reverse=True)
    return candidates


def _infer_field_label(chunk_text: str, start_idx: int, field_hint: str | None) -> str:
    if field_hint:
        return field_hint.strip()
    prefix = chunk_text[max(0, start_idx - 60):start_idx]
    tail = re.sub(r'[\s:|]+$', '', prefix)
    match = _FIELD_LABEL_NEAR_NUMBER_PATTERN.search(tail)
    if not match:
        return 'value'
    candidate = match.group(1).strip().lower()
    candidate = re.sub(r'\s+', ' ', candidate)
    if len(candidate) > 28:
        return 'value'
    alpha_count = sum(1 for char in candidate if char.isalpha())
    if alpha_count < 2:
        return 'value'
    return candidate


def _build_evidence_span(chunk_text: str, start_idx: int, end_idx: int, radius: int = 90) -> str:
    span_start = max(0, start_idx - radius)
    span_end = min(len(chunk_text), end_idx + radius)
    snippet = chunk_text[span_start:span_end].replace('\n', ' ')
    return re.sub(r'\s+', ' ', snippet).strip()


def _extract_required_years(question: str) -> list[int]:
    return _contract_prompt_parser.extract_required_years(question)


def _extract_exact_top_level_bullet_limit(question: str) -> int | None:
    patterns = (
        r'exactly\s+(\d+)\s+top[-\s]level\s+bullets?',
        r'top[-\s]level\s+bullets?\s*:\s*(\d+)',
        r'exactly\s+(\d+)\s+bullets?',
    )
    for pattern in patterns:
        match = re.search(pattern, question or '', re.IGNORECASE)
        if not match:
            continue
        try:
            parsed = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _extract_requested_table_columns(question: str) -> list[str]:
    match = _REQUESTED_COLUMNS_PATTERN.search(question or '')
    if not match:
        return []
    raw_columns = re.sub(r'\s+', ' ', match.group(1).strip())
    if not raw_columns:
        return []
    normalized = re.sub(r'\s+and\s+', ', ', raw_columns, flags=re.IGNORECASE)
    parts = [part.strip(' "\'`.') for part in normalized.split(',')]
    return [part for part in parts if part]


def _extract_requested_pipe_labels(question: str) -> list[str]:
    match = _PIPE_FORMAT_LABELS_PATTERN.search(question or '')
    if not match:
        return []
    raw_parts = [part.strip() for part in match.group(1).split('|')]
    labels = [part for part in raw_parts if part]
    if len(labels) < 3:
        return []
    return labels[:3]




def _extract_required_headings(question: str) -> list[str]:
    return _contract_prompt_parser.extract_required_headings(question)


def _extract_required_markdown_table_columns(question: str) -> list[str]:
    text = str(question or '')
    patterns = (
        r'markdown\s+table\s+with\s+columns?\s*:\s*([^\n.]+)',
        r'columns?\s*:\s*([^\n.]+)',
    )
    raw_columns = ''
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            raw_columns = str(match.group(1) or '').strip()
            if raw_columns:
                break
    if not raw_columns:
        return []

    split_candidates = re.split(r',|\||\band\b', raw_columns, flags=re.IGNORECASE)
    columns: list[str] = []
    seen: set[str] = set()
    for candidate in split_candidates:
        normalized = str(candidate or '').strip().strip('`').strip('"').strip("'")
        normalized = re.sub(r'\s+', ' ', normalized).rstrip(' .;:')
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        columns.append(normalized)
        if len(columns) >= 8:
            break
    return columns


def _derive_format_requirements(
    question: str,
    action_hints: dict[str, bool] | None = None,
) -> list[str]:
    requirements: list[str] = []
    seen_requirements: set[str] = set()

    def _append_requirement(value: str) -> None:
        normalized = str(value or '').strip()
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen_requirements:
            return
        seen_requirements.add(key)
        requirements.append(normalized)

    headings = _extract_required_headings(question)
    if headings:
        has_ordered_cue = _contract_prompt_parser.has_ordered_heading_cue(question)
        if has_ordered_cue:
            _append_requirement('use the required headings exactly and in the requested order')
        else:
            _append_requirement('use all headings explicitly requested by the user')
        for heading in headings[:12]:
            _append_requirement(f'include heading: {heading}')
    bullet_depth_match = re.search(r'exactly\s+([23])\s+levels?', question, re.IGNORECASE)
    if bullet_depth_match:
        _append_requirement(f'use nested bullet lists with exactly {bullet_depth_match.group(1)} levels where requested')
    year_subsection_cues = [
        r'one\s+subsection\s+per\s+(?:indexed|available|requested)?\s*year',
        r'for\s+each\s+year',
        r'findings\s+by\s+year',
    ]
    if any(re.search(pattern, question, re.IGNORECASE) for pattern in year_subsection_cues):
        _append_requirement('for year-grouped sections, include one subsection per year using markdown headings like "### YYYY"')
        if re.search(r'across\s+all\s+indexed\s+records|year[-\s]*over[-\s]*year|cross[-\s]*year', question, re.IGNORECASE):
            _append_requirement('when multiple years are available in context, include at least 2 distinct year subsections')
    if re.search(r'missing evidence|missing records|gaps', question, re.IGNORECASE):
        _append_requirement('explicitly call out missing evidence by requested group and/or year')
    # action_hints are additive signals only: they do not replace regex/user-contract cues,
    # and they flow through the same deduplicated append path for deterministic behavior.
    if bool(action_hints and action_hints.get('should_enumerate')):
        _append_requirement('present findings as a numbered or bulleted list when no stricter format contract overrides it')
    if bool(action_hints and action_hints.get('should_compare')):
        _append_requirement('use a side-by-side or structured comparison format grounded in retrieved evidence')
    required_terms = _extract_required_terms_from_user_contract(question)
    for term in required_terms[:10]:
        _append_requirement(f'include term: {term}')
    table_columns = _extract_required_markdown_table_columns(question)
    if table_columns:
        _append_requirement(f'include markdown table columns: {" | ".join(table_columns)}')
    pipe_labels = _extract_requested_pipe_labels(question)
    if pipe_labels:
        _append_requirement(f'for delimiter schemas, include exact header/template line: {" | ".join(pipe_labels)}')
        for label in pipe_labels[:8]:
            _append_requirement(f'include term: {label}')
    return requirements


def _extract_required_terms_from_user_contract(question: str) -> list[str]:
    clauses: list[str] = []
    for cue in ('include', 'cover'):
        pattern = re.compile(rf'\b{cue}\b\s+(.+?)(?:[.;]|$)', re.IGNORECASE | re.DOTALL)
        for match in pattern.finditer(question):
            clause = str(match.group(1) or '').strip()
            if clause:
                clauses.append(clause)
    if not clauses:
        return []
    stopwords = {
        'a', 'an', 'and', 'or', 'the', 'this', 'that', 'these', 'those',
        'with', 'from', 'for', 'of', 'to', 'in', 'on', 'by', 'as', 'at',
        'all', 'across', 'available', 'indexed', 'records', 'record',
        'different', 'where', 'only', 'using', 'grounded', 'likely', 'biggest',
        'clear', 'key', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'do', 'not', 'until', 'are', 'is', 'be',
        'include', 'cover', 'request', 'requested', 'output',
    }
    terms: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        for token in re.findall(r'[a-z][a-z-]{2,}', clause.casefold()):
            normalized = token.strip('-')
            if not normalized or normalized in stopwords:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
            if len(terms) >= 12:
                return terms
    return terms

