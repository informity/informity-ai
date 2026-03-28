import re
from collections import defaultdict

import aiosqlite

from informity.api.schemas import ChatSourceReference
from informity.config import settings
from informity.llm.query_classifier import QueryClassification
from informity.llm.query_patterns import build_conflict_amount_pattern
from informity.llm.rag_runtime.retrieval_validation import _normalize_relevance_score
from informity.llm.types import GroupBy, OutputShape, QuerySubtype

_STRUCTURED_EXTRACTION_SUBTYPES = {QuerySubtype.EXTRACT_STRUCTURED_VALUES, QuerySubtype.AGGREGATE_BY_PERIOD}
_NUMBER_PATTERN = re.compile(r'\(?\$?\d[\d,]*(?:\.\d{1,2})?\)?')
_FIELD_LABEL_NEAR_NUMBER_PATTERN = re.compile(r'([A-Za-z][A-Za-z0-9\s/_-]{1,36})$')
_EXPLICIT_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')
_REQUESTED_COLUMNS_PATTERN = re.compile(
    r'\bcolumns?\s*:\s*([^\n]+?)(?:\.\s|$)',
    re.IGNORECASE,
)
_PIPE_FORMAT_LABELS_PATTERN = re.compile(
    r'\bformat\s*:\s*([^\n]+?)(?:\.\s|$)',
    re.IGNORECASE,
)
_CONFLICT_AMOUNT_PATTERN = build_conflict_amount_pattern()
_SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_FINANCE_EVIDENCE_TOKEN_PATTERN = re.compile(r'[a-z]{3,}')
_FINANCE_EVIDENCE_STOPWORDS = {
    'and', 'the', 'for', 'from', 'with', 'that', 'this', 'are', 'was', 'were',
    'not', 'found', 'values', 'value', 'reported', 'report', 'based', 'entries',
    'amount', 'amounts', 'document', 'documents', 'file', 'files',
}
_ORDERED_HEADING_CUES = (
    r'\bin order\b',
    r'\bin this order\b',
    r'\bin sequence\b',
    r'output\s+must\s+contain\s*:\s*##',
    r'sections?\s+must\s+contain\s*:\s*##',
    r'headings?\s+exactly',
    r'headings?\s+in\s+exact\s+order',
)


def _should_run_structured_extraction(
    *,
    question: str,
    classification: QueryClassification,
    response_shape: OutputShape,
) -> bool:
    if _is_finance_conflict_prompt(question):
        return True
    required_headings = _extract_required_headings(question)
    has_required_headings = bool(required_headings)
    has_ordered_heading_cue = any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in _ORDERED_HEADING_CUES
    )
    has_strict_heading_contract = has_required_headings and has_ordered_heading_cue
    has_max_words = bool(re.search(r'(?:total\s*)?(?:<=|less than or equal to)\s*(\d+)\s*words?', question, re.IGNORECASE))
    has_exact_top_level_bullets = _extract_exact_top_level_bullet_limit(question) is not None
    has_global_strict_contract = has_max_words and has_exact_top_level_bullets and not has_required_headings
    requires_missing_evidence_callout = bool(
        re.search(r'\bmissing\s+evidence\b', question, re.IGNORECASE)
    )
    if requires_missing_evidence_callout or bool(
        re.search(r'\bmissing\s+evidence\b|\bgaps?\b', question, re.IGNORECASE)
    ):
        return False
    has_explicit_extraction_cue = bool(
        re.search(r'\b(?:extract|table|line item|box-level|box)\b', question, re.IGNORECASE)
    )

    if has_strict_heading_contract:
        return False
    if has_global_strict_contract and not has_explicit_extraction_cue:
        return False
    if classification.field_hint is not None:
        return True
    return (
        response_shape in {OutputShape.STRUCTURED_EXTRACT, OutputShape.METADATA_TABLE}
        and classification.subtype in _STRUCTURED_EXTRACTION_SUBTYPES
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
    years = sorted({int(match.group(0)) for match in _EXPLICIT_YEAR_PATTERN.finditer(question)})
    return years


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


def _is_finance_conflict_prompt(question: str) -> bool:
    if not question.strip():
        return False
    return bool(_CONFLICT_AMOUNT_PATTERN.search(question))


def _normalize_field_category(field_label: str, evidence_span: str) -> str:
    folded = f'{field_label} {evidence_span}'.casefold()
    if 'interest' in folded:
        return 'interest'
    if 'dividend' in folded:
        return 'dividend'
    if 'balance' in folded:
        return 'balance'
    if 'total' in folded:
        return 'total'
    if 'gross receipt' in folded or 'gross receipts' in folded:
        return 'gross_receipts'
    return 'unknown'


def _normalize_numeric_token(raw_value: str) -> str:
    return re.sub(r'[^0-9.\-]', '', raw_value or '')


def _evidence_overlap_tokens(left_evidence: str, right_evidence: str) -> int:
    left_tokens = {
        token for token in _FINANCE_EVIDENCE_TOKEN_PATTERN.findall((left_evidence or '').casefold())
        if token not in _FINANCE_EVIDENCE_STOPWORDS
    }
    right_tokens = {
        token for token in _FINANCE_EVIDENCE_TOKEN_PATTERN.findall((right_evidence or '').casefold())
        if token not in _FINANCE_EVIDENCE_STOPWORDS
    }
    if not left_tokens or not right_tokens:
        return 0
    return len(left_tokens & right_tokens)


def _is_plausible_finance_value(raw_value: str, numeric_value: float) -> bool:
    token = (raw_value or '').strip()
    if not token:
        return False
    compact_digits = re.sub(r'[^0-9]', '', token)
    if not compact_digits:
        return False
    # Reject OCR-concatenated long integers without separators (common in scanned forms).
    max_unformatted_digits = int(getattr(settings, 'extraction_numeric_max_unformatted_digits', 9) or 9)
    if len(compact_digits) > max_unformatted_digits and ',' not in token and '.' not in token:
        return False
    # Reject obviously implausible magnitudes for document-level totals/balances in this flow.
    max_abs_value = float(getattr(settings, 'extraction_numeric_max_abs_value', 100000000.0) or 100000000.0)
    return not abs(float(numeric_value)) > max_abs_value


def _build_finance_conflict_placeholder_bullet() -> str:
    return (
        '- Conflict Statement: Not found; '
        + 'Involved Documents: Not found; '
        + 'Conflicting Values: Not found; '
        + 'Likely Reason: Not found; '
        + 'Missing Evidence: no comparable source pair found.'
    )


def _render_finance_conflict_bullets(
    *,
    selected_conflicts: list[dict[str, object]],
    bullet_limit: int = 4,
) -> str:
    resolved_limit = bullet_limit if bullet_limit > 0 else 4
    lines: list[str] = []
    for conflict in selected_conflicts[:resolved_limit]:
        statement = str(conflict.get('statement') or 'Not found')
        docs = str(conflict.get('docs') or 'Not found')
        values = str(conflict.get('values') or 'Not found')
        reason = str(conflict.get('reason') or 'Not found')
        evidence_files = []
        for row in conflict.get('rows', ()):
            if not isinstance(row, dict):
                continue
            filename = str(row.get('filename') or '').strip()
            if filename and filename not in evidence_files:
                evidence_files.append(filename)
        evidence_suffix = 'Missing Evidence: no comparable source pair found.'
        if evidence_files:
            evidence_suffix = '; '.join([f'Evidence: {filename}' for filename in evidence_files[:2]])
        lines.append(
            '- '
            + f'Conflict Statement: {statement}; '
            + f'Involved Documents: {docs}; '
            + f'Conflicting Values: {values}; '
            + f'Likely Reason: {reason}; '
            + evidence_suffix
        )
    while len(lines) < resolved_limit:
        lines.append(_build_finance_conflict_placeholder_bullet())
    return '\n'.join(lines).strip()


def _extract_required_headings(question: str) -> list[str]:
    headings: list[str] = []
    seen: set[str] = set()
    markdown_headings = re.findall(r'##\s+([^\n#]+)', question or '')
    for raw_heading in markdown_headings:
        normalized = str(raw_heading).strip().rstrip(' .')
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        headings.append(normalized)

    numbered_heads = re.findall(
        r'(?:^|:\s*|,\s*)(?:\d+\)\s*)(.+?)(?=(?:,\s*\d+\)\s)|(?:\.\s|$))',
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_heading in numbered_heads:
        heading = raw_heading.strip().rstrip(' .')
        if heading.casefold() in seen:
            continue
        seen.add(heading.casefold())
        headings.append(heading)
    return headings


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
        has_ordered_cue = any(re.search(pattern, question, re.IGNORECASE) for pattern in _ORDERED_HEADING_CUES)
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


def _validate_structured_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validated: list[dict[str, object]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        field_label = str(row.get('field_label', '')).strip().casefold()
        raw_value = str(row.get('raw_value', '')).strip()
        evidence_span = str(row.get('evidence_span', '')).strip()
        if _SSN_PATTERN.search(raw_value) or _SSN_PATTERN.search(evidence_span):
            continue
        if field_label and (len(field_label) > 32 or sum(1 for char in field_label if char.isalpha()) < 2):
            field_label = 'value'
            row['field_label'] = field_label
        if not raw_value or raw_value.endswith(','):
            continue
        compact_digits = re.sub(r'[^0-9]', '', raw_value)
        if not compact_digits:
            continue
        if (
            len(compact_digits) >= 8
            and '.' in raw_value
            and ',' not in raw_value
        ):
            # Reject likely OCR-concatenated numerics ("79453187.30") without locale separators.
            continue
        if re.search(r'\b(?:box|line|field)\s*([0-9]{1,3}[a-z]?)\b', field_label):
            label_index_match = re.search(r'([0-9]{1,3})', field_label)
            if label_index_match and compact_digits.isdigit() and int(compact_digits) == int(label_index_match.group(1)):
                continue
        dedupe_key = (
            int(row.get('file_id', 0)),
            raw_value,
            evidence_span.casefold()[:120],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        validated.append(row)
    return sorted(validated, key=lambda item: float(item.get('confidence', 0.0)), reverse=True)


async def _fetch_file_metadata(
    db: aiosqlite.Connection,
    file_ids: list[int],
) -> dict[int, dict[str, object]]:
    if not file_ids:
        return {}
    placeholders = ','.join('?' * len(file_ids))
    cursor = await db.execute(
        f'SELECT id, filename, path, year, category FROM files WHERE id IN ({placeholders})',
        file_ids,
    )
    rows = await cursor.fetchall()
    return {
        int(row['id']): {
            'filename': row['filename'] or '',
            'file_path': row['path'] or '',
            'year': row['year'],
            'category': row['category'],
        }
        for row in rows
    }


def _group_label_for_file(
    *,
    group_by: GroupBy | None,
    metadata: dict[str, object],
) -> str:
    if group_by == GroupBy.YEAR:
        year = metadata.get('year')
        return str(year) if year is not None else 'Unknown Year'
    if group_by == GroupBy.CATEGORY:
        category = metadata.get('category')
        return str(category) if category else 'Unknown Category'
    filename = str(metadata.get('filename') or '').strip()
    return filename or 'Unknown File'


def _render_structured_rows_answer(
    rows: list[dict[str, object]],
    *,
    table_columns: list[str] | None = None,
) -> str:
    if not rows:
        return 'I could not extract validated structured values from the retrieved context.'
    normalized_columns = [str(col).strip() for col in (table_columns or []) if str(col).strip()]
    if len(normalized_columns) < 3:
        normalized_columns = ['Field', 'Value', 'Source Snippet']
    lines = [
        '### Deterministic Structured Extraction',
        '',
        f"| {' | '.join(normalized_columns[:3])} |",
        '| --- | ---: | --- |',
    ]
    for row in rows:
        snippet = str(row['evidence_span']).strip()
        if len(snippet) > 140:
            snippet = snippet[:137] + '...'
        lines.append(f"| {row['field_label']} | {row['raw_value']} | {snippet} |")
    return '\n'.join(lines).strip()


def _render_structured_rows_bullets_answer(
    rows: list[dict[str, object]],
    bullet_limit: int,
    *,
    header_labels: list[str] | None = None,
) -> str:
    if bullet_limit <= 0:
        return _render_structured_rows_answer(rows)
    lines = ['### Deterministic Structured Extraction', '']
    normalized_labels = [str(label).strip() for label in (header_labels or []) if str(label).strip()]
    if len(normalized_labels) < 3:
        normalized_labels = ['Field', 'Value', 'Source Snippet']
    source_label = normalized_labels[2]
    lines.append(' | '.join(normalized_labels[:3]))
    lines.append('')
    selected_rows = rows[:bullet_limit]
    for row in selected_rows:
        snippet = str(row.get('evidence_span', '')).strip()
        if len(snippet) > 140:
            snippet = snippet[:137] + '...'
        lines.append(
            f"- {row.get('field_label', 'value')} | {row.get('raw_value', '')} | {source_label}: {snippet}"
        )
    while len([line for line in lines if line.startswith('- ')]) < bullet_limit:
        lines.append(
            f'- Missing Evidence | N/A | {source_label}: Missing Evidence: insufficient validated rows for requested bullet count.'
        )
    return '\n'.join(lines).strip()


def _render_year_aggregate_answer(
    *,
    rows: list[dict[str, object]],
    metadata_by_file_id: dict[int, dict[str, object]],
    required_years: list[int],
    available_years: list[int] | None = None,
    field_hint: str | None = None,
) -> str:
    values_by_year: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        file_id = int(row.get('file_id', 0))
        metadata = metadata_by_file_id.get(file_id, {})
        year_value = metadata.get('year')
        if isinstance(year_value, int):
            values_by_year[year_value].append(row)

    if required_years:
        normalized_required_years = sorted({int(year) for year in required_years})
        min_required_year = min(normalized_required_years)
        max_required_year = max(normalized_required_years)
        if (max_required_year - min_required_year) <= 15:
            display_years = list(range(min_required_year, max_required_year + 1))
        else:
            display_years = normalized_required_years
    else:
        display_years = sorted(set(values_by_year.keys()) | set(available_years or []))
        if display_years:
            min_display_year = min(display_years)
            max_display_year = max(display_years)
            # Keep year tables continuous for bounded spans so interior years are represented
            # as explicit missing-evidence rows instead of silently disappearing.
            if (max_display_year - min_display_year) <= 15:
                display_years = list(range(min_display_year, max_display_year + 1))
    if not display_years:
        return 'I could not extract validated year-grouped numeric values from the retrieved context.'

    lines = [
        '### Deterministic Numeric Extraction',
        '',
        '| Year | Extracted Values | Total | Evidence |',
        '| --- | --- | ---: | --- |',
    ]

    for year in display_years:
        year_rows = values_by_year.get(year, [])
        if not year_rows:
            lines.append(f'| {year} | Missing evidence | N/A | No validated numeric evidence found for this year. |')
            continue
        raw_values = [str(row.get('raw_value', '')).strip() for row in year_rows if str(row.get('raw_value', '')).strip()]
        total = sum(float(row.get('value', 0.0)) for row in year_rows)
        sample = str(year_rows[0].get('evidence_span', '')).strip()
        if len(sample) > 120:
            sample = sample[:117] + '...'
        lines.append(
            f"| {year} | {', '.join(raw_values[:4])} | {total:,.2f} | {sample or 'Evidence available in source snippets.'} |"
        )

    available_values = [
        float(row.get('value', 0.0))
        for year_rows in values_by_year.values()
        for row in year_rows
    ]
    if available_values:
        available_total = sum(available_values)
        lines.extend(['', f'**Grand total (available years):** {available_total:,.2f}'])
    else:
        lines.extend(['', '**Grand total (available years):** N/A (no validated values)'])
    return '\n'.join(lines).strip()


async def _try_structured_value_extraction(
    *,
    question: str,
    classification: QueryClassification,
    response_shape: OutputShape,
    chunks: list[dict],
    db: aiosqlite.Connection,
    trace: object | None,
) -> tuple[str, list[ChatSourceReference], dict[str, object]] | None:
    if not _should_run_structured_extraction(
        question=question,
        classification=classification,
        response_shape=response_shape,
    ):
        return None
    file_ids = sorted({int(chunk.get('file_id', 0)) for chunk in chunks if chunk.get('file_id') is not None})
    if not file_ids:
        return None
    metadata_by_file_id = await _fetch_file_metadata(db, file_ids)
    extracted_rows: list[dict[str, object]] = []
    first_chunk_by_file: dict[int, dict] = {}
    for chunk in chunks:
        raw_file_id = chunk.get('file_id')
        if raw_file_id is None:
            continue
        file_id = int(raw_file_id)
        first_chunk_by_file.setdefault(file_id, chunk)
        text = str(chunk.get('chunk_text', ''))
        if not text.strip():
            continue
        candidates = _extract_candidate_values(
            chunk_text=text,
            field_hint=classification.field_hint,
            base_score=float(chunk.get('score', 0.0)),
        )
        for candidate in candidates[:6]:
            start_idx = int(candidate.get('start', 0))
            end_idx = int(candidate.get('end', 0))
            extracted_rows.append({
                'file_id': file_id,
                'field_label': _infer_field_label(text, start_idx, classification.field_hint),
                'value': float(candidate['value']),
                'raw_value': str(candidate['raw_value']),
                'confidence': float(candidate['confidence']),
                'evidence_span': _build_evidence_span(text, start_idx, end_idx),
                'chunk': chunk,
            })
    if not extracted_rows and not first_chunk_by_file:
        return None
    validated_rows = _validate_structured_rows(extracted_rows)
    if len(validated_rows) < 2:
        return None

    if _is_finance_conflict_prompt(question):
        bullet_limit = _extract_exact_top_level_bullet_limit(question) or 4
        required_years = _extract_required_years(question)
        min_year = min(required_years) if required_years else None
        max_year = max(required_years) if required_years else None
        candidate_rows: list[dict[str, object]] = []
        for row in validated_rows:
            file_id = int(row.get('file_id', 0))
            metadata = metadata_by_file_id.get(file_id, {})
            year_value = metadata.get('year')
            if isinstance(min_year, int) and isinstance(year_value, int) and year_value < min_year:
                continue
            if isinstance(max_year, int) and isinstance(year_value, int) and year_value > max_year:
                continue
            field_label = str(row.get('field_label', 'value') or 'value')
            evidence_span = str(row.get('evidence_span', '') or '')
            category = _normalize_field_category(field_label, evidence_span)
            row_copy = dict(row)
            row_copy['category'] = category
            row_copy['year'] = metadata.get('year')
            row_copy['filename'] = str(metadata.get('filename') or '')
            row_copy['file_path'] = str(metadata.get('file_path') or '')
            candidate_rows.append(row_copy)

        # Keep strongest numeric row per file/category to avoid OCR noise duplicates.
        strongest_rows_by_file_category: dict[tuple[int, str], dict[str, object]] = {}
        for row in candidate_rows:
            file_id = int(row.get('file_id', 0))
            category = str(row.get('category', 'unknown'))
            key = (file_id, category)
            current = strongest_rows_by_file_category.get(key)
            if current is None or float(row.get('confidence', 0.0)) > float(current.get('confidence', 0.0)):
                strongest_rows_by_file_category[key] = row
        strongest_rows = list(strongest_rows_by_file_category.values())
        strongest_rows = [
            row for row in strongest_rows
            if _is_plausible_finance_value(
                str(row.get('raw_value', '') or ''),
                float(row.get('value', 0.0) or 0.0),
            )
        ]

        conflicts: list[dict[str, object]] = []
        for idx, left in enumerate(strongest_rows):
            for right in strongest_rows[idx + 1:]:
                left_file_id = int(left.get('file_id', 0))
                right_file_id = int(right.get('file_id', 0))
                if left_file_id == right_file_id:
                    continue
                left_value = float(left.get('value', 0.0))
                right_value = float(right.get('value', 0.0))
                if abs(left_value - right_value) < 0.01:
                    continue
                left_raw = str(left.get('raw_value', '')).strip()
                right_raw = str(right.get('raw_value', '')).strip()
                if not _is_plausible_finance_value(left_raw, left_value):
                    continue
                if not _is_plausible_finance_value(right_raw, right_value):
                    continue
                left_num = _normalize_numeric_token(left_raw)
                right_num = _normalize_numeric_token(right_raw)
                if not left_num or not right_num or left_num == right_num:
                    continue

                left_category = str(left.get('category', 'unknown'))
                right_category = str(right.get('category', 'unknown'))
                if left_category == 'unknown' and right_category == 'unknown':
                    continue
                same_category = left_category == right_category and left_category != 'unknown'
                require_same_category = bool(
                    getattr(settings, 'extraction_finance_conflict_require_same_category', True)
                )
                if require_same_category and not same_category:
                    continue
                if (not require_same_category) and not same_category and {'interest', 'dividend'}.isdisjoint({left_category, right_category}):
                    # When same-category is disabled, still prefer clearly finance-comparable fields.
                    continue
                # Avoid OCR single-digit noise rows (e.g. "$1") as conflict anchors.
                small_value_threshold = float(
                    getattr(settings, 'extraction_numeric_noise_small_value_threshold', 2.0) or 2.0
                )
                large_value_threshold = float(
                    getattr(settings, 'extraction_numeric_noise_large_value_threshold', 100.0) or 100.0
                )
                if abs(left_value) < small_value_threshold and abs(right_value) > large_value_threshold:
                    continue
                if abs(right_value) < small_value_threshold and abs(left_value) > large_value_threshold:
                    continue

                left_file = str(left.get('filename') or 'Unknown file')
                right_file = str(right.get('filename') or 'Unknown file')
                left_evidence = str(left.get('evidence_span') or '').strip()
                right_evidence = str(right.get('evidence_span') or '').strip()
                evidence_overlap = _evidence_overlap_tokens(left_evidence, right_evidence)
                min_overlap_tokens = int(
                    getattr(settings, 'extraction_finance_conflict_min_evidence_overlap_tokens', 2) or 2
                )
                if evidence_overlap < min_overlap_tokens:
                    continue
                if len(left_evidence) > 90:
                    left_evidence = left_evidence[:87] + '...'
                if len(right_evidence) > 90:
                    right_evidence = right_evidence[:87] + '...'
                category_phrase = left_category.replace('_', ' ') if same_category else 'financial field'

                reason_text = (
                    f'Different amounts are reported for comparable {category_phrase} entries '
                    f'({left_file}: "{left_evidence}"; {right_file}: "{right_evidence}").'
                    if same_category
                    else (
                        f'Not found (direct comparability is limited across differing field types; '
                        f'{left_file}: "{left_evidence}"; {right_file}: "{right_evidence}").'
                    )
                )
                statement = (
                    f'{left_file} and {right_file} conflict on reported {category_phrase} values '
                    f'based on extracted values.'
                )
                values = (
                    f'{left_file}: {left_raw}; {right_file}: {right_raw} '
                    f'("{left_evidence}" | "{right_evidence}").'
                )
                docs = f'{left_file}, {right_file}'
                confidence_score = (
                    abs(left_value - right_value)
                    + float(left.get('confidence', 0.0))
                    + float(right.get('confidence', 0.0))
                    + float(evidence_overlap)
                )
                conflicts.append({
                    'statement': statement,
                    'docs': docs,
                    'values': values,
                    'reason': reason_text,
                    'score': confidence_score,
                    'rows': (left, right),
                })

        conflicts.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
        selected_conflicts = conflicts[:bullet_limit]
        answer = _render_finance_conflict_bullets(
            selected_conflicts=selected_conflicts,
            bullet_limit=bullet_limit,
        )
        used_rows = [row for conflict in selected_conflicts for row in conflict.get('rows', ())]
        source_chunks_by_file: dict[int, dict] = {}
        for row in used_rows:
            if not isinstance(row, dict):
                continue
            file_id = int(row.get('file_id', 0))
            source_chunks_by_file.setdefault(file_id, row.get('chunk', {}))
        if not source_chunks_by_file:
            for chunk in chunks[:4]:
                raw_file_id = chunk.get('file_id')
                if raw_file_id is None:
                    continue
                source_chunks_by_file.setdefault(int(raw_file_id), chunk)
        sources = [
            ChatSourceReference(
                filename=str(chunk.get('filename', 'unknown')),
                path=str(chunk.get('file_path', '')),
                chunk_preview=str(chunk.get('chunk_text', ''))[:200],
                relevance_score=_normalize_relevance_score(chunk.get('score', 0.0)),
            )
            for chunk in source_chunks_by_file.values()
            if isinstance(chunk, dict)
        ]
        metrics = {
            'structured_extraction_applied': True,
            'structured_extraction_values_found': len(strongest_rows),
            'structured_extraction_missing_files': 0,
            'structured_extraction_mode': 'finance_conflict_compare',
            'structured_extraction_conflicts_found': len(selected_conflicts),
            'not_found_count': answer.casefold().count('not found'),
        }
        if trace is not None:
            trace.record('structured_extraction', {
                'applied': True,
                'subtype': classification.subtype,
                'group_by': classification.group_by,
                'field_hint': classification.field_hint,
                'mode': 'finance_conflict_compare',
                'values_found': len(strongest_rows),
                'conflicts_found': len(selected_conflicts),
            })
        return answer, sources, metrics

    selected_rows: list[dict[str, object]]
    required_years = _extract_required_years(question)
    available_years = sorted({
        int(year)
        for year in (metadata.get('year') for metadata in metadata_by_file_id.values())
        if isinstance(year, int)
    })
    is_year_aggregate = (
        classification.subtype == QuerySubtype.AGGREGATE_BY_PERIOD
        and (classification.group_by == GroupBy.YEAR or bool(required_years))
    )
    if is_year_aggregate:
        best_by_file: dict[int, dict[str, object]] = {}
        for row in validated_rows:
            file_id = int(row.get('file_id', 0))
            current = best_by_file.get(file_id)
            if current is None or float(row.get('confidence', 0.0)) > float(current.get('confidence', 0.0)):
                best_by_file[file_id] = row
        selected_rows = list(best_by_file.values())
        if required_years:
            min_required_year = min(required_years)
            max_required_year = max(required_years)
            bounded_rows: list[dict[str, object]] = []
            for row in selected_rows:
                file_id = int(row.get('file_id', 0))
                metadata = metadata_by_file_id.get(file_id, {})
                year_value = metadata.get('year')
                if not isinstance(year_value, int):
                    continue
                if year_value < min_required_year or year_value > max_required_year:
                    continue
                bounded_rows.append(row)
            selected_rows = bounded_rows
        selected_rows.sort(key=lambda item: float(item.get('confidence', 0.0)), reverse=True)
        selected_rows = selected_rows[:36]
        grouped_years = {
            str(_group_label_for_file(group_by=GroupBy.YEAR, metadata=metadata_by_file_id.get(int(row['file_id']), {})))
            for row in selected_rows
        }
        if (
            not required_years
            and len([year for year in grouped_years if year != 'Unknown Year']) < 2
        ):
            return None
    else:
        if classification.group_by == GroupBy.YEAR and classification.year_filter is None:
            grouped_years = {
                str(_group_label_for_file(group_by=GroupBy.YEAR, metadata=metadata_by_file_id.get(int(row['file_id']), {})))
                for row in validated_rows
            }
            if len([year for year in grouped_years if year != 'Unknown Year']) < 2:
                return None
        selected_rows = validated_rows[:24]

    source_chunks_by_file: dict[int, dict] = {}
    for row in selected_rows:
        file_id = int(row['file_id'])
        source_chunks_by_file.setdefault(file_id, row['chunk'])
    if not source_chunks_by_file:
        for chunk in chunks[:4]:
            raw_file_id = chunk.get('file_id')
            if raw_file_id is None:
                continue
            source_chunks_by_file.setdefault(int(raw_file_id), chunk)

    if is_year_aggregate:
        answer = _render_year_aggregate_answer(
            rows=selected_rows,
            metadata_by_file_id=metadata_by_file_id,
            required_years=required_years,
            available_years=available_years,
            field_hint=classification.field_hint,
        )
    else:
        bullet_limit = _extract_exact_top_level_bullet_limit(question)
        if isinstance(bullet_limit, int):
            answer = _render_structured_rows_bullets_answer(
                selected_rows,
                bullet_limit,
                header_labels=_extract_requested_pipe_labels(question),
            )
        else:
            answer = _render_structured_rows_answer(
                selected_rows,
                table_columns=_extract_requested_table_columns(question),
            )
    source_chunks = list(source_chunks_by_file.values())
    missing_file_ids = [file_id for file_id in first_chunk_by_file if file_id not in source_chunks_by_file]
    missing_files = [
        {
            'filename': str(
                metadata_by_file_id.get(file_id, {}).get('filename')
                or first_chunk_by_file[file_id].get('filename')
                or 'unknown'
            ),
            'file_path': str(
                metadata_by_file_id.get(file_id, {}).get('file_path')
                or first_chunk_by_file[file_id].get('file_path')
                or ''
            ),
        }
        for file_id in missing_file_ids
    ]
    sources = [
        ChatSourceReference(
            filename=str(chunk.get('filename', 'unknown')),
            path=str(chunk.get('file_path', '')),
            chunk_preview=str(chunk.get('chunk_text', ''))[:200],
            relevance_score=_normalize_relevance_score(chunk.get('score', 0.0)),
        )
        for chunk in source_chunks
    ]
    metrics = {
        'structured_extraction_applied': True,
        'structured_extraction_values_found': len(selected_rows),
        'structured_extraction_missing_files': len(missing_files),
        'structured_extraction_mode': 'year_aggregate' if is_year_aggregate else 'field_rows',
    }
    if trace is not None:
        trace.record('structured_extraction', {
            'applied': True,
            'subtype': classification.subtype,
            'group_by': classification.group_by,
            'field_hint': classification.field_hint,
            'required_years': required_years,
            'mode': 'year_aggregate' if is_year_aggregate else 'field_rows',
            'values_found': len(selected_rows),
            'missing_files_count': len(missing_files),
            'validated_rows_count': len(validated_rows),
        })
    return answer, sources, metrics
