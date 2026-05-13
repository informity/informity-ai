# ==============================================================================
# Informity AI — Chat Closeout Helpers
# Done-payload assembly and closeout-only numeric evidence helpers.
# ==============================================================================

from __future__ import annotations

import re

_TABLE_SEPARATOR_RE = re.compile(r'^\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?$')
_CODE_FENCE_OPEN_RE = re.compile(r'^```(?P<lang>[A-Za-z0-9_+\-]*)\s*$')
_LIST_ITEM_RE = re.compile(r'^(?P<indent>\s*)(?P<marker>(?:[-*+])|(?:\d+[.)]))\s+(?P<body>.+)$')
_CHECKBOX_RE = re.compile(r'^\[(?P<state>[xX ])\]\s+(?P<text>.+)$')
_QUOTE_LINE_RE = re.compile(r'^\s*>\s?(?P<body>.*)$')
_DISCLAIMER_LINE_RE = re.compile(
    r'^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*Disclaimer\s*:\s*(?P<body>.*?)\s*(?:\*\*)?\s*$',
    re.IGNORECASE,
)
_HORIZONTAL_RULE_RE = re.compile(r'^\s*(?:-{3,}|\*{3,}|_{3,})\s*$')
_UNFENCED_CODE_LINE_HINT_RE = re.compile(
    r'^\s*(?:'
    r'(?:const|let|var|function|class|interface|type|enum|import|export|from|if|else|for|while|switch|case|return|try|catch|finally|throw|new)\b'
    r'|(?:def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|lambda)\b'
    r'|(?:public|private|protected|static|final|void)\b'
    r'|(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b'
    r'|(?:#include|package|func|fn)\b'
    r')',
)
_UNFENCED_CODE_INLINE_HINT_RE = re.compile(r'[{}()[\];]|=>|::|:=|==|!=|<=|>=|&&|\|\|')
_UNFENCED_CODE_MARKDOWN_BLOCKER_RE = re.compile(r'^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+|\|)')
_UNFENCED_CODE_LANGUAGE_HINTS: list[tuple[str, str]] = [
    (r'^\s*(?:def\b|import\b|from\b|class\b|elif\b|except\b|lambda\b)', 'python'),
    (r'^\s*(?:const\b|let\b|var\b|function\b|interface\b|type\b|import\b|export\b)', 'typescript'),
    (r'^\s*(?:public\b|private\b|protected\b|class\b|interface\b)', 'java'),
    (r'^\s*(?:SELECT\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\b|ALTER\b|DROP\b|WITH\b)', 'sql'),
]


def _split_table_cells(line: str) -> list[str]:
    text = line.strip()
    if text.startswith('|'):
        text = text[1:]
    if text.endswith('|'):
        text = text[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '|':
            cells.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    cells.append(''.join(current).strip())
    return cells


def _flush_text(lines: list[str], blocks: list[dict[str, object]]) -> None:
    if not lines:
        return
    markdown = '\n'.join(lines).strip('\n')
    if markdown:
        blocks.append({'type': 'text', 'markdown': markdown})
    lines.clear()


def _trim_trailing_divider(lines: list[str]) -> None:
    while lines:
        candidate = lines[-1].strip()
        if not candidate:
            lines.pop()
            continue
        if _HORIZONTAL_RULE_RE.match(candidate):
            lines.pop()
            continue
        break


def _infer_unfenced_code_language(lines: list[str]) -> str | None:
    for pattern, language in _UNFENCED_CODE_LANGUAGE_HINTS:
        regex = re.compile(pattern, re.IGNORECASE)
        if any(regex.search(line) for line in lines):
            return language
    return None


def _looks_like_unfenced_code_block(answer: str) -> bool:
    raw_lines = answer.splitlines()
    non_empty = [line for line in raw_lines if line.strip()]
    if len(non_empty) < 3:
        return False
    if any(_UNFENCED_CODE_MARKDOWN_BLOCKER_RE.match(line) for line in non_empty):
        return False
    if any(_TABLE_SEPARATOR_RE.match(line.strip()) for line in non_empty):
        return False

    hint_lines = 0
    symbol_lines = 0
    prose_like_lines = 0
    for line in non_empty:
        stripped = line.strip()
        if _UNFENCED_CODE_LINE_HINT_RE.search(stripped):
            hint_lines += 1
        if _UNFENCED_CODE_INLINE_HINT_RE.search(stripped):
            symbol_lines += 1
        if stripped.endswith(('.', '?', '!')) and not _UNFENCED_CODE_INLINE_HINT_RE.search(stripped):
            prose_like_lines += 1

    # Conservative threshold to avoid coercing normal prose.
    return hint_lines >= 2 and symbol_lines >= 2 and prose_like_lines <= max(1, len(non_empty) // 4)


def build_display_blocks(cleaned_answer: str) -> list[dict[str, object]]:
    if not cleaned_answer:
        return []
    if '```' not in cleaned_answer and _looks_like_unfenced_code_block(cleaned_answer):
        code_lines = cleaned_answer.splitlines()
        return [{
            'type': 'code',
            'code': '\n'.join(code_lines).strip('\n'),
            'language': _infer_unfenced_code_language(code_lines),
        }]

    blocks: list[dict[str, object]] = []
    lines = cleaned_answer.splitlines()
    text_buffer: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        code_open = _CODE_FENCE_OPEN_RE.match(line.strip())
        if code_open:
            _flush_text(text_buffer, blocks)
            lang = code_open.group('lang') or None
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip().startswith('```'):
                i += 1
            blocks.append({
                'type': 'code',
                'code': '\n'.join(code_lines),
                'language': lang,
            })
            continue

        has_next = i + 1 < len(lines)
        if (
            has_next
            and '|' in line
            and _TABLE_SEPARATOR_RE.match(lines[i + 1].strip())
        ):
            header_cells = _split_table_cells(line)
            if header_cells:
                _flush_text(text_buffer, blocks)
                i += 2
                table_rows: list[list[str | int | float | None]] = []
                while i < len(lines):
                    row_line = lines[i]
                    if not row_line.strip() or '|' not in row_line:
                        break
                    row_cells = _split_table_cells(row_line)
                    if not row_cells:
                        break
                    if len(row_cells) < len(header_cells):
                        row_cells.extend([''] * (len(header_cells) - len(row_cells)))
                    table_rows.append(row_cells[:len(header_cells)])
                    i += 1
                blocks.append({
                    'type': 'table',
                    'columns': header_cells,
                    'rows': table_rows,
                })
                continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            _flush_text(text_buffer, blocks)
            marker = list_match.group('marker')
            ordered = marker[0].isdigit()
            items: list[dict[str, object]] = []
            while i < len(lines):
                candidate = lines[i]
                candidate_match = _LIST_ITEM_RE.match(candidate)
                if not candidate_match:
                    if not candidate.strip():
                        i += 1
                        continue
                    break
                candidate_marker = candidate_match.group('marker')
                candidate_ordered = candidate_marker[0].isdigit()
                indent_spaces = len(candidate_match.group('indent') or '')
                # Allow nested list markers under an ordered list item (e.g. "1. ...", then "    * ...")
                # by keeping them as part of the same logical list block.
                if candidate_ordered != ordered and indent_spaces == 0:
                    break
                body = candidate_match.group('body').strip()
                # Treat 2-space indentation as one nested list level.
                level = max(0, indent_spaces // 2)
                checked: bool | None = None
                checkbox_match = _CHECKBOX_RE.match(body)
                if checkbox_match:
                    checked = checkbox_match.group('state').strip().lower() == 'x'
                    body = checkbox_match.group('text').strip()
                items.append({
                    'text': body,
                    'level': level,
                    'checked': checked,
                })
                i += 1
                # Consume indented continuation lines as part of the current list item.
                while i < len(lines):
                    continuation_line = lines[i]
                    if not continuation_line.strip():
                        i += 1
                        continue
                    continuation_indent = len(continuation_line) - len(continuation_line.lstrip(' '))
                    if continuation_indent == 0:
                        break
                    next_list = _LIST_ITEM_RE.match(continuation_line)
                    if next_list is not None:
                        # Nested list items are handled by the main list loop.
                        break
                    items[-1]['text'] = f"{items[-1]['text']}\n{continuation_line.strip()}"
                    i += 1
            if items:
                blocks.append({
                    'type': 'list',
                    'ordered': ordered,
                    'items': items,
                })
                continue

        quote_match = _QUOTE_LINE_RE.match(line)
        if quote_match:
            _flush_text(text_buffer, blocks)
            quote_lines: list[str] = []
            while i < len(lines):
                quote_candidate = _QUOTE_LINE_RE.match(lines[i])
                if not quote_candidate:
                    if not lines[i].strip():
                        i += 1
                        continue
                    break
                quote_lines.append(quote_candidate.group('body'))
                i += 1
            quote_text = '\n'.join(quote_lines).strip()
            if quote_text:
                blocks.append({
                    'type': 'quote',
                    'text': quote_text,
                })
                continue

        disclaimer_match = _DISCLAIMER_LINE_RE.match(line)
        if disclaimer_match:
            _trim_trailing_divider(text_buffer)
            _flush_text(text_buffer, blocks)
            disclaimer_body = disclaimer_match.group('body').strip()
            disclaimer_lines: list[str] = []
            if disclaimer_body:
                if disclaimer_body.startswith('**'):
                    disclaimer_body = disclaimer_body[2:].lstrip()
                if disclaimer_body.endswith('**'):
                    disclaimer_body = disclaimer_body[:-2].rstrip()
                disclaimer_lines.append(disclaimer_body)
            i += 1
            while i < len(lines):
                continuation = lines[i].strip()
                if not continuation:
                    break
                if _LIST_ITEM_RE.match(lines[i]) or _QUOTE_LINE_RE.match(lines[i]) or _CODE_FENCE_OPEN_RE.match(lines[i].strip()):
                    break
                if _HORIZONTAL_RULE_RE.match(continuation):
                    i += 1
                    continue
                disclaimer_lines.append(continuation)
                i += 1
            if not disclaimer_lines:
                continue
            blocks.append({
                'type': 'callout',
                'tone': 'info',
                'text': f"Disclaimer: {' '.join(disclaimer_lines).strip()}",
            })
            continue

        text_buffer.append(line)
        i += 1

    _flush_text(text_buffer, blocks)
    return blocks


def build_done_payload(
    *,
    elapsed_seconds: float | None,
    request_id: str | None,
    chat_mode: str,
    timeout_occurred: bool,
    timeout_reason: str | object | None,
    completion_mode: str | object,
    has_remaining_scope: bool,
    stopped_by_user: bool,
    next_action: str | object,
    next_action_reason: str | None,
    sources_count: int,
    message_persisted: bool,
    cleaned_answer: str,
    budget_metrics: dict[str, object],
    budget_checkpoints: list[dict[str, object]],
    continuation_passes: int,
    continuation_resolution_reason: str | object | None,
    continuation_progress_state: str | None,
    pass_details: list[dict[str, object]],
    status_transitions: list[dict[str, object]],
    resource_metrics: dict[str, object],
    message_id: int | None,
) -> dict:
    payload: dict[str, object] = {
        'elapsed_seconds': elapsed_seconds,
        'request_id': request_id if request_id else None,
        'chat_mode': chat_mode,
        'timeout_occurred': timeout_occurred,
        'timeout_reason': timeout_reason,
        'completion_mode': completion_mode,
        'has_remaining_scope': has_remaining_scope,
        'stopped_by_user': stopped_by_user,
        'next_action': next_action,
        'next_action_reason': next_action_reason,
        'sources_count': sources_count,
        'message_persisted': message_persisted,
        'display_blocks': build_display_blocks(cleaned_answer),
        'budget_metrics': budget_metrics,
        'web_search_used': bool(budget_metrics.get('web_search_used')),
        'budget_checkpoints': budget_checkpoints,
        'continuation_passes': continuation_passes,
        'continuation_resolution_reason': continuation_resolution_reason,
        'continuation_progress_state': continuation_progress_state,
        'pass_details': pass_details,
        'status_transitions': status_transitions,
        'resource_metrics': resource_metrics,
    }
    if message_id is not None:
        payload['message_id'] = message_id
    return payload


__all__ = [
    'build_display_blocks',
    'build_done_payload',
]
