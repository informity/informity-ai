from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from informity.answer_sanitization import build_display_answer
from informity.api.chat_closeout import build_display_blocks
from informity.db.models import ChatMessage
from informity.llm.types import ChatRole

_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')
_MULTI_DASH_RE = re.compile(r'-+')


@dataclass(frozen=True)
class MarkdownExportOptions:
    include_frontmatter: bool = False
    template: str = 'full_transcript'  # full_transcript | concise_summary
    generated_at: datetime | None = None


def build_markdown_filename(*, chat_title: str | None, now: datetime | None = None) -> str:
    ts = (now or datetime.now(UTC)).astimezone(UTC).strftime('%Y%m%d-%H%M%S')
    title = _slugify(chat_title or 'chat')
    return f'{title}-{ts}.md'


def render_current_answer_markdown(
    *,
    chat_title: str | None,
    chat_id: str,
    answer_message: ChatMessage,
    chat_mode: str | None,
    options: MarkdownExportOptions | None = None,
) -> str:
    opts = options or MarkdownExportOptions()
    generated_at = (opts.generated_at or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    title = _safe_title(chat_title)
    sections: list[str] = []
    if opts.include_frontmatter:
        sections.append(_render_frontmatter(
            generated_at=generated_at,
            chat_id=chat_id,
            mode=chat_mode,
            sources_count=len(answer_message.sources or []),
            scope='single_message',
            template=opts.template,
        ))
    sections.extend([
        f'# {title}',
        '',
        _render_metadata_table(
            generated_at=generated_at,
            chat_id=chat_id,
            mode=chat_mode,
            sources_count=len(answer_message.sources or []),
            scope='single_message',
            template=opts.template,
        ),
        '',
        '## Answer',
        _message_markdown(answer_message),
    ])
    sources = _render_sources(answer_message.sources or [])
    if sources:
        sections.extend(['', sources])
    return '\n'.join(sections).rstrip() + '\n'


def render_full_chat_markdown(
    *,
    chat_title: str | None,
    chat_id: str,
    messages: list[ChatMessage],
    chat_mode: str | None,
    options: MarkdownExportOptions | None = None,
) -> str:
    opts = options or MarkdownExportOptions()
    generated_at = (opts.generated_at or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    title = _safe_title(chat_title)
    normalized_messages = [m for m in messages if not (m.role == ChatRole.USER and bool(m.is_internal))]
    all_sources: list[dict] = []
    for message in normalized_messages:
        if message.role == ChatRole.ASSISTANT and message.sources:
            all_sources.extend(message.sources)

    sections: list[str] = []
    if opts.include_frontmatter:
        sections.append(_render_frontmatter(
            generated_at=generated_at,
            chat_id=chat_id,
            mode=chat_mode,
            sources_count=len(_normalize_sources(all_sources)),
            scope='full_chat',
            template=opts.template,
        ))
    sections.extend([
        f'# {title}',
        '',
        _render_metadata_table(
            generated_at=generated_at,
            chat_id=chat_id,
            mode=chat_mode,
            sources_count=len(_normalize_sources(all_sources)),
            scope='full_chat',
            template=opts.template,
        ),
    ])

    if opts.template == 'concise_summary':
        sections.extend(['', '## Summary'])
        assistant_blocks = [
            _message_markdown(message).strip()
            for message in normalized_messages
            if message.role == ChatRole.ASSISTANT
        ]
        concise = '\n\n'.join(block for block in assistant_blocks if block).strip()
        sections.append(concise or '_No assistant responses._')
    else:
        sections.extend(['', '## Chat'])
        for message in normalized_messages:
            role_label = 'User' if message.role == ChatRole.USER else 'Assistant'
            sections.extend([
                '',
                '---',
                '',
                f'### {role_label}',
                _message_markdown(message),
            ])

    sources = _render_sources(all_sources)
    if sources:
        sections.extend(['', sources])
    return '\n'.join(sections).rstrip() + '\n'


def _safe_title(value: str | None) -> str:
    return str(value or '').strip() or 'Chat Export'


def _slugify(value: str) -> str:
    lowered = str(value or '').strip().lower()
    normalized = _NON_ALNUM_RE.sub('-', lowered)
    return _MULTI_DASH_RE.sub('-', normalized).strip('-') or 'chat'


def _render_frontmatter(
    *,
    generated_at: datetime,
    chat_id: str,
    mode: str | None,
    sources_count: int,
    scope: str,
    template: str,
) -> str:
    lines = [
        '---',
        f'generated_at: {generated_at.isoformat()}',
        f'chat_id: {chat_id}',
        f'export_scope: {scope}',
        f'template: {template}',
        f'mode: {str(mode or "").strip() or "unknown"}',
        f'sources_count: {max(0, int(sources_count))}',
        '---',
        '',
    ]
    return '\n'.join(lines)


def _render_metadata_table(
    *,
    generated_at: datetime,
    chat_id: str,
    mode: str | None,
    sources_count: int,
    scope: str,
    template: str,
) -> str:
    resolved_mode = str(mode or '').strip() or 'Unknown'
    scope_label = 'Single Message' if scope == 'single_message' else 'Full Chat'
    template_label = 'Concise Summary' if template == 'concise_summary' else 'Full Transcript'
    return '\n'.join([
        '|  |  |',
        '| --- | --- |',
        f'| Export Scope | {scope_label} |',
        f'| Template | {template_label} |',
        f'| Mode | {resolved_mode} |',
        f'| Sources Count | {max(0, int(sources_count))} |',
        f'| Generated At | {generated_at.isoformat()} |',
        f'| Chat ID | `{chat_id}` |',
    ])


def _message_markdown(message: ChatMessage) -> str:
    if message.role == ChatRole.USER:
        return str(message.content or '').strip() or '_No content._'

    cleaned, _ = build_display_answer(str(message.content or ''), preserve_task_checkboxes=False)
    blocks = build_display_blocks(cleaned)
    if not blocks:
        return cleaned.strip() or '_No content._'

    rendered_parts: list[str] = []
    for block in blocks:
        rendered = _render_block(block)
        if rendered:
            rendered_parts.append(rendered)
    return '\n\n'.join(rendered_parts).strip() or '_No content._'


def _render_block(block: dict[str, object]) -> str:
    block_type = str(block.get('type') or '').strip().lower()
    if block_type == 'text':
        return str(block.get('markdown') or '').strip()
    if block_type == 'code':
        language = str(block.get('language') or '').strip()
        code = str(block.get('code') or '')
        return f'```{language}\n{code}\n```'.strip()
    if block_type == 'table':
        columns = [str(col) for col in (block.get('columns') or [])]
        rows = block.get('rows') or []
        return _render_markdown_table(columns=columns, rows=rows)
    if block_type == 'list':
        ordered = bool(block.get('ordered'))
        items = block.get('items') or []
        lines: list[str] = []
        for index, raw_item in enumerate(items, start=1):
            item = raw_item if isinstance(raw_item, dict) else {}
            text = str(item.get('text') or '').strip()
            if not text:
                continue
            checked = item.get('checked')
            level = max(0, int(item.get('level') or 0))
            prefix = f'{index}. ' if ordered else '- '
            if checked is True:
                prefix = '- [x] '
            elif checked is False:
                prefix = '- [ ] '
            indent = '  ' * level
            lines.append(f'{indent}{prefix}{text}')
        return '\n'.join(lines).strip()
    if block_type == 'quote':
        text = str(block.get('text') or '').strip()
        attribution = str(block.get('attribution') or '').strip()
        if not text:
            return ''
        if attribution:
            return f'> {text}\n>\n> — {attribution}'
        return f'> {text}'
    if block_type == 'callout':
        text = str(block.get('text') or '').strip()
        tone = str(block.get('tone') or 'info').strip().upper()
        return f'> [{tone}] {text}'.strip()
    if block_type == 'metric':
        label = str(block.get('label') or '').strip()
        value = str(block.get('value') or '').strip()
        if label and value:
            return f'- **{label}:** {value}'
        return ''
    return ''


def _render_markdown_table(*, columns: list[str], rows: object) -> str:
    if not columns:
        return ''
    header = '| ' + ' | '.join(_escape_pipe(col) for col in columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in columns) + ' |'
    row_lines: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, list):
                continue
            cells = [str(cell or '') for cell in row[: len(columns)]]
            if len(cells) < len(columns):
                cells.extend([''] * (len(columns) - len(cells)))
            row_lines.append('| ' + ' | '.join(_escape_pipe(cell) for cell in cells) + ' |')
    return '\n'.join([header, separator, *row_lines]).strip()


def _escape_pipe(value: str) -> str:
    return str(value or '').replace('|', '\\|')


def _normalize_sources(sources: list[dict]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        filename = str(source.get('filename') or '').strip()
        path = str(source.get('path') or '').strip()
        key = (filename, path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    normalized.sort(key=lambda item: (item[0].lower(), item[1].lower()))
    return normalized


def _render_sources(sources: list[dict]) -> str:
    normalized = _normalize_sources(sources)
    if not normalized:
        return ''
    lines = ['## Sources']
    for filename, path in normalized:
        if filename and path:
            lines.append(f'- `{filename}` — `{path}`')
        elif filename:
            lines.append(f'- `{filename}`')
        elif path:
            lines.append(f'- `{path}`')
    return '\n'.join(lines)
