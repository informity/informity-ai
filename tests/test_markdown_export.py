from __future__ import annotations

from datetime import UTC, datetime

from informity.db.models import ChatMessage
from informity.markdown_export import (
    MarkdownExportOptions,
    build_markdown_filename,
    render_current_answer_markdown,
    render_full_chat_markdown,
)


def test_render_full_chat_markdown_renders_frontmatter_and_deduplicated_sources() -> None:
    fixed_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    output = render_full_chat_markdown(
        chat_title='Export Chat',
        chat_id='chat-markdown',
        messages=[
            ChatMessage(chat_id='chat-markdown', role='user', content='Summarize the upload.'),
            ChatMessage(
                chat_id='chat-markdown',
                role='assistant',
                content='## Result\n\nAll good.',
                sources=[
                    {'filename': 'a.md', 'path': '/tmp/a.md'},
                    {'filename': 'a.md', 'path': '/tmp/a.md'},
                    {'filename': 'b.md', 'path': '/tmp/b.md'},
                ],
            ),
        ],
        chat_mode='researcher',
        options=MarkdownExportOptions(include_frontmatter=True, template='full_transcript', generated_at=fixed_now),
    )
    assert output.startswith('---\n')
    assert 'generated_at: 2026-05-20T12:00:00+00:00' in output
    assert 'export_scope: full_chat' in output
    assert 'template: full_transcript' in output
    assert 'sources_count: 2' in output
    assert '# Export Chat' in output
    assert output.count('`a.md` — `/tmp/a.md`') == 1
    assert output.count('`b.md` — `/tmp/b.md`') == 1


def test_render_current_answer_markdown_is_deterministic() -> None:
    answer = ChatMessage(
        chat_id='chat-1',
        role='assistant',
        content='Here is the answer.',
        sources=[{'filename': 'doc.md', 'path': '/docs/doc.md'}],
    )
    fixed_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    output = render_current_answer_markdown(
        chat_title='Roadmap Chat',
        chat_id='chat-1',
        answer_message=answer,
        chat_mode='researcher',
        options=MarkdownExportOptions(include_frontmatter=True, template='concise_summary', generated_at=fixed_now),
    )
    assert output.startswith('---\n')
    assert 'generated_at: 2026-05-20T12:00:00+00:00' in output
    assert 'export_scope: single_message' in output
    assert 'template: concise_summary' in output
    assert '# Roadmap Chat' in output
    assert '| Field | Value |' not in output
    assert '| Export Scope | Single Message |' in output
    assert '## Answer' in output
    assert '## Sources' in output
    assert 'doc.md' in output


def test_render_full_chat_markdown_supports_concise_summary_template() -> None:
    fixed_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    output = render_full_chat_markdown(
        chat_title='My Chat',
        chat_id='chat-2',
        messages=[
            ChatMessage(chat_id='chat-2', role='user', content='Question'),
            ChatMessage(chat_id='chat-2', role='assistant', content='Answer one.'),
            ChatMessage(chat_id='chat-2', role='assistant', content='Answer two.'),
        ],
        chat_mode='assistant',
        options=MarkdownExportOptions(include_frontmatter=False, template='concise_summary', generated_at=fixed_now),
    )
    assert output.startswith('# My Chat')
    assert '| Export Scope | Full Chat |' in output
    assert '| Template | Concise Summary |' in output
    assert '## Summary' in output
    assert 'Answer one.' in output and 'Answer two.' in output
    assert '## Conversation' not in output


def test_render_full_chat_markdown_separates_user_and_assistant_blocks() -> None:
    fixed_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    output = render_full_chat_markdown(
        chat_title='Chat',
        chat_id='chat-3',
        messages=[
            ChatMessage(chat_id='chat-3', role='user', content='Question'),
            ChatMessage(chat_id='chat-3', role='assistant', content='Answer'),
        ],
        chat_mode='researcher',
        options=MarkdownExportOptions(include_frontmatter=False, template='full_transcript', generated_at=fixed_now),
    )
    assert '## Chat' in output
    assert '\n---\n\n### User\n' in output
    assert '\n---\n\n### Assistant\n' in output


def test_build_markdown_filename_uses_slug_and_timestamp() -> None:
    fixed_now = datetime(2026, 5, 20, 12, 34, 56, tzinfo=UTC)
    filename = build_markdown_filename(chat_title='Hello, Markdown Export!', now=fixed_now)
    assert filename == 'hello-markdown-export-20260520-123456.md'
