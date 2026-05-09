from __future__ import annotations

from informity.api.chat_closeout import build_display_blocks


def test_build_display_blocks_returns_empty_for_empty_answer() -> None:
    assert build_display_blocks('') == []


def test_build_display_blocks_extracts_markdown_table() -> None:
    answer = (
        'Permission Matrix\n\n'
        '| Role | Access |\n'
        '| --- | --- |\n'
        '| Admin | Full |\n'
        '| Viewer | Read |\n'
    )
    blocks = build_display_blocks(answer)
    assert len(blocks) == 2
    assert blocks[0] == {'type': 'text', 'markdown': 'Permission Matrix'}
    assert blocks[1] == {
        'type': 'table',
        'columns': ['Role', 'Access'],
        'rows': [
            ['Admin', 'Full'],
            ['Viewer', 'Read'],
        ],
    }


def test_build_display_blocks_extracts_code_fence() -> None:
    answer = (
        'Use this snippet:\n\n'
        '```json\n'
        '{\n'
        '  "ok": true\n'
        '}\n'
        '```\n'
        '\n'
        'Then continue.'
    )
    blocks = build_display_blocks(answer)
    assert len(blocks) == 3
    assert blocks[0] == {'type': 'text', 'markdown': 'Use this snippet:'}
    assert blocks[1] == {
        'type': 'code',
        'code': '{\n  "ok": true\n}',
        'language': 'json',
    }
    assert blocks[2] == {'type': 'text', 'markdown': 'Then continue.'}


def test_build_display_blocks_extracts_nested_checklist() -> None:
    answer = (
        'Checklist:\n\n'
        '- [x] Top done\n'
        '- [ ] Top todo\n'
        '  - Child note\n'
    )
    blocks = build_display_blocks(answer)
    assert len(blocks) == 2
    assert blocks[0] == {'type': 'text', 'markdown': 'Checklist:'}
    assert blocks[1] == {
        'type': 'list',
        'ordered': False,
        'items': [
            {'text': 'Top done', 'level': 0, 'checked': True},
            {'text': 'Top todo', 'level': 0, 'checked': False},
            {'text': 'Child note', 'level': 1, 'checked': None},
        ],
    }


def test_build_display_blocks_extracts_quote_block() -> None:
    answer = (
        'Context first.\n\n'
        '> Quoted line one.\n'
        '> Quoted line two.\n'
        '\n'
        'After quote.'
    )
    blocks = build_display_blocks(answer)
    assert len(blocks) == 3
    assert blocks[0] == {'type': 'text', 'markdown': 'Context first.'}
    assert blocks[1] == {'type': 'quote', 'text': 'Quoted line one.\nQuoted line two.'}
    assert blocks[2] == {'type': 'text', 'markdown': 'After quote.'}
