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


def test_build_display_blocks_coerces_unfenced_code_only_answer() -> None:
    answer = (
        'function sum(a: number, b: number): number {\n'
        '  return a + b;\n'
        '}\n'
        '\n'
        'const value = sum(2, 3);\n'
        'console.log(value);\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {
            'type': 'code',
            'code': (
                'function sum(a: number, b: number): number {\n'
                '  return a + b;\n'
                '}\n'
                '\n'
                'const value = sum(2, 3);\n'
                'console.log(value);'
            ),
            'language': 'typescript',
        },
    ]


def test_build_display_blocks_keeps_plain_prose_as_text_block() -> None:
    answer = (
        'We should implement this in two phases.\n'
        'First, we standardize format selection.\n'
        'Then, we validate outputs in tests.\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {
            'type': 'text',
            'markdown': (
                'We should implement this in two phases.\n'
                'First, we standardize format selection.\n'
                'Then, we validate outputs in tests.'
            ),
        },
    ]


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


def test_build_display_blocks_keeps_indented_list_continuations_out_of_code_blocks() -> None:
    answer = (
        '1. **Refinancing is usually superior if rates drop significantly.**\n'
        '    If you can refinance to a shorter term at a lower rate, this is typically cheaper.\n'
        '    * Risk: closing costs can erase short-term savings.\n'
        '2. **Extra payments are superior if rates are stable or rising.**\n'
        '    If rates are not favorable, extra principal reduces interest without refinance fees.\n'
    )
    blocks = build_display_blocks(answer)
    assert any(block.get('type') == 'list' for block in blocks)
    assert not any(block.get('type') == 'code' for block in blocks)
    list_block = next(block for block in blocks if block.get('type') == 'list')
    assert 'If you can refinance to a shorter term' in list_block['items'][0]['text']
    assert any(
        'Risk: closing costs can erase short-term savings.' in item['text']
        for item in list_block['items']
    )


def test_build_display_blocks_normalizes_disclaimer_with_consistent_callout() -> None:
    answer = (
        'Findings summary paragraph.\n'
        '\n'
        '---\n'
        'Disclaimer: Informity AI is not a lawyer and this is not legal advice.\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {'type': 'text', 'markdown': 'Findings summary paragraph.'},
        {
            'type': 'callout',
            'tone': 'info',
            'text': 'Disclaimer: Informity AI is not a lawyer and this is not legal advice.',
        },
    ]


def test_build_display_blocks_normalizes_bold_disclaimer_with_continuation_line() -> None:
    answer = (
        '**Disclaimer:** Informity AI is not a compliance auditor.\n'
        'This is not a formal compliance attestation.\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {
            'type': 'callout',
            'tone': 'info',
            'text': (
                'Disclaimer: Informity AI is not a compliance auditor. '
                'This is not a formal compliance attestation.'
            ),
        },
    ]


def test_build_display_blocks_normalizes_heading_disclaimer_with_body_on_next_line() -> None:
    answer = (
        '### Disclaimer:\n'
        'Informity AI is not a lawyer and this is not legal advice.\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {
            'type': 'callout',
            'tone': 'info',
            'text': 'Disclaimer: Informity AI is not a lawyer and this is not legal advice.',
        },
    ]


def test_build_display_blocks_normalizes_plain_disclaimer_label_with_next_line_body() -> None:
    answer = (
        'Disclaimer:\n'
        'Informity AI is not a lawyer and this is not legal advice.\n'
    )
    blocks = build_display_blocks(answer)
    assert blocks == [
        {
            'type': 'callout',
            'tone': 'info',
            'text': 'Disclaimer: Informity AI is not a lawyer and this is not legal advice.',
        },
    ]
