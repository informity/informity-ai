import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ChatMessage } from './ChatMessage'
import type { DisplayBlock } from '../../types/api'

describe('ChatMessage markdown rendering', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders nested list structure for markdown lists', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'## Summary\n\n- first item\n- second item\n  - nested item'}
        isStreaming={false}
      />,
    )

    expect(screen.getByRole('heading', { level: 2, name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByText('first item')).toBeInTheDocument()
    expect(screen.getByText('second item')).toBeInTheDocument()
    expect(screen.getByText('nested item')).toBeInTheDocument()
    expect(container.querySelectorAll('ul').length).toBeGreaterThanOrEqual(2)
  })

  it('renders markdown table content and structure', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'| Metric | Value |\n| --- | --- |\n| Accuracy | 99.2% |\n| Count | 1,250 |'}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Metric')).toBeInTheDocument()
    expect(screen.getByText('Value')).toBeInTheDocument()
    expect(screen.getByText('Accuracy')).toBeInTheDocument()
    expect(screen.getByText('99.2%')).toBeInTheDocument()
    expect(screen.getByText('Count')).toBeInTheDocument()
    expect(screen.getByText('1,250')).toBeInTheDocument()
    expect(container.querySelectorAll('tbody tr').length).toBe(2)
  })

  it('renders markdown code block with copy action', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'```python\nfor i in range(3):\n    print(i)\n```'}
        isStreaming={false}
      />,
    )

    expect(container.querySelector('.chat-message__code-wrapper')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Copy code block' })).toBeInTheDocument()
    const codeElement = container.querySelector('pre code')
    expect(codeElement?.textContent).toContain('for i in range(3):')
    expect(codeElement?.textContent).toContain('print(i)')
  })

  it('renders heading and list content uniformly', () => {
    render(
      <ChatMessage
        role="assistant"
        content={'## Summary\n\n- first\n- second'}
        isStreaming={false}
      />,
    )

    expect(screen.getByRole('heading', { level: 2, name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByText('first')).toBeInTheDocument()
    expect(screen.getByText('second')).toBeInTheDocument()
  })

  it('wraps markdown tables in the scroll container', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'| Col A | Col B |\n| --- | --- |\n| v1 | v2 |'}
        isStreaming={false}
      />,
    )

    expect(container.querySelector('.chat-message__table-scroll')).not.toBeNull()
    expect(screen.getByText('Col A')).toBeInTheDocument()
    expect(screen.getByText('v2')).toBeInTheDocument()
  })

  it('right-aligns numeric markdown table cells', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'| Item | Value |\n| --- | --- |\n| Cost | 1,250.50 |\n| Ratio | 17% |'}
        isStreaming={false}
      />,
    )

    expect(container.querySelector('td[data-align="right"]')).not.toBeNull()
  })

  it('renders structured text display blocks when provided', () => {
    render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[{ type: 'text', markdown: '## Block Title\n\nBody from block.' }]}
        isStreaming={false}
      />,
    )

    expect(screen.getByRole('heading', { level: 2, name: 'Block Title' })).toBeInTheDocument()
    expect(screen.getByText('Body from block.')).toBeInTheDocument()
  })

  it('falls back to markdown when display blocks are unknown', () => {
    render(
      <ChatMessage
        role="assistant"
        content={'## Fallback Heading\n\nfallback text'}
        displayBlocks={[{ type: 'unknown_custom', payload: 'x' }]}
        isStreaming={false}
      />,
    )

    expect(screen.getByRole('heading', { level: 2, name: 'Fallback Heading' })).toBeInTheDocument()
    expect(screen.getByText('fallback text')).toBeInTheDocument()
  })

  it('falls back to markdown when structured table block is malformed', () => {
    const malformedTableBlock = {
      type: 'table',
      columns: 'bad',
      rows: [],
    } as unknown as DisplayBlock

    render(
      <ChatMessage
        role="assistant"
        content={'| A | B |\n| --- | --- |\n| 1 | 2 |'}
        displayBlocks={[malformedTableBlock]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('renders structured callout and metric blocks', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          { type: 'callout', tone: 'warning', text: 'Check this before proceeding.' },
          { type: 'metric', label: 'Latency', value: '142 ms' },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Check this before proceeding.')).toBeInTheDocument()
    expect(screen.getByText('Latency')).toBeInTheDocument()
    expect(screen.getByText('142 ms')).toBeInTheDocument()
    expect(container.querySelector('.chat-message__block--warning')).not.toBeNull()
    expect(container.querySelector('.chat-message__block--metric')).not.toBeNull()
  })

  it('renders disclaimer callouts as standardized header cards', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          {
            type: 'callout',
            tone: 'info',
            text: 'Disclaimer: Informity AI is not a lawyer and this is not legal advice.',
          },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Disclaimer')).toBeInTheDocument()
    expect(screen.getByText('Informity AI is not a lawyer and this is not legal advice.')).toBeInTheDocument()
    expect(container.querySelector('.chat-message__disclaimer-card')).not.toBeNull()
    expect(container.querySelector('.chat-message__disclaimer-body')).not.toBeNull()
  })

  it('renders structured code blocks with copy action wrapper', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          { type: 'code', language: 'ts', code: 'const answer: number = 42;' },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByRole('button', { name: 'Copy code block' })).toBeInTheDocument()
    expect(screen.getByText('TypeScript')).toBeInTheDocument()
    expect(container.querySelector('.chat-message__block--code .chat-message__code-wrapper')).not.toBeNull()
    const codeElement = container.querySelector('.chat-message__block--code pre code')
    expect(codeElement?.textContent).toContain('const answer: number = 42;')
    expect(codeElement?.className).toContain('language-ts')
  })

  it('renders structured list/checklist blocks with nesting', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          {
            type: 'list',
            ordered: false,
            items: [
              { text: 'Top done', level: 0, checked: true },
              { text: 'Top todo', level: 0, checked: false },
              { text: 'Child', level: 1, checked: null },
            ],
          },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Top done')).toBeInTheDocument()
    expect(screen.getByText('Top todo')).toBeInTheDocument()
    expect(screen.getByText('Child')).toBeInTheDocument()
    expect(container.querySelectorAll('.chat-message__block--list ul li').length).toBe(3)
    expect(container.querySelectorAll('.chat-message__block--list input[type="checkbox"]').length).toBe(2)
  })

  it('renders inline markdown formatting inside structured list items', () => {
    render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          {
            type: 'list',
            ordered: false,
            items: [
              { text: '**Important:** check this', level: 0, checked: null },
            ],
          },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Important:')).toBeInTheDocument()
    expect(document.querySelector('.chat-message__block--list strong')).not.toBeNull()
  })

  it('renders structured quote blocks', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          { type: 'quote', text: 'Trust, but verify.', attribution: 'Security maxim' },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Trust, but verify.')).toBeInTheDocument()
    expect(screen.getByText('Security maxim')).toBeInTheDocument()
    expect(container.querySelector('.chat-message__block--quote')).not.toBeNull()
  })

  it('renders inline markdown formatting inside structured table cells', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'fallback'}
        displayBlocks={[
          {
            type: 'table',
            columns: ['Risk', 'Role'],
            rows: [['**Data Exfiltration**', 'Analyst']],
          },
        ]}
        isStreaming={false}
      />,
    )

    expect(screen.getByText('Data Exfiltration')).toBeInTheDocument()
    expect(container.querySelector('.chat-message__table-scroll td strong')).not.toBeNull()
  })

  it('renders readable draft text while streaming instead of markdown structure', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'## Live summary\n\n- first item\n- second item'}
        isStreaming={true}
      />,
    )

    expect(screen.getByText('## Live summary', { exact: false })).toBeInTheDocument()
    expect(container.querySelector('h2')).toBeNull()
    expect(container.querySelector('ul')).toBeNull()
    expect(container.querySelector('.chat-message__cursor')).not.toBeNull()
  })

  it('does not render syntax-highlighted code block wrappers while streaming', () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={'```python\nprint("streaming")\n```'}
        isStreaming={true}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Copy code block' })).not.toBeInTheDocument()
    expect(container.querySelector('.chat-message__code-wrapper')).toBeNull()
    expect(screen.getByText('```python', { exact: false })).toBeInTheDocument()
  })

  it('re-renders when stream section progress updates', () => {
    const { rerender } = render(
      <ChatMessage
        role="assistant"
        content=""
        isStreaming={true}
        streamStatusText="Generating response..."
        streamSectionProgress={{
          completed: ['## Executive Summary'],
          remaining: ['## Risks and Gaps'],
          total: 2,
        }}
      />,
    )

    expect(screen.getByText(/\u2713 Executive Summary/)).toBeInTheDocument()
    expect(screen.getByText(/\| Risks and Gaps/)).toBeInTheDocument()

    rerender(
      <ChatMessage
        role="assistant"
        content=""
        isStreaming={true}
        streamStatusText="Generating response..."
        streamSectionProgress={{
          completed: ['## Executive Summary', '## Risks and Gaps'],
          remaining: ['## Action Checklist'],
          total: 3,
        }}
      />,
    )

    expect(screen.getByText(/\u2713 Executive Summary · Risks and Gaps/)).toBeInTheDocument()
    expect(screen.getByText(/\| Action Checklist/)).toBeInTheDocument()
  })
})
