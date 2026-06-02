import { describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach } from 'vitest'
import { MarkdownContent } from './MarkdownContent'

afterEach(() => cleanup())

// ──────────────────────────────────────────────────────────────────────────────
// Table rendering — core regression guard
// ──────────────────────────────────────────────────────────────────────────────

describe('MarkdownContent — table rendering', () => {
  it('renders a well-formed GFM table as an HTML <table> element', () => {
    const { container } = render(
      <MarkdownContent>{'| Name | Value |\n|---|---|\n| Alpha | 1 |'}</MarkdownContent>
    )
    expect(container.querySelector('table')).not.toBeNull()
    expect(container.querySelector('th')).not.toBeNull()
    expect(container.querySelector('td')).not.toBeNull()
    expect(screen.getByText('Name')).toBeInTheDocument()
    expect(screen.getByText('Alpha')).toBeInTheDocument()
  })

  it('wraps the table in a scroll container', () => {
    const { container } = render(
      <MarkdownContent>{'| A | B |\n|---|---|\n| x | y |'}</MarkdownContent>
    )
    const scrollDiv = container.querySelector('.md-content__table-scroll')
    expect(scrollDiv).not.toBeNull()
    expect(scrollDiv?.querySelector('table')).not.toBeNull()
  })

  it('does NOT render raw pipe characters for a well-formed table', () => {
    const { container } = render(
      <MarkdownContent>{'| Col A | Col B |\n|---|---|\n| val 1 | val 2 |'}</MarkdownContent>
    )
    // The table should be rendered as HTML, not raw markdown text with pipes
    expect(container.textContent).not.toContain('|')
  })

  it('removes orphaned separator rows so they do not render as literal dashes', () => {
    // This is the CISA document scenario from the screenshots
    const problematic = [
      '**Título del ejercicio**',
      '',
      '|---|---|---|',
      '|---|---|---|',
      '|---|---|---|',
      '',
      '**Descripción**',
    ].join('\n')

    const { container } = render(<MarkdownContent>{problematic}</MarkdownContent>)
    // Should NOT render the dashes as paragraph text
    expect(container.textContent).not.toContain('|---|---|---|')
    expect(screen.getByText(/Título del ejercicio/)).toBeInTheDocument()
    expect(screen.getByText(/Descripción/)).toBeInTheDocument()
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Standard markdown elements
// ──────────────────────────────────────────────────────────────────────────────

describe('MarkdownContent — standard elements', () => {
  it('renders headings as heading elements', () => {
    const { container } = render(<MarkdownContent>{'# Heading 1\n\n## Heading 2'}</MarkdownContent>)
    expect(container.querySelector('h1')).not.toBeNull()
    expect(container.querySelector('h2')).not.toBeNull()
  })

  it('renders bold text as <strong>', () => {
    const { container } = render(<MarkdownContent>{'**bold text**'}</MarkdownContent>)
    expect(container.querySelector('strong')).not.toBeNull()
    expect(screen.getByText('bold text')).toBeInTheDocument()
  })

  it('renders bullet lists as <ul><li>', () => {
    const { container } = render(<MarkdownContent>{'- Item one\n- Item two'}</MarkdownContent>)
    expect(container.querySelector('ul')).not.toBeNull()
    expect(container.querySelectorAll('li').length).toBe(2)
  })

  it('renders numbered lists as <ol><li>', () => {
    const { container } = render(<MarkdownContent>{'1. First\n2. Second'}</MarkdownContent>)
    expect(container.querySelector('ol')).not.toBeNull()
  })

  it('renders inline code as <code>', () => {
    const { container } = render(<MarkdownContent>{'Use `npm install` here.'}</MarkdownContent>)
    expect(container.querySelector('code')).not.toBeNull()
    expect(screen.getByText('npm install')).toBeInTheDocument()
  })

  it('renders links as <a>', () => {
    const { container } = render(
      <MarkdownContent>{'[Click here](https://example.com)'}</MarkdownContent>
    )
    expect(container.querySelector('a')).not.toBeNull()
    expect(screen.getByText('Click here')).toBeInTheDocument()
  })

  it('strips HTML tags from input before rendering', () => {
    const { container } = render(
      <MarkdownContent>{'<b>Bold via HTML</b> and plain text'}</MarkdownContent>
    )
    // HTML tags should be stripped by preprocessMarkdown
    expect(container.querySelector('b')).toBeNull()
    expect(screen.getByText(/Bold via HTML/)).toBeInTheDocument()
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Translation section rendering — integration scenario
// ──────────────────────────────────────────────────────────────────────────────

describe('MarkdownContent — translation section scenarios', () => {
  it('renders a translated section with mixed content cleanly', () => {
    const section = [
      '## Objetivos del Ejercicio',
      '',
      'Este ejercicio evalúa la **coordinación operativa** entre organizaciones.',
      '',
      '- Comunicación Operativa',
      '- Coordinación entre agencias',
      '- Planificación de respuesta',
    ].join('\n')

    const { container } = render(<MarkdownContent>{section}</MarkdownContent>)
    expect(container.querySelector('h2')).not.toBeNull()
    expect(container.querySelector('strong')).not.toBeNull()
    expect(container.querySelectorAll('li').length).toBe(3)
    expect(screen.getByText(/Objetivos del Ejercicio/)).toBeInTheDocument()
  })

  it('renders a translated table section without raw markdown artifacts', () => {
    const section = [
      '| Objetivo | Capacidad |',
      '|---|---|',
      '| Comunicación | Coordinación Operativa |',
      '| Planificación | Respuesta Integrada |',
    ].join('\n')

    const { container } = render(<MarkdownContent>{section}</MarkdownContent>)
    expect(container.querySelector('table')).not.toBeNull()
    expect(container.textContent).not.toContain('|')
    expect(screen.getByText('Comunicación')).toBeInTheDocument()
    expect(screen.getByText('Coordinación Operativa')).toBeInTheDocument()
  })

  it('handles the mixed pipe-in-bullets scenario from screenshots', () => {
    // Some LLMs output table content as bullet items with pipe separators —
    // this should at least not crash and should show the readable content
    const section = [
      '- Comunicación Operativa | Coordinación entre organizaciones',
      '- Planificación | Discutir la coordinación entre organizaciones',
    ].join('\n')

    render(<MarkdownContent>{section}</MarkdownContent>)
    expect(screen.getByText(/Comunicación Operativa/)).toBeInTheDocument()
  })
})
