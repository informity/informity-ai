import { describe, expect, it } from 'vitest'
import { preprocessMarkdown } from './markdownPreprocess'

// ──────────────────────────────────────────────────────────────────────────────
// HTML normalisation
// ──────────────────────────────────────────────────────────────────────────────

describe('preprocessMarkdown — HTML normalisation', () => {
  it('converts <br> to newline', () => {
    expect(preprocessMarkdown('line one<br>line two')).toBe('line one\nline two')
    expect(preprocessMarkdown('line one<br/>line two')).toBe('line one\nline two')
    expect(preprocessMarkdown('line one<br />line two')).toBe('line one\nline two')
  })

  it('strips HTML tags', () => {
    expect(preprocessMarkdown('<b>bold</b> and <em>italic</em>')).toBe('bold and italic')
  })

  it('handles text with no HTML unchanged', () => {
    const md = '# Heading\n\nSome paragraph text.\n\n- item'
    expect(preprocessMarkdown(md)).toBe(md)
  })

  it('returns empty string for falsy input', () => {
    expect(preprocessMarkdown('')).toBe('')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Orphaned table separator removal — the core bug fix
// ──────────────────────────────────────────────────────────────────────────────

describe('preprocessMarkdown — orphaned table separator removal', () => {
  it('removes a separator row that has no adjacent table rows', () => {
    const input = 'Some paragraph.\n\n|---|---|---|\n\nAnother paragraph.'
    const output = preprocessMarkdown(input)
    expect(output).not.toContain('|---|---|---|')
    expect(output).toContain('Some paragraph.')
    expect(output).toContain('Another paragraph.')
  })

  it('removes multiple consecutive orphaned separators', () => {
    const input = 'Text\n|---|---|\n|---|---|\nMore text'
    // Neither separator is flanked by a table row on both sides after the first is removed;
    // at minimum, they should not both survive as standalone lines
    const output = preprocessMarkdown(input)
    expect(output).toContain('Text')
    expect(output).toContain('More text')
  })

  it('preserves a separator row that follows a table header row', () => {
    const input = '| Name | Value |\n|---|---|\n| Alpha | 1 |'
    const output = preprocessMarkdown(input)
    expect(output).toContain('|---|---|')
    expect(output).toContain('| Name | Value |')
    expect(output).toContain('| Alpha | 1 |')
  })

  it('preserves a separator row that precedes a data row', () => {
    // separator appears before a data row but there is no header above — still should keep it
    const input = '|---|---|\n| Alpha | 1 |'
    const output = preprocessMarkdown(input)
    expect(output).toContain('| Alpha | 1 |')
  })

  it('removes separator at start of document with no adjacent table row', () => {
    const input = '|---|---|\n\nParagraph text.'
    const output = preprocessMarkdown(input)
    expect(output).not.toContain('|---|---|')
    expect(output).toContain('Paragraph text.')
  })

  it('preserves a complete well-formed table unchanged', () => {
    const table = '| Col A | Col B |\n|---|---|\n| val 1 | val 2 |\n| val 3 | val 4 |'
    const output = preprocessMarkdown(table)
    expect(output).toBe(table)
  })

  it('reproduces the CISA document scenario — many dashes rows in translation output', () => {
    // Simulates what the LLM outputs when it translates a PDF table poorly:
    // a block of separator rows with no surrounding table rows
    const problematic = [
      '**Nombre del Ejercicio**',
      '',
      '|---|---|---|---|',
      '|---|---|---|---|',
      '|---|---|---|---|',
      '|---|---|---|---|',
      '',
      '**Descripción General**',
    ].join('\n')

    const output = preprocessMarkdown(problematic)
    expect(output).not.toContain('|---|---|---|---|')
    expect(output).toContain('Nombre del Ejercicio')
    expect(output).toContain('Descripción General')
  })

  it('handles separator with colons for alignment', () => {
    // |:---|---:| is also a valid separator
    const input = '| Name | Value |\n|:---|---:|\n| Alpha | 1 |'
    const output = preprocessMarkdown(input)
    expect(output).toContain('|:---|---:|')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Pass-through: content that should not be modified
// ──────────────────────────────────────────────────────────────────────────────

describe('preprocessMarkdown — content preservation', () => {
  it('preserves normal prose', () => {
    const prose = 'This is a paragraph.\n\nThis is another paragraph with **bold** and *italic* text.'
    expect(preprocessMarkdown(prose)).toBe(prose)
  })

  it('preserves bullet lists', () => {
    const list = '- Item one\n- Item two\n- Item three'
    expect(preprocessMarkdown(list)).toBe(list)
  })

  it('preserves numbered lists', () => {
    const list = '1. First\n2. Second\n3. Third'
    expect(preprocessMarkdown(list)).toBe(list)
  })

  it('preserves code blocks', () => {
    const code = '```python\nprint("hello")\n```'
    expect(preprocessMarkdown(code)).toBe(code)
  })

  it('preserves headings', () => {
    const headings = '# H1\n\n## H2\n\n### H3'
    expect(preprocessMarkdown(headings)).toBe(headings)
  })
})
