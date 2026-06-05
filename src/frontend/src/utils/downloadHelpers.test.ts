import { describe, expect, it } from 'vitest'
import { markdownToPlainText } from './downloadHelpers'

// ──────────────────────────────────────────────────────────────────────────────
// Table flattening — the primary regression this test guards against
// ──────────────────────────────────────────────────────────────────────────────

describe('markdownToPlainText — table handling', () => {
  it('flattens a simple table to space-separated rows with NO pipe characters', () => {
    const md = '| Name | Value |\n|---|---|\n| Alpha | 1 |\n| Beta | 2 |'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('|')
    expect(out).toContain('Name')
    expect(out).toContain('Value')
    expect(out).toContain('Alpha')
    expect(out).toContain('Beta')
  })

  it('flattens a table with header and multiple data rows', () => {
    const md = [
      '| Exercise Name | Time | Objective |',
      '|---|---|---|',
      '| Scenario A | 09:00 | Comms |',
      '| Scenario B | 10:30 | Coordination |',
    ].join('\n')
    const out = markdownToPlainText(md)
    expect(out).not.toContain('|')
    expect(out).not.toContain('---')
    expect(out).toContain('Exercise Name')
    expect(out).toContain('Scenario A')
    expect(out).toContain('Scenario B')
    expect(out).toContain('Coordination')
  })

  it('does not include table separator dashes in output', () => {
    const md = '| A | B |\n|---|---|\n| x | y |'
    const out = markdownToPlainText(md)
    // The separator row should not appear as dashes in the output
    expect(out).not.toMatch(/^-{3,}/m)
    expect(out).not.toContain('---|')
  })

  it('flattens a translated document table — simulates CISA export scenario', () => {
    // This is the kind of content from the screenshots that was broken
    const translatedSection = [
      '## Objetivos del Ejercicio',
      '',
      '| Objetivo | Capacidad Asociada | Hora |',
      '|---|---|---|',
      '| Comunicación Operativa | Coordinación | 09:00 |',
      '| Planificación | Respuesta | 10:30 |',
      '',
      'Tabla 1. Objetivos del Ejercicio y Capacidades Asociadas',
    ].join('\n')

    const out = markdownToPlainText(translatedSection)
    expect(out).not.toContain('|')
    expect(out).toContain('Objetivos del Ejercicio')
    expect(out).toContain('Comunicación Operativa')
    expect(out).toContain('Coordinación')
    expect(out).toContain('Tabla 1.')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Standard Markdown elements
// ──────────────────────────────────────────────────────────────────────────────

describe('markdownToPlainText — standard elements', () => {
  it('normalizes escaped newlines before parsing', () => {
    const md = '# Title\\n\\n## Subtitle'
    const out = markdownToPlainText(md)
    expect(out).toContain('Title')
    expect(out).toContain('Subtitle')
    expect(out).not.toContain('\\n')
  })

  it('strips heading markers', () => {
    const md = '# Title\n\n## Subtitle\n\n### Sub-sub'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('#')
    expect(out).toContain('Title')
    expect(out).toContain('Subtitle')
  })

  it('strips bold and italic markers', () => {
    const md = '**bold text** and *italic text* and _also italic_'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('*')
    expect(out).not.toContain('_')
    expect(out).toContain('bold text')
    expect(out).toContain('italic text')
    expect(out).toContain('also italic')
  })

  it('converts bullet lists to text with bullet prefix', () => {
    const md = '- Item one\n- Item two\n- Item three'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('- Item')
    expect(out).toContain('Item one')
    expect(out).toContain('Item two')
    expect(out).toContain('• Item one')
  })

  it('converts numbered lists keeping numeric prefix', () => {
    const md = '1. First\n2. Second\n3. Third'
    const out = markdownToPlainText(md)
    expect(out).toContain('1.')
    expect(out).toContain('First')
    expect(out).toContain('Third')
  })

  it('strips inline code backticks, preserves content', () => {
    const md = 'Use the `npm install` command.'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('`')
    expect(out).toContain('npm install')
  })

  it('strips code fences, preserves code content', () => {
    const md = '```python\nprint("hello")\n```'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('```')
    expect(out).toContain('print("hello")')
  })

  it('uses link text, strips URL', () => {
    const md = 'See [the docs](https://example.com) for details.'
    const out = markdownToPlainText(md)
    expect(out).not.toContain('https://')
    expect(out).not.toContain('[')
    expect(out).toContain('the docs')
  })

  it('strips blockquote markers, preserves content', () => {
    const md = '> This is a quoted line.\n> And another.'
    const out = markdownToPlainText(md)
    expect(out).toContain('This is a quoted line.')
    expect(out).toContain('And another.')
  })

  it('returns empty string for empty input', () => {
    expect(markdownToPlainText('')).toBe('')
  })

  it('handles null/undefined gracefully', () => {
    expect(markdownToPlainText(null as unknown as string)).toBe('')
    expect(markdownToPlainText(undefined as unknown as string)).toBe('')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// Complex real-world document structure
// ──────────────────────────────────────────────────────────────────────────────

describe('markdownToPlainText — realistic document', () => {
  it('converts a mixed-content document cleanly', () => {
    const md = [
      '# Quarterly Report',
      '',
      'This report covers **Q3 2024** performance.',
      '',
      '## Financial Summary',
      '',
      '| Metric | Q2 | Q3 | Change |',
      '|---|---|---|---|',
      '| Revenue | $1.2M | $1.5M | +25% |',
      '| Costs | $0.8M | $0.9M | +12% |',
      '| Profit | $0.4M | $0.6M | +50% |',
      '',
      '## Key Initiatives',
      '',
      '1. Launch new product line',
      '2. Expand to European market',
      '3. Improve customer retention',
      '',
      '> All figures are approximate and subject to audit.',
    ].join('\n')

    const out = markdownToPlainText(md)
    expect(out).not.toContain('|')
    expect(out).not.toContain('#')
    expect(out).not.toContain('**')
    expect(out).not.toContain('>')
    expect(out).toContain('Quarterly Report')
    expect(out).toContain('Revenue')
    expect(out).toContain('$1.5M')
    expect(out).toContain('+50%')
    expect(out).toContain('Launch new product line')
    expect(out).toContain('All figures are approximate')
  })

  it('chat export: an assistant answer with a table exports without pipe artifacts', () => {
    // Simulates what gets exported from a Chat answer containing a table
    const chatAnswer = [
      'Here is a comparison of the options:',
      '',
      '| Option | Pros | Cons |',
      '|---|---|---|',
      '| A | Fast | Expensive |',
      '| B | Cheap | Slow |',
      '',
      'I recommend **Option A** based on your requirements.',
    ].join('\n')

    const out = markdownToPlainText(chatAnswer)
    expect(out).not.toContain('|')
    expect(out).toContain('Option')
    expect(out).toContain('Fast')
    expect(out).toContain('Expensive')
    expect(out).toContain('recommend')
    expect(out).toContain('Option A')
  })
})
