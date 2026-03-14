import { describe, expect, it } from 'vitest'
import { lintOutputFormatting } from './outputLint'

describe('lintOutputFormatting', () => {
  it('returns no issues for well-formed markdown', () => {
    const markdown = `## Summary

- Item A
- Item B

| Metric | Value |
| --- | ---: |
| Throughput | 120 |
`
    expect(lintOutputFormatting(markdown)).toEqual([])
  })

  it('detects malformed table rows', () => {
    const markdown = `| A | B |
| --- |
| 1 | 2 |`
    expect(lintOutputFormatting(markdown)).toContain('malformed_table_row')
  })

  it('detects heading depth jumps', () => {
    const markdown = `## Scope

#### Deep section`
    expect(lintOutputFormatting(markdown)).toContain('heading_level_jump')
  })

  it('detects source marker leakage', () => {
    const markdown = `Summary text.

Sources: [Source: 1]`
    expect(lintOutputFormatting(markdown)).toContain('source_marker_leakage')
  })
})
