export type OutputLintIssue =
  | 'malformed_table_row'
  | 'heading_level_jump'
  | 'source_marker_leakage'

export function lintOutputFormatting(markdown: string): OutputLintIssue[] {
  const issues: OutputLintIssue[] = []
  if (!markdown.trim()) return issues

  const lines = markdown.split(/\r?\n/)
  const headingLevels = lines
    .map((line) => {
      const match = line.match(/^\s*(#{1,6})\s+\S+/)
      return match ? match[1].length : null
    })
    .filter((v): v is number => typeof v === 'number')

  for (let i = 1; i < headingLevels.length; i += 1) {
    if (headingLevels[i] - headingLevels[i - 1] > 1) {
      issues.push('heading_level_jump')
      break
    }
  }

  if (/(?:^|\s)(?:sources?:|\[source:\s*\d+\]|\(source\s*\d+\))/im.test(markdown)) {
    issues.push('source_marker_leakage')
  }

  const tableLines = lines.filter((line) => line.includes('|') && line.trim().length > 0)
  if (tableLines.length >= 2) {
    const dividerPattern = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/
    const nonDividerRows = tableLines.filter((line) => !dividerPattern.test(line))
    const columnCounts = nonDividerRows.map((line) => {
      const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
      return trimmed.split('|').length
    })
    const dividerColumns = tableLines
      .filter((line) => dividerPattern.test(line))
      .map((line) => {
        const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
        return trimmed.split('|').length
      })
    if (columnCounts.length > 1) {
      const first = columnCounts[0]
      if (columnCounts.some((count) => count !== first)) {
        issues.push('malformed_table_row')
      }
    }
    if (!issues.includes('malformed_table_row') && dividerColumns.length > 0 && columnCounts.length > 0) {
      const expected = columnCounts[0]
      if (dividerColumns.some((count) => count !== expected)) {
        issues.push('malformed_table_row')
      }
    }
  }

  return issues
}
