/**
 * Markdown pre-processing utilities.
 * Cleans common LLM output artifacts before rendering or converting to plaintext.
 */

/** True when the line is a GFM table separator: |---|:---:|--- */
function isTableSeparator(line: string): boolean {
  const t = line.trim()
  // Must start with | (optional) then only dashes, colons, pipes, and spaces
  return /^\|[\s]*[-:][- |:]*$/.test(t) && t.includes('-')
}

/**
 * True when the line looks like a GFM data/header row (starts with | and has
 * non-separator content). Explicitly excludes separator rows so that consecutive
 * separators don't count as "adjacent table rows" for each other.
 */
function isTableRow(line: string): boolean {
  const t = line.trim()
  return /^\|.+/.test(t) && !isTableSeparator(t)
}

/**
 * Pre-process LLM markdown output before rendering or plaintext conversion.
 *
 * 1. Normalises HTML artefacts (<br> → newline, strips HTML tags)
 * 2. Removes orphaned table separator rows (|---|---| with no adjacent table row on either side).
 *    These appear when the LLM outputs a separator row without the surrounding header/data rows,
 *    which remark-gfm cannot parse as a table and renders as literal dashes.
 */
export function preprocessMarkdown(text: string): string {
  if (!text) return ''

  const normalised = text
    .replace(/\\n/g, '\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')

  const lines = normalised.split('\n')
  const kept: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (isTableSeparator(line)) {
      const prevLine = kept[kept.length - 1] ?? ''
      const nextLine = lines[i + 1] ?? ''
      if (isTableRow(prevLine) || isTableRow(nextLine)) {
        kept.push(line)
      }
      // Orphaned separator — silently dropped
    } else {
      kept.push(line)
    }
  }

  return kept.join('\n')
}
