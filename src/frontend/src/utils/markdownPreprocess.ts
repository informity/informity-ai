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
 * 3. Wraps inline source markers in inert markdown links so the chat renderer can
 *    display them with muted theme-aware styling.
 */
export function preprocessMarkdown(text: string): string {
  if (!text) return ''

  const normalised = text
    .replace(/\\n/g, '\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')

  const lines = normalised.split('\n')
  const kept: string[] = []
  let insideFence = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    if (/^(```|~~~)/.test(trimmed)) {
      insideFence = !insideFence
      kept.push(line)
      continue
    }
    if (insideFence) {
      kept.push(line)
      continue
    }
    if (isTableSeparator(line)) {
      const prevLine = kept[kept.length - 1] ?? ''
      const nextLine = lines[i + 1] ?? ''
      if (isTableRow(prevLine) || isTableRow(nextLine)) {
        kept.push(line)
      }
      // Orphaned separator — silently dropped
    } else {
      kept.push(muteSourceMarkers(line))
    }
  }

  return kept.join('\n')
}

const SOURCE_MARKER_PATTERN = /\[(?:\s*sources?\s*:\s*\d+(?:\s*,\s*sources?\s*:\s*\d+)*\s*)\]/gi

function muteSourceMarkers(line: string): string {
  return line.replace(
    SOURCE_MARKER_PATTERN,
    (match) => `[${match.slice(1, -1)}](#informity-source-marker)`,
  )
}
