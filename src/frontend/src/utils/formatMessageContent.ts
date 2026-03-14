/**
 * Format message content for display.
 * Strips <think> blocks and citation references so the user sees clean answers.
 * Used for both streaming and loaded chat history.
 */

/**
 * Strip complete <think> blocks and orphaned opening tags.
 * Handles streaming: when content is cut mid-block, truncate at <think>.
 */
function stripThinkBlocks(text: string): string {
  let t = text

  // 1. Strip Phi-4 format first to avoid leftover wrapper remnants ("<>")
  t = t.replace(/<<think>>[\s\S]*?<\/think>>/gi, '')
  // 2. Strip complete think blocks (standard format: <think>...</think>)
  t = t.replace(/<think>[\s\S]*?<\/think>/gi, '')

  // 3. Strip orphaned opening tags (streaming cut off mid-block)
  const lower = t.toLowerCase()
  const thinkIdx = lower.indexOf('<think>')
  if (thinkIdx !== -1 && lower.indexOf('</think>', thinkIdx) === -1) {
    t = t.substring(0, thinkIdx)
  }
  const lowerAfterTrim = t.toLowerCase()
  const thinkIdxDouble = lowerAfterTrim.indexOf('<<think>>')
  if (thinkIdxDouble !== -1 && lowerAfterTrim.indexOf('</think>>', thinkIdxDouble) === -1) {
    t = t.substring(0, thinkIdxDouble)
  }

  return t
}

/**
 * Remove [Source: N] citation references from text.
 */
function stripCitations(text: string): string {
  return text
    .replace(/\[Source:\s*\d+\]/gi, '')
    .replace(/\(Source\s*\d+\)/gi, '')
    .replace(/\(\s*Source\s*\d+(?:\s*,\s*Source\s*\d+)*\s*\)/gi, '')
    .replace(/\(\s*Sources?\s*\d+(?:\s*,\s*\d+)*\s*\)/gi, '')
    .replace(/^\s*Sources?\s*:\s*.*$/gim, '')
    .replace(/^\s*Source\s+\d+(?:\s*,\s*Source\s+\d+)*\s*$/gim, '')
}

function normalizeDisplayWhitespace(text: string): string {
  return text
    .replace(/<br\s*\/?>/gi, '; ')
    // Preserve leading indentation so nested markdown bullets keep structure.
    .split('\n')
    .map((line) => {
      const match = line.match(/^([ \t]*)(.*)$/)
      if (!match) return line
      const leading = match[1]
      const content = match[2].replace(/[ \t]{2,}/g, ' ')
      return `${leading}${content}`
    })
    .join('\n')
    .replace(/\n\s*\n\s*\n/g, '\n\n')
    .trim()
}

/**
 * Format assistant message content for display.
 * Strips <think> blocks and citation references.
 */
export function formatMessageContent(text: string | null | undefined): string {
  if (text == null || typeof text !== 'string') return ''
  let cleaned = stripThinkBlocks(text)
  cleaned = stripCitations(cleaned)
  cleaned = normalizeDisplayWhitespace(cleaned)
  return cleaned
}
