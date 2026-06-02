/**
 * Shared download utilities used by both Chat and Translate.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function nodeToText(node: any): string {
  if (!node) return ''
  switch (node.type) {
    case 'root':
      return (node.children as any[]).map(nodeToText).join('\n\n').replace(/\n{3,}/g, '\n\n').trim()
    case 'paragraph':
      return (node.children as any[]).map(nodeToText).join('')
    case 'text':
      return node.value ?? ''
    case 'strong': case 'emphasis': case 'delete':
      return (node.children as any[]).map(nodeToText).join('')
    case 'inlineCode':
      return node.value ?? ''
    case 'code':
      return node.value ?? ''
    case 'link':
      return (node.children as any[]).map(nodeToText).join('')
    case 'image':
      return node.alt ?? ''
    case 'heading':
      return (node.children as any[]).map(nodeToText).join('')
    case 'list':
      return (node.children as any[]).map((item: any, i: number) => {
        const bullet = node.ordered ? `${(node.start ?? 1) + i}. ` : '• '
        return bullet + nodeToText(item).replace(/\n/g, '\n  ')
      }).join('\n')
    case 'listItem':
      return (node.children as any[]).map(nodeToText).join('\n')
    case 'blockquote':
      return (node.children as any[]).map(nodeToText).join('\n')
    case 'table':
      return (node.children as any[]).map((row: any) =>
        (row.children as any[]).map((cell: any) =>
          (cell.children as any[]).map(nodeToText).join('').trim()
        ).join('  ')
      ).join('\n')
    case 'tableRow':
      return (node.children as any[]).map((cell: any) =>
        (cell.children as any[]).map(nodeToText).join('').trim()
      ).join('  ')
    case 'tableCell':
      return (node.children as any[]).map(nodeToText).join('')
    case 'thematicBreak': case 'html': case 'definition': case 'footnoteDefinition':
      return ''
    case 'break':
      return '\n'
    default:
      if (Array.isArray(node.children)) return (node.children as any[]).map(nodeToText).join('')
      return node.value ?? ''
  }
}

/**
 * Convert Markdown to readable plain text using an AST-based approach.
 * Tables are flattened to space-separated rows. Formatting markers are stripped.
 * Falls back to basic regex stripping if parsing fails.
 */
export function markdownToPlainText(md: string): string {
  if (!md || typeof md !== 'string') return ''
  try {
    const tree = unified().use(remarkParse).use(remarkGfm).parse(md)
    return nodeToText(tree)
  } catch {
    return md
      .replace(/<[^>]+>/g, '')
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/\*{1,2}(.+?)\*{1,2}/gs, '$1')
      .replace(/`+([^`]+)`+/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  }
}

/** Trigger a browser download of a text file. */
export function downloadTextFile(filename: string, content: string, mimeType = 'text/plain;charset=utf-8'): void {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = String(filename || 'download.txt').trim() || 'download.txt'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Trigger a browser download of a Markdown file. */
export function downloadMarkdownFile(filename: string, content: string): void {
  downloadTextFile(filename, content, 'text/markdown; charset=utf-8')
}
