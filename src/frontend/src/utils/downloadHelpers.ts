/**
 * Shared download utilities used by both Chat and Translate.
 */
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import { preprocessMarkdown } from './markdownPreprocess'

type MarkdownNode = {
  type: string
  children?: MarkdownNode[]
  value?: string
  alt?: string
  ordered?: boolean
  start?: number
}

function nodeToText(node: MarkdownNode | null | undefined): string {
  if (!node) return ''
  switch (node.type) {
    case 'root':
      return (node.children ?? []).map(nodeToText).join('\n\n').replace(/\n{3,}/g, '\n\n').trim()
    case 'paragraph':
      return (node.children ?? []).map(nodeToText).join('')
    case 'text':
      return node.value ?? ''
    case 'strong': case 'emphasis': case 'delete':
      return (node.children ?? []).map(nodeToText).join('')
    case 'inlineCode':
      return node.value ?? ''
    case 'code':
      return node.value ?? ''
    case 'link':
      return (node.children ?? []).map(nodeToText).join('')
    case 'image':
      return node.alt ?? ''
    case 'heading':
      return (node.children ?? []).map(nodeToText).join('')
    case 'list':
      return (node.children ?? []).map((item, i: number) => {
        const bullet = node.ordered ? `${(node.start ?? 1) + i}. ` : '• '
        return bullet + nodeToText(item).replace(/\n/g, '\n  ')
      }).join('\n')
    case 'listItem':
      return (node.children ?? []).map(nodeToText).join('\n')
    case 'blockquote':
      return (node.children ?? []).map(nodeToText).join('\n')
    case 'table':
      return (node.children ?? []).map((row) =>
        (row.children ?? []).map((cell) =>
          (cell.children ?? []).map(nodeToText).join('').trim()
        ).join('  ')
      ).join('\n')
    case 'tableRow':
      return (node.children ?? []).map((cell) =>
        (cell.children ?? []).map(nodeToText).join('').trim()
      ).join('  ')
    case 'tableCell':
      return (node.children ?? []).map(nodeToText).join('')
    case 'thematicBreak': case 'html': case 'definition': case 'footnoteDefinition':
      return ''
    case 'break':
      return '\n'
    default:
      if (Array.isArray(node.children)) return node.children.map(nodeToText).join('')
      return node.value ?? ''
  }
}

/**
 * Convert Markdown to readable plain text using an AST-based approach.
 * Tables are flattened to space-separated rows. Formatting markers are stripped.
 */
export function markdownToPlainText(md: string): string {
  if (!md || typeof md !== 'string') return ''
  const tree = unified().use(remarkParse).use(remarkGfm).parse(preprocessMarkdown(md))
  return nodeToText(tree as unknown as MarkdownNode)
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
