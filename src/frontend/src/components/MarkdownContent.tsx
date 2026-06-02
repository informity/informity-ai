/**
 * Informity AI — Shared Markdown renderer
 * Used by TranslatePage and any surface needing basic GFM + math rendering.
 * Mirrors the plugin set used in ChatMessage's MessageBlocks so rendering
 * is consistent between Chat and Translate.
 */
import { Children, isValidElement, memo, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { preprocessMarkdown } from '../utils/markdownPreprocess'
import 'katex/dist/katex.min.css'
import './MarkdownContent.css'

/** True when a table cell's children contain no visible text content. */
function isCellEmpty(cellChildren: ReactNode): boolean {
  const nodes = Children.toArray(cellChildren)
  if (nodes.length === 0) return true
  return nodes.every(node => {
    if (typeof node === 'string') return node.trim() === ''
    if (typeof node === 'number') return false
    if (isValidElement(node)) return isCellEmpty((node.props as { children?: ReactNode }).children)
    return true
  })
}

interface MarkdownContentProps {
  children: string
  className?: string
}

export const MarkdownContent = memo(function MarkdownContent({ children, className }: MarkdownContentProps) {
  return (
    <div className={`md-content${className ? ` ${className}` : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          table: ({ children: tableChildren }) => (
            <div className="md-content__table-scroll">
              <table>{tableChildren}</table>
            </div>
          ),
          // Detect group-header rows: one non-empty cell, rest empty.
          // The LLM outputs "| **Category** | | | |" for what was a
          // colspan row in the source PDF. Render as a full-width header cell.
          tr: ({ children: rowChildren }) => {
            const cells = Children.toArray(rowChildren).filter(isValidElement)
            if (cells.length > 1) {
              const emptyCells = cells.filter(c =>
                isCellEmpty((c.props as { children?: ReactNode }).children)
              )
              if (emptyCells.length === cells.length - 1) {
                const filled = cells.find(c =>
                  !isCellEmpty((c.props as { children?: ReactNode }).children)
                )
                if (filled) {
                  return (
                    <tr className="md-content__table-group-header">
                      <td colSpan={100}>
                        {(filled.props as { children?: ReactNode }).children}
                      </td>
                    </tr>
                  )
                }
              }
            }
            return <tr>{rowChildren}</tr>
          },
        }}
      >
        {preprocessMarkdown(children)}
      </ReactMarkdown>
    </div>
  )
})
