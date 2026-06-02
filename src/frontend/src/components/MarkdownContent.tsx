/**
 * Informity AI — Shared Markdown renderer
 * Used by TranslatePage and any surface needing basic GFM rendering.
 */
import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { preprocessMarkdown } from '../utils/markdownPreprocess'
import './MarkdownContent.css'

interface MarkdownContentProps {
  children: string
  className?: string
}

export const MarkdownContent = memo(function MarkdownContent({ children, className }: MarkdownContentProps) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children: tableChildren }) => (
            <div className="md-content__table-scroll">
              <table>{tableChildren}</table>
            </div>
          ),
        }}
      >
        {preprocessMarkdown(children)}
      </ReactMarkdown>
    </div>
  )
})
