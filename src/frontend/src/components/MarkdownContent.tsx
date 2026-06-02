/**
 * Informity AI — Shared Markdown renderer
 * Used by TranslatePage and any surface needing basic GFM + math rendering.
 * Mirrors the plugin set used in ChatMessage's MessageBlocks so rendering
 * is consistent between Chat and Translate.
 */
import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { preprocessMarkdown } from '../utils/markdownPreprocess'
import 'katex/dist/katex.min.css'
import './MarkdownContent.css'

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
        }}
      >
        {preprocessMarkdown(children)}
      </ReactMarkdown>
    </div>
  )
})
