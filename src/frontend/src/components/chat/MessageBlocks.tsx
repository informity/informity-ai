import { Children, isValidElement, memo, useMemo, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import type {
  DisplayBlock,
  DisplayCalloutBlock,
  DisplayCodeBlock,
  DisplayMetricBlock,
  DisplayTableBlock,
  DisplayTextBlock,
} from '../../types/api'

interface MessageBlocksProps {
  blocks: DisplayBlock[]
  fallbackMarkdown: string
  onCopyCode: (code: string) => void
  codeBlockCopied: boolean
}

type RenderableBlock =
  | DisplayTextBlock
  | DisplayCodeBlock
  | DisplayCalloutBlock
  | DisplayMetricBlock
  | DisplayTableBlock

function MessageBlocksComponent({ blocks, fallbackMarkdown, onCopyCode, codeBlockCopied }: MessageBlocksProps) {
  const normalizedBlocks = useMemo(
    () => (Array.isArray(blocks) && blocks.length > 0 ? normalizeRenderableBlocks(blocks) : []),
    [blocks],
  )

  if (normalizedBlocks.length === 0) {
    return <MarkdownBlock markdown={fallbackMarkdown} onCopyCode={onCopyCode} codeBlockCopied={codeBlockCopied} />
  }

  return (
    <>
      {normalizedBlocks.map((block, index) => {
        const key = `${block.type}-${index}`

        switch (block.type) {
          case 'text':
            return (
              <MarkdownBlock
                key={key}
                markdown={typeof block.markdown === 'string' ? block.markdown : ''}
                onCopyCode={onCopyCode}
                codeBlockCopied={codeBlockCopied}
              />
            )
          case 'code':
            return (
              <div key={key} className="chat-message__block chat-message__block--code">
                <pre>
                  <code>{typeof block.code === 'string' ? block.code : ''}</code>
                </pre>
              </div>
            )
          case 'callout':
            return (
              <div key={key} className={`chat-message__block chat-message__block--callout chat-message__block--${block.tone ?? 'info'}`}>
                {typeof block.text === 'string' ? block.text : ''}
              </div>
            )
          case 'metric':
            return (
              <div key={key} className="chat-message__block chat-message__block--metric">
                <div className="chat-message__metric-label">{typeof block.label === 'string' ? block.label : ''}</div>
                <div className="chat-message__metric-value">{typeof block.value === 'string' ? block.value : ''}</div>
              </div>
            )
          case 'table': {
            const columns = block.columns
            const rows = block.rows
            return (
              <div key={key} className="chat-message__table-scroll">
                <table>
                  {columns.length > 0 && (
                    <thead>
                      <tr>
                        {columns.map((column, colIdx) => (
                          <th key={`${key}-h-${colIdx}`}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                  )}
                  <tbody>
                    {rows.map((row, rowIdx) => (
                      <tr key={`${key}-r-${rowIdx}`}>
                        {row.map((cell, colIdx) => {
                          const value = cell == null ? '' : String(cell)
                          const align = isNumericCellContent(value) ? 'right' : undefined
                          return (
                            <td key={`${key}-c-${rowIdx}-${colIdx}`} data-align={align}>
                              {value}
                            </td>
                          )
                        })}
                        {row.length < columns.length &&
                          Array.from({ length: columns.length - row.length }).map((_, idx) => (
                            <td key={`${key}-pad-${rowIdx}-${idx}`} />
                          ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }
        }
      })}
    </>
  )
}

export const MessageBlocks = memo(MessageBlocksComponent)

interface MarkdownBlockProps {
  markdown: string
  onCopyCode: (code: string) => void
  codeBlockCopied: boolean
}

function MarkdownBlock({ markdown, onCopyCode, codeBlockCopied }: MarkdownBlockProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        table: ({ children }) => (
          <div className="chat-message__table-scroll">
            <table>{children}</table>
          </div>
        ),
        td: ({ children }) => {
          const text = flattenNodeText(Children.toArray(children))
          const align = isNumericCellContent(text) ? 'right' : undefined
          return <td data-align={align}>{children}</td>
        },
        pre: ({ children }) => (
          <div className="chat-message__code-wrapper">
            <button
              type="button"
              className="chat-message__copy-code"
              onClick={(e) => {
                const pre = e.currentTarget.nextElementSibling
                onCopyCode(pre?.textContent ?? '')
              }}
              title="Copy code"
              aria-label="Copy code block"
            >
              {codeBlockCopied ? (
                <i className="ri-check-line" aria-hidden style={{ fontSize: '0.875rem' }} />
              ) : (
                <i className="ri-file-copy-line" aria-hidden style={{ fontSize: '0.875rem' }} />
              )}
            </button>
            <pre>{children}</pre>
          </div>
        ),
      }}
    >
      {markdown}
    </ReactMarkdown>
  )
}

function flattenNodeText(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map((child) => flattenNodeText(child)).join('')
  if (isValidElement<{ children?: ReactNode }>(node)) return flattenNodeText(node.props.children)
  return ''
}

function isNumericCellContent(value: string): boolean {
  const normalized = value.trim()
  if (!normalized) return false
  return /^[+-]?(?:[$€£]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$/.test(normalized)
}

function normalizeRenderableBlocks(blocks: DisplayBlock[]): RenderableBlock[] {
  const normalized: RenderableBlock[] = []

  for (const block of blocks) {
    switch (block.type) {
      case 'text': {
        if (typeof block.markdown === 'string') {
          normalized.push({ type: 'text', markdown: block.markdown })
        }
        break
      }
      case 'code': {
        if (typeof block.code === 'string') {
          normalized.push({
            type: 'code',
            code: block.code,
            language: typeof block.language === 'string' ? block.language : undefined,
          })
        }
        break
      }
      case 'callout': {
        if (typeof block.text === 'string') {
          const toneValue = typeof block.tone === 'string' ? block.tone : ''
          const tone: DisplayCalloutBlock['tone'] =
            toneValue === 'info' || toneValue === 'warning' || toneValue === 'success' || toneValue === 'danger'
              ? toneValue
              : 'info'
          normalized.push({ type: 'callout', text: block.text, tone })
        }
        break
      }
      case 'metric': {
        if (typeof block.label === 'string' && typeof block.value === 'string') {
          normalized.push({ type: 'metric', label: block.label, value: block.value })
        }
        break
      }
      case 'table': {
        if (!Array.isArray(block.columns) || !Array.isArray(block.rows)) break
        const columns = block.columns.filter((column): column is string => typeof column === 'string')
        if (columns.length === 0) break
        const rows = block.rows
          .filter((row): row is Array<string | number | null> => Array.isArray(row))
          .map((row) =>
            row
              .filter(
                (cell): cell is string | number | null =>
                  cell === null || typeof cell === 'string' || typeof cell === 'number',
              ),
          )
        normalized.push({ type: 'table', columns, rows })
        break
      }
      default:
        break
    }
  }

  return normalized
}
