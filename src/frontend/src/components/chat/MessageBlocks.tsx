import { Children, isValidElement, memo, useLayoutEffect, useMemo, useRef, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { formatCodeLanguageLabel } from '../../utils/codeLanguageLabel'
import type {
  DisplayBlock,
  DisplayCalloutBlock,
  DisplayCodeBlock,
  DisplayListBlock,
  DisplayMetricBlock,
  DisplayQuoteBlock,
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
  | DisplayQuoteBlock
  | DisplayTableBlock
  | DisplayListBlock

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
            {
              const language = typeof block.language === 'string' && block.language.trim().length > 0
                ? block.language.trim().toLowerCase()
                : 'code'
            return (
              <div key={key} className="chat-message__block chat-message__block--code">
                <CodeCard
                  language={formatCodeLanguageLabel(language)}
                  code={typeof block.code === 'string' ? block.code : ''}
                  codeBlockCopied={codeBlockCopied}
                  onCopyCode={onCopyCode}
                  codeClassName={typeof block.language === 'string' && block.language ? `language-${block.language}` : undefined}
                />
              </div>
            )
            }
          case 'callout':
            {
              const text = typeof block.text === 'string' ? block.text : ''
              const disclaimerBody = extractDisclaimerBody(text)
              const isDisclaimer = disclaimerBody !== null
              const className = [
                'chat-message__block',
                'chat-message__block--callout',
                `chat-message__block--${block.tone ?? 'info'}`,
                isDisclaimer ? 'chat-message__block--disclaimer' : '',
              ].filter(Boolean).join(' ')
              return (
                <div key={key} className={className}>
                  {isDisclaimer ? (
                    <DisclaimerCard text={disclaimerBody} />
                  ) : (
                    text
                  )}
                </div>
              )
            }
          case 'metric':
            return (
              <div key={key} className="chat-message__block chat-message__block--metric">
                <div className="chat-message__metric-label">{typeof block.label === 'string' ? block.label : ''}</div>
                <div className="chat-message__metric-value">{typeof block.value === 'string' ? block.value : ''}</div>
              </div>
            )
          case 'quote':
            return (
              <blockquote key={key} className="chat-message__block chat-message__block--quote">
                <p>{typeof block.text === 'string' ? block.text : ''}</p>
                {typeof block.attribution === 'string' && block.attribution.trim().length > 0 && (
                  <cite>{block.attribution}</cite>
                )}
              </blockquote>
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
                          <th key={`${key}-h-${colIdx}`}>
                            <InlineMarkdown markdown={column} />
                          </th>
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
                              <InlineMarkdown markdown={value} />
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
          case 'list': {
            return (
              <div key={key} className="chat-message__block chat-message__block--list">
                <StructuredList ordered={!!block.ordered} items={block.items} />
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
        th: ({ children }) => {
          const text = flattenNodeText(Children.toArray(children))
          const align = isNumericCellContent(text) ? 'right' : 'left'
          return <th data-align={align}>{children}</th>
        },
        td: ({ children }) => {
          const text = flattenNodeText(Children.toArray(children))
          const align = isNumericCellContent(text) ? 'right' : undefined
          return <td data-align={align}>{children}</td>
        },
        pre: ({ children }) => (
          <CodeCard
            language={formatCodeLanguageLabel('code')}
            code={flattenNodeText(Children.toArray(children))}
            codeBlockCopied={codeBlockCopied}
            onCopyCode={onCopyCode}
            preChildren={children}
          />
        ),
      }}
    >
      {markdown}
    </ReactMarkdown>
  )
}

interface CodeCardProps {
  language: string
  code: string
  codeBlockCopied: boolean
  onCopyCode: (code: string) => void
  codeClassName?: string
  preChildren?: ReactNode
}

function CodeCard({ language, code, codeBlockCopied, onCopyCode, codeClassName, preChildren }: CodeCardProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null)

  useLayoutEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollLeft = 0
    }
  }, [code])

  return (
    <div className="chat-message__code-wrapper">
      <div className="chat-message__code-header">
        <span className="chat-message__code-language">{language}</span>
        <button
          type="button"
          className="chat-message__copy-code"
          onClick={() => onCopyCode(code)}
          title="Copy code"
          aria-label="Copy code block"
        >
          {codeBlockCopied ? (
            <i className="ri-check-line" aria-hidden style={{ fontSize: '0.875rem' }} />
          ) : (
            <i className="ri-file-copy-line" aria-hidden style={{ fontSize: '0.875rem' }} />
          )}
        </button>
      </div>
      <div ref={bodyRef} className="chat-message__code-body">
        {preChildren ? (
          <pre>{preChildren}</pre>
        ) : (
          <pre>
            <code className={codeClassName}>{code}</code>
          </pre>
        )}
      </div>
    </div>
  )
}

function DisclaimerCard({ text }: { text: string }) {
  return (
    <div className="chat-message__disclaimer-card">
      <div className="chat-message__disclaimer-header">
        <span className="chat-message__disclaimer-label">Disclaimer</span>
      </div>
      <div className="chat-message__disclaimer-body">
        <p>{text}</p>
      </div>
    </div>
  )
}

function extractDisclaimerBody(text: string): string | null {
  const normalized = String(text || '').trim()
  const match = normalized.match(/^disclaimer\s*:\s*(.+)$/i)
  if (!match) return null
  return match[1].trim()
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

interface StructuredListItem {
  text: string
  level?: number
  checked?: boolean | null
}

function StructuredList({ ordered, items }: { ordered: boolean; items: StructuredListItem[] }) {
  const ListTag = ordered ? 'ol' : 'ul'
  const normalized = items.map((item) => ({
    text: item.text,
    level: Math.max(0, typeof item.level === 'number' ? item.level : 0),
    checked: item.checked,
  }))
  return (
    <ListTag>
      {normalized.map((item, idx) => (
        <li key={`${idx}-${item.level}`} style={{ marginLeft: `${item.level * 0.8}rem` }}>
          {typeof item.checked === 'boolean' && (
            <input type="checkbox" checked={item.checked} readOnly tabIndex={-1} aria-hidden />
          )}
          <InlineMarkdown markdown={item.text} />
        </li>
      ))}
    </ListTag>
  )
}

function InlineMarkdown({ markdown }: { markdown: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <>{children}</>,
      }}
    >
      {markdown}
    </ReactMarkdown>
  )
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
      case 'quote': {
        if (typeof block.text !== 'string' || block.text.trim().length === 0) break
        normalized.push({
          type: 'quote',
          text: block.text,
          attribution: typeof block.attribution === 'string' ? block.attribution : undefined,
        })
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
      case 'list': {
        if (!Array.isArray(block.items) || block.items.length === 0) break
        const items = block.items
          .filter((item): item is { text: string; level?: number; checked?: boolean | null } => (
            !!item && typeof item === 'object' && typeof item.text === 'string'
          ))
          .map((item) => ({
            text: item.text,
            level: typeof item.level === 'number' ? Math.max(0, Math.floor(item.level)) : 0,
            checked: typeof item.checked === 'boolean' ? item.checked : null,
          }))
        if (items.length === 0) break
        normalized.push({
          type: 'list',
          ordered: !!block.ordered,
          items,
        })
        break
      }
      default:
        break
    }
  }

  return normalized
}
