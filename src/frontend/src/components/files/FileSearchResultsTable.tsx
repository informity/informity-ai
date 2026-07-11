/**
 * Informity AI — Semantic file search results table
 * Standalone duplicate of FileTable for future semantic-specific customization.
 */
import { useCallback } from 'react'
import type React from 'react'
import { openFile } from '../../api'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatDate } from '../../utils/formatDate'
import { getFileIcon } from '../../utils/fileFormatting'
import { StateMessage } from '../StateMessage'
import type { IndexedFile } from '../../types/api'
import './FileSearchResultsTable.css'

const PAGE_SIZE = 50

type SemanticSearchRow = IndexedFile & { score?: number }

function formatScore(score: number) {
  if (!Number.isFinite(score)) return '—'
  return String(Math.round(score))
}

interface FileSearchResultsTableProps {
  files?: SemanticSearchRow[]
  total?: number
  offset?: number
  limit?: number
  onPageChange?: (offset: number) => void
  onChatAboutFile?: (file: IndexedFile) => void
  onTranslate?: (file: IndexedFile) => void
  onReindex?: (file: IndexedFile) => void
  onRemove?: (file: IndexedFile, e: React.MouseEvent) => void
  onOpenFile?: (file: IndexedFile) => void | Promise<void>
  reindexingFileIds?: Set<number>
  offline?: boolean
}

export function FileSearchResultsTable({
  files = [],
  total = 0,
  offset = 0,
  limit = PAGE_SIZE,
  onPageChange,
  onChatAboutFile,
  onTranslate,
  onReindex,
  onRemove,
  onOpenFile,
  reindexingFileIds = new Set<number>(),
  offline = false,
}: FileSearchResultsTableProps) {
  const handleOpenFile = useCallback(async (file: IndexedFile) => {
    if (!file?.path?.trim()) return
    await openFile(file.path)
  }, [])

  const currentPage = Math.floor(offset / limit) + 1
  const totalPages = Math.max(1, Math.ceil(total / limit))
  const canPrev = offset > 0
  const canNext = offset + limit < total

  if (files.length === 0 && total === 0) {
    if (offline) {
      return (
        <div className="file-search-results-table file-search-results-table--empty data-table data-table--empty">
          <StateMessage className="file-search-results-table__empty-state data-table__empty-state" />
        </div>
      )
    }

    return (
      <div className="file-search-results-table file-search-results-table--empty data-table data-table--empty">
        <div className="file-search-results-table__empty-state data-table__empty-state">
          <i className="ri-file-copy-2-line file-search-results-table__empty-icon data-table__empty-icon" aria-hidden="true" />
          <p>No files indexed yet.</p>
          <p className="file-search-results-table__empty-hint data-table__empty-hint">Go to Settings to add folders, then run a scan.</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`file-search-results-table data-table${offline ? ' data-table--offline' : ''}`}>
      <div className="file-search-results-table__scroll data-table__scroll">
        <table className="file-search-results-table__table data-table__table">
          <thead>
            <tr>
              <th className="file-search-results-table__th file-search-results-table__th--filename data-table__th">Filename</th>
              <th className="file-search-results-table__th file-search-results-table__th--score data-table__th">Score</th>
              <th className="file-search-results-table__th file-search-results-table__th--size file-search-results-table__th--right data-table__th data-table__th--right">Size</th>
              <th className="file-search-results-table__th file-search-results-table__th--indexed file-search-results-table__th--right data-table__th data-table__th--right">Indexed</th>
              <th className="file-search-results-table__th file-search-results-table__th--modified file-search-results-table__th--right data-table__th data-table__th--right">Modified</th>
              <th className="file-search-results-table__th file-search-results-table__th--actions data-table__th" />
            </tr>
          </thead>
          <tbody>
            {files.map((file) => {
              const iconClass = getFileIcon(file.extension)
              const isReindexing = reindexingFileIds.has(file.id)

              return (
                <tr key={file.id} className="file-search-results-table__row data-table__row">
                  <td className="file-search-results-table__td file-search-results-table__td--filename data-table__td">
                    <div className="file-search-results-table__filename-row">
                      <i className={`${iconClass} file-search-results-table__filename-icon`} aria-hidden style={{ fontSize: '1rem' }} />
                      <button
                        type="button"
                        className="file-search-results-table__filename-button"
                        onClick={async (e) => {
                          e.stopPropagation()
                          if (offline) return
                          if (onOpenFile) {
                            await onOpenFile(file)
                            return
                          }
                          await handleOpenFile(file)
                        }}
                        disabled={offline || !file.path?.trim()}
                        aria-label={`Open ${file.filename || 'file'}`}
                      >
                        <span className="file-search-results-table__filename-text">{file.filename || '—'}</span>
                      </button>
                    </div>
                  </td>
                  <td className="file-search-results-table__td file-search-results-table__td--score data-table__td">
                    <span className="file-search-results-table__score-badge data-table__badge">{formatScore(file.score ?? 0)}</span>
                  </td>
                  <td className="file-search-results-table__td file-search-results-table__td--size file-search-results-table__td--right data-table__td data-table__td--right">
                    {formatFileSize(file.size_bytes)}
                  </td>
                  <td className="file-search-results-table__td file-search-results-table__td--indexed file-search-results-table__td--right data-table__td data-table__td--right">{formatDate(file.indexed_at)}</td>
                  <td className="file-search-results-table__td file-search-results-table__td--modified file-search-results-table__td--right data-table__td data-table__td--right">{formatDate(file.modified_at)}</td>
                  <td className="file-search-results-table__td file-search-results-table__td--actions data-table__td" onClick={(e) => e.stopPropagation()}>
                    <div className="file-search-results-table__actions">
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-search-results-table__action-btn data-table__action-btn"
                          onClick={() => onChatAboutFile?.(file)}
                          disabled={offline}
                          title="Chat with this file"
                        >
                          <i className="ri-chat-ai-4-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">Chat with this file</span>
                      </span>
                      {onTranslate && (
                        <span className="data-table__action-wrap ui-tooltip-trigger">
                          <button
                            type="button"
                            className="file-search-results-table__action-btn data-table__action-btn"
                            onClick={() => onTranslate(file)}
                            disabled={offline}
                            title="Translate this file"
                          >
                            <i className="ri-translate-2" aria-hidden style={{ fontSize: '0.875rem' }} />
                          </button>
                          <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">Translate this file</span>
                        </span>
                      )}
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-search-results-table__action-btn data-table__action-btn"
                          onClick={() => onReindex?.(file)}
                          disabled={offline || isReindexing}
                          title={isReindexing ? 'Reindex in progress' : 'Reindex file'}
                        >
                          <i
                            className={isReindexing ? 'ri-loader-4-line file-search-results-table__spinner' : 'ri-refresh-line'}
                            aria-hidden
                            style={{ fontSize: '0.875rem' }}
                          />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">
                          {isReindexing ? 'Reindexing…' : 'Reindex file'}
                        </span>
                      </span>
                      <span className="data-table__action-wrap data-table__action-wrap--last ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-search-results-table__action-btn file-search-results-table__action-btn--danger data-table__action-btn"
                          onClick={(e) => onRemove?.(file, e)}
                          disabled={offline}
                          title="Remove file from index"
                        >
                          <i className="ri-delete-bin-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap ui-tooltip--right-anchor">Remove file from index</span>
                      </span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="file-search-results-table__pagination">
          <span className="file-search-results-table__pagination-info">
            {offset + 1}–{Math.min(offset + limit, total)} of {total}
          </span>
          <div className="file-search-results-table__pagination-buttons">
            <button
              type="button"
              className="file-search-results-table__pagination-btn"
              onClick={() => onPageChange?.(Math.max(0, offset - limit))}
              disabled={!canPrev || offline}
              aria-label="Previous page"
            >
              ←
            </button>
            <span className="file-search-results-table__pagination-page">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="file-search-results-table__pagination-btn"
              onClick={() => onPageChange?.(offset + limit)}
              disabled={!canNext || offline}
              aria-label="Next page"
            >
              →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
