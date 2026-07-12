/**
 * Informity AI — File table
 * Sortable columns, row selection, multi-select, pagination.
 */
import { useCallback } from 'react'
import { openFile } from '../../api'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatDate } from '../../utils/formatDate'
import { formatCategory, getFileIcon } from '../../utils/fileFormatting'
import { SortIcon } from '../SortIcon'
import { StateMessage } from '../StateMessage'
import type { IndexedFile } from '../../types/api'
import './FileTable.css'

const PAGE_SIZE = 50
const SORT_COLUMNS = ['filename', 'category', 'extension', 'size_bytes', 'modified_at', 'indexed_at']

type SortColumn = (typeof SORT_COLUMNS)[number]
type SortOrder = 'asc' | 'desc'

interface FileTableProps {
  files?: IndexedFile[]
  total?: number
  offset?: number
  limit?: number
  sort?: SortColumn
  order?: SortOrder
  onSortChange?: (col: SortColumn, order: SortOrder) => void
  onPageChange?: (offset: number) => void
  onChatAboutFile?: (file: IndexedFile) => void
  onTranslate?: (file: IndexedFile) => void
  onReindex?: (file: IndexedFile) => void
  onRemove?: (file: IndexedFile, e: React.MouseEvent) => void
  onOpenFile?: (file: IndexedFile) => void | Promise<void>
  reindexingFileIds?: Set<number>
  offline?: boolean
}

export function FileTable({
  files = [],
  total = 0,
  offset = 0,
  limit = PAGE_SIZE,
  sort = 'indexed_at',
  order = 'desc',
  onSortChange,
  onPageChange,
  onChatAboutFile,
  onTranslate,
  onReindex,
  onRemove,
  onOpenFile,
  reindexingFileIds = new Set<number>(),
  offline = false,
}: FileTableProps) {
  const handleHeaderClick = useCallback(
    (col: string) => {
      if (offline) return
      if (!SORT_COLUMNS.includes(col)) return
      const nextOrder = sort === col && order === 'desc' ? 'asc' : 'desc'
      onSortChange?.(col as SortColumn, sort === col ? (nextOrder as SortOrder) : 'desc')
    },
    [offline, sort, order, onSortChange],
  )

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
        <div className="file-table file-table--empty data-table data-table--empty">
          <StateMessage className="file-table__empty-state data-table__empty-state" />
        </div>
      )
    }

    return (
      <div className="file-table file-table--empty data-table data-table--empty">
        <div className="file-table__empty-state data-table__empty-state">
          <i className="ri-file-copy-2-line file-table__empty-icon data-table__empty-icon" aria-hidden="true" />
          <p>No files indexed yet.</p>
          <p className="file-table__empty-hint data-table__empty-hint">Go to Settings to add folders, then run a scan.</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`file-table data-table${offline ? ' data-table--offline' : ''}`}>
      <div className="file-table__scroll data-table__scroll">
        <table className="file-table__table data-table__table">
          <thead>
            <tr>
              <th
                className={`file-table__th file-table__th--filename file-table__th--sortable data-table__th data-table__th--sortable ${
                  sort === 'filename' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('filename')}
              >
                Filename
                <SortIcon sort={sort} order={order} column="filename" />
              </th>
              <th
                className={`file-table__th file-table__th--category file-table__th--sortable data-table__th data-table__th--sortable ${
                  sort === 'extension' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('extension')}
              >
                Category
                <SortIcon sort={sort} order={order} column="extension" />
              </th>
              <th
                className={`file-table__th file-table__th--size file-table__th--sortable file-table__th--right data-table__th data-table__th--sortable data-table__th--right ${
                  sort === 'size_bytes' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('size_bytes')}
              >
                Size
                <SortIcon sort={sort} order={order} column="size_bytes" />
              </th>
              <th
                className={`file-table__th file-table__th--indexed file-table__th--sortable file-table__th--right data-table__th data-table__th--sortable data-table__th--right ${
                  sort === 'indexed_at' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('indexed_at')}
              >
                Indexed
                <SortIcon sort={sort} order={order} column="indexed_at" />
              </th>
              <th
                className={`file-table__th file-table__th--modified file-table__th--sortable file-table__th--right data-table__th data-table__th--sortable data-table__th--right ${
                  sort === 'modified_at' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('modified_at')}
              >
                Modified
                <SortIcon sort={sort} order={order} column="modified_at" />
              </th>
              <th className="file-table__th file-table__th--actions data-table__th" />
            </tr>
          </thead>
          <tbody>
            {files.map((file) => {
              const iconClass = getFileIcon(file.extension)
              const isReindexing = reindexingFileIds.has(file.id)

              return (
                <tr key={file.id} className="file-table__row data-table__row">
                  <td className="file-table__td file-table__td--filename data-table__td">
                    <div className="data-table__value">
                      <div className="file-table__filename-row">
                        <i className={`${iconClass} file-table__filename-icon`} aria-hidden style={{ fontSize: '1rem' }} />
                        <button
                          type="button"
                          className="file-table__filename-button"
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
                          <span className="file-table__filename-text">{file.filename || '—'}</span>
                        </button>
                      </div>
                    </div>
                  </td>
                  <td className="file-table__td file-table__td--category data-table__td">
                    <div className="data-table__value">
                      <span className="file-table__category-badge data-table__badge">{formatCategory(file.category, file.extension)}</span>
                    </div>
                  </td>
                  <td className="file-table__td file-table__td--size file-table__td--right data-table__td data-table__td--right">
                    <div className="data-table__value data-table__value--right">{formatFileSize(file.size_bytes)}</div>
                  </td>
                  <td className="file-table__td file-table__td--indexed file-table__td--right data-table__td data-table__td--right">
                    <div className="data-table__value data-table__value--right">{formatDate(file.indexed_at)}</div>
                  </td>
                  <td className="file-table__td file-table__td--modified file-table__td--right data-table__td data-table__td--right">
                    <div className="data-table__value data-table__value--right">{formatDate(file.modified_at)}</div>
                  </td>
                  <td className="file-table__td file-table__td--actions data-table__td" onClick={(e) => e.stopPropagation()}>
                    <div className="file-table__actions">
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-table__action-btn data-table__action-btn"
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
                            className="file-table__action-btn data-table__action-btn"
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
                          className="file-table__action-btn data-table__action-btn"
                          onClick={() => onReindex?.(file)}
                          disabled={offline || isReindexing}
                          title={isReindexing ? 'Reindex in progress' : 'Reindex file'}
                        >
                          <i
                            className={isReindexing ? 'ri-loader-4-line file-table__spinner' : 'ri-refresh-line'}
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
                          className="file-table__action-btn file-table__action-btn--danger data-table__action-btn"
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
        <div className="file-table__pagination">
          <span className="file-table__pagination-info">
            {offset + 1}–{Math.min(offset + limit, total)} of {total}
          </span>
          <div className="file-table__pagination-buttons">
            <button
              type="button"
              className="file-table__pagination-btn data-table__icon-btn"
              disabled={offline || !canPrev}
              onClick={() => onPageChange?.(Math.max(0, offset - limit))}
            >
              <i className="ri-arrow-left-s-line" aria-hidden style={{ fontSize: '1rem' }} />
            </button>
            <span className="file-table__pagination-page">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="file-table__pagination-btn data-table__icon-btn"
              disabled={offline || !canNext}
              onClick={() => onPageChange?.(offset + limit)}
            >
              <i className="ri-arrow-right-s-line" aria-hidden style={{ fontSize: '1rem' }} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
