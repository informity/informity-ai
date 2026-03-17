/**
 * Informity AI — File table
 * Sortable columns, row selection, multi-select, pagination.
 */
import { useState, useRef, useCallback } from 'react'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatDate } from '../../utils/formatDate'
import { SortIcon } from '../SortIcon'
import { StateMessage } from '../StateMessage'
import type { IndexedFile } from '../../types/api'
import './FileTable.css'

const EXTENSION_ICONS: Record<string, string> = {
  pdf:  'ri-file-text-line',
  docx: 'ri-file-text-line',
  doc:  'ri-file-text-line',
  txt:  'ri-file-text-line',
  md:   'ri-file-text-line',
  rst:  'ri-file-text-line',
  log:  'ri-file-text-line',
  xlsx: 'ri-file-excel-2-line',
  xls:  'ri-file-excel-2-line',
  csv:  'ri-file-excel-2-line',
  pptx: 'ri-file-text-line',
  ppt:  'ri-file-text-line',
  html: 'ri-code-s-line',
  htm:  'ri-code-s-line',
}

function getFileIcon(extension: string | undefined): string {
  const ext = (extension || '').toLowerCase().replace(/^\./, '')
  return EXTENSION_ICONS[ext] || 'ri-file-line'
}

function formatCategory(cat: string | undefined): string {
  if (!cat) return '—'
  return String(cat).charAt(0).toUpperCase() + String(cat).slice(1)
}

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
  onSelectFile?: (file: IndexedFile) => void
  onReindex?: (file: IndexedFile) => void
  onRemove?: (file: IndexedFile, e: React.MouseEvent) => void
  selectedFileId?: number | null
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
  onSelectFile,
  onReindex,
  onRemove,
  selectedFileId = null,
  offline = false,
}: FileTableProps) {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const lastClickedIndexRef = useRef(-1)

  const handleHeaderClick = useCallback(
    (col: string) => {
      if (offline) return
      if (!SORT_COLUMNS.includes(col)) return
      const nextOrder = sort === col && order === 'desc' ? 'asc' : 'desc'
      onSortChange?.(col as SortColumn, sort === col ? (nextOrder as SortOrder) : 'desc')
    },
    [offline, sort, order, onSortChange],
  )

  const handleRowClick = useCallback(
    (file: IndexedFile, index: number, e: React.MouseEvent) => {
      if (offline) return
      if (e.metaKey || e.ctrlKey) {
        setSelectedIds((prev) => {
          const next = new Set(prev)
          if (next.has(file.id)) next.delete(file.id)
          else next.add(file.id)
          return next
        })
        lastClickedIndexRef.current = index
        return
      }
      if (e.shiftKey) {
        const start = Math.min(lastClickedIndexRef.current, index)
        const end = Math.max(lastClickedIndexRef.current, index)
        setSelectedIds((prev) => {
          const next = new Set(prev)
          for (let i = start; i <= end; i++) {
            const f = files[i]
            if (f) next.add(f.id)
          }
          return next
        })
        return
      }
      lastClickedIndexRef.current = index
      onSelectFile?.(file)
    },
    [offline, files, onSelectFile],
  )

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
              <th className="file-table__th file-table__th--icon data-table__th data-table__th--icon" />
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
                  sort === 'category' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('category')}
              >
                Category
                <SortIcon sort={sort} order={order} column="category" />
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
                className={`file-table__th file-table__th--modified file-table__th--sortable file-table__th--right data-table__th data-table__th--sortable data-table__th--right ${
                  sort === 'modified_at' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('modified_at')}
              >
                Modified
                <SortIcon sort={sort} order={order} column="modified_at" />
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
              <th className="file-table__th file-table__th--actions data-table__th" />
            </tr>
          </thead>
          <tbody>
            {files.map((file, index) => {
              const iconClass = getFileIcon(file.extension)
              const isSelected = selectedFileId === file.id || selectedIds.has(file.id)

              return (
                <tr
                  key={file.id}
                  className={`file-table__row data-table__row ${isSelected ? 'file-table__row--selected' : ''}`}
                  onClick={(e) => !offline && handleRowClick(file, index, e)}
                >
                  <td className="file-table__td file-table__td--icon data-table__td data-table__td--icon">
                    <div className="file-table__icon data-table__icon">
                      <i className={iconClass} aria-hidden style={{ fontSize: '1rem' }} />
                    </div>
                  </td>
                  <td className="file-table__td file-table__td--filename data-table__td">
                    <span className="file-table__filename-text" title={file.filename || '—'}>
                      {file.filename || '—'}
                    </span>
                  </td>
                  <td className="file-table__td file-table__td--category data-table__td">
                    <span className="file-table__category-badge data-table__badge">{formatCategory(file.category)}</span>
                  </td>
                  <td className="file-table__td file-table__td--size file-table__td--right data-table__td data-table__td--right">
                    {formatFileSize(file.size_bytes)}
                  </td>
                  <td className="file-table__td file-table__td--modified file-table__td--right data-table__td data-table__td--right">{formatDate(file.modified_at)}</td>
                  <td className="file-table__td file-table__td--indexed file-table__td--right data-table__td data-table__td--right">{formatDate(file.indexed_at)}</td>
                  <td className="file-table__td file-table__td--actions data-table__td" onClick={(e) => e.stopPropagation()}>
                    <div className="file-table__actions">
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-table__action-btn data-table__action-btn"
                          onClick={() => onSelectFile?.(file)}
                          disabled={offline}
                          title="View details"
                        >
                          <i className="ri-eye-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">View details</span>
                      </span>
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="file-table__action-btn data-table__action-btn"
                          onClick={() => onReindex?.(file)}
                          disabled={offline}
                          title="Reindex file"
                        >
                          <i className="ri-refresh-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">Reindex file</span>
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
