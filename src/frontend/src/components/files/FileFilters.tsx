/**
 * Informity AI — File filters
 * Search, category, file type, tag filters with removable chips.
 * Fully controlled: value from parent, onChange fires immediately (parent debounces).
 */
import { useState, useEffect } from 'react'
import { getFileTypes } from '../../api'
import { logApiError } from '../../utils/logApiError'
import './FileFilters.css'

interface FileTypeOption {
  id: string
  label: string
  extensions: string[]
}

interface FileFiltersState {
  search?: string
  extension?: string[]
}

interface FileFiltersProps {
  filters: FileFiltersState
  onChange?: (filters: FileFiltersState) => void
  disabled?: boolean
}

export function FileFilters({ filters, onChange, disabled = false }: FileFiltersProps) {
  const [fileTypes, setFileTypes] = useState<FileTypeOption[]>([])

  useEffect(() => {
    getFileTypes()
      .then((data) => setFileTypes(Array.isArray(data) ? (data as FileTypeOption[]) : []))
      .catch((err) => {
        logApiError(err, 'FileFilters.getFileTypes')
        setFileTypes([])
      })
  }, [])

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    const value = e.target.value
    onChange?.({ ...filters, search: value.length > 0 ? value : undefined })
  }

  const handleFileTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (disabled) return
    const value = e.target.value
    if (!value) {
      onChange?.({ ...filters, extension: undefined })
      return
    }
    const opt = fileTypes.find((ft) => ft.id === value)
    const extensions = opt?.extensions ?? []
    onChange?.({ ...filters, extension: extensions.length > 0 ? extensions : undefined })
  }

  const handleClearChip = (key: 'search' | 'extension') => {
    if (disabled) return
    const next = { ...filters }
    if (key === 'search') next.search = undefined
    else if (key === 'extension') next.extension = undefined
    onChange?.(next)
  }

  const handleClearAll = () => {
    if (disabled) return
    onChange?.({
      search: undefined,
      extension: undefined,
    })
  }

  const hasFilters =
    (filters.search?.trim?.()?.length ?? 0) > 0 ||
    (Array.isArray(filters.extension) && filters.extension.length > 0)

  const extMatch = (a: string[], b: string[]) => {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
    const sa = [...a].sort()
    const sb = [...b].sort()
    return sa.every((v, i) => v === sb[i])
  }
  const selectedFileTypeId =
    fileTypes.find(
      (ft) =>
        Array.isArray(filters.extension) &&
        filters.extension?.length > 0 &&
        extMatch(ft.extensions ?? [], filters.extension),
    )?.id ?? ''

  return (
    <div className="file-filters">
      <div className="file-filters__row">
        <div className="file-filters__search filter-search">
          <i className="ri-search-line file-filters__search-icon filter-search__icon" aria-hidden style={{ fontSize: '1rem' }} />
          <input
            type="text"
            className="file-filters__search-input filter-search__input"
            placeholder="Search filename or path..."
            value={filters.search ?? ''}
            onChange={handleSearchChange}
            disabled={disabled}
          />
        </div>
        <select
          className="file-filters__select"
          value={selectedFileTypeId}
          onChange={handleFileTypeChange}
          disabled={disabled}
        >
          <option value="">All File Categories</option>
          {fileTypes.map((ft) => (
            <option key={ft.id} value={ft.id}>
              {ft.label}
            </option>
          ))}
        </select>
        {hasFilters && (
          <button type="button" className="file-filters__clear-all filter-clear-btn" onClick={handleClearAll} disabled={disabled}>
            Clear all
          </button>
        )}
      </div>
      {hasFilters && (
        <div className="file-filters__chips">
          {filters.search?.trim?.() && (
            <span className="file-filters__chip">
              Search: {filters.search}
              <button type="button" onClick={() => handleClearChip('search')} disabled={disabled}>
                <i className="ri-close-line" aria-hidden style={{ fontSize: '0.75rem' }} />
              </button>
            </span>
          )}
          {Array.isArray(filters.extension) && filters.extension.length > 0 && (
            <span className="file-filters__chip">
              Type: {fileTypes.find((ft) => ft.id === selectedFileTypeId)?.label ?? filters.extension.join(', ')}
              <button type="button" onClick={() => handleClearChip('extension')} disabled={disabled}>
                <i className="ri-close-line" aria-hidden style={{ fontSize: '0.75rem' }} />
              </button>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
