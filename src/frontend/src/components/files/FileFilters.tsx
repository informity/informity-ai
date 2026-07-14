/**
 * Informity AI — File filters
 * Search, category, file type, tag filters with removable chips.
 * Fully controlled: value from parent, onChange fires immediately (parent debounces).
 */
import { useState, useEffect } from 'react'
import { getFileTypes } from '../../api'
import { logApiError } from '../../utils/logApiError'
import { sortFileTypeOptions } from '../../utils/fileTypeOrdering'
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
  searchMode?: 'standard' | 'semantic'
  onSearchModeChange?: (mode: 'standard' | 'semantic') => void
  semanticResultLimit?: number
  onSemanticResultLimitChange?: (limit: number) => void
}

const SEARCH_LIMIT_OPTIONS = [10, 20, 50]

export function FileFilters({
  filters,
  onChange,
  disabled = false,
  searchMode = 'standard',
  onSearchModeChange,
  semanticResultLimit = 20,
  onSemanticResultLimitChange,
}: FileFiltersProps) {
  const [fileTypes, setFileTypes] = useState<FileTypeOption[]>([])

  useEffect(() => {
    getFileTypes()
      .then((data) => setFileTypes(Array.isArray(data) ? sortFileTypeOptions(data as FileTypeOption[]) : []))
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

  const hasExtensionFilter = Array.isArray(filters.extension) && filters.extension.length > 0
  const isSemanticMode = searchMode === 'semantic'

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
            placeholder={isSemanticMode ? 'Search document meaning…' : 'Search filename or path…'}
            value={filters.search ?? ''}
            onChange={handleSearchChange}
            disabled={disabled}
          />
          {filters.search?.trim?.() && (
            <button
              type="button"
              className="filter-search__clear"
              onClick={() => handleClearChip('search')}
              disabled={disabled}
              aria-label="Clear search"
              title="Clear search"
            >
              <i className="ri-close-line" aria-hidden />
            </button>
          )}
        </div>
        <div className="file-filters__mode-stack">
          <div className="file-filters__mode-group" role="group" aria-label="Search mode">
            <button
              type="button"
              className={`file-filters__mode-btn${!isSemanticMode ? ' file-filters__mode-btn--active' : ''}`}
              aria-pressed={!isSemanticMode}
              onClick={() => onSearchModeChange?.('standard')}
              disabled={disabled}
            >
              Filename
            </button>
            <button
              type="button"
              className={`file-filters__mode-btn${isSemanticMode ? ' file-filters__mode-btn--active' : ''}`}
              aria-pressed={isSemanticMode}
              onClick={() => onSearchModeChange?.('semantic')}
              disabled={disabled}
            >
              Semantic
            </button>
          </div>
          <div className="file-filters__mode-slot">
            <select
              className={`file-filters__select file-filters__select--compact file-filters__mode-select${isSemanticMode ? '' : ' file-filters__mode-select--hidden'}`}
              value={semanticResultLimit}
              onChange={(event) => onSemanticResultLimitChange?.(Number(event.target.value))}
              disabled={disabled || !isSemanticMode}
              aria-label="Semantic result count"
              tabIndex={isSemanticMode ? 0 : -1}
              aria-hidden={!isSemanticMode}
            >
              {SEARCH_LIMIT_OPTIONS.map((limit) => (
                <option key={limit} value={limit}>
                  Top {limit}
                </option>
              ))}
            </select>
          </div>
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
      </div>
      {hasExtensionFilter && (
        <div className="file-filters__chips">
          {hasExtensionFilter && (
            <span className="file-filters__chip">
              Type: {fileTypes.find((ft) => ft.id === selectedFileTypeId)?.label ?? (filters.extension ?? []).join(', ')}
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
