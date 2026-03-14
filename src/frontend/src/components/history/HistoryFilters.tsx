/**
 * Informity AI — History filters
 * Search by title or preview. Fully controlled (parent debounces).
 */
import './HistoryFilters.css'

interface HistoryFiltersState {
  search?: string
}

interface HistoryFiltersProps {
  filters: HistoryFiltersState
  onChange?: (filters: HistoryFiltersState) => void
  disabled?: boolean
}

export function HistoryFilters({ filters, onChange, disabled = false }: HistoryFiltersProps) {
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    const value = e.target.value
    onChange?.({ ...filters, search: value.trim() || undefined })
  }

  const handleClear = () => {
    if (disabled) return
    onChange?.({ ...filters, search: undefined })
  }

  const hasSearch = (filters.search?.trim?.()?.length ?? 0) > 0

  return (
    <div className="history-filters">
      <div className="history-filters__row">
        <div className="history-filters__search filter-search">
          <i className="ri-search-line history-filters__search-icon filter-search__icon" aria-hidden style={{ fontSize: '1rem' }} />
          <input
            type="text"
            className="history-filters__search-input filter-search__input"
            placeholder="Search by title or preview..."
            value={filters.search ?? ''}
            onChange={handleSearchChange}
            disabled={disabled}
          />
          {hasSearch && (
            <button
              type="button"
              className="history-filters__clear"
              onClick={handleClear}
              disabled={disabled}
              aria-label="Clear search"
            >
              <i className="ri-close-line" aria-hidden style={{ fontSize: '0.875rem' }} />
            </button>
          )}
        </div>
        {hasSearch && (
          <button type="button" className="history-filters__clear-all filter-clear-btn" onClick={handleClear} disabled={disabled}>
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
