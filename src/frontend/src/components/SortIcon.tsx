/**
 * Informity AI — Sort icon for data table headers
 * Shared by FileTable and HistoryTable.
 */
interface SortIconProps {
  sort: string
  order: string
  column: string
}

export function SortIcon({ sort, order, column }: SortIconProps) {
  if (sort !== column) {
    return (
      <i
        className="ri-arrow-up-down-line data-table__sort-icon data-table__sort-icon--inactive"
        aria-hidden
        style={{ fontSize: '0.75rem' }}
      />
    )
  }
  return order === 'asc' ? (
    <i
      className="ri-arrow-up-line data-table__sort-icon data-table__sort-icon--active"
      aria-hidden
      style={{ fontSize: '0.75rem' }}
    />
  ) : (
    <i
      className="ri-arrow-down-line data-table__sort-icon data-table__sort-icon--active"
      aria-hidden
      style={{ fontSize: '0.75rem' }}
    />
  )
}
