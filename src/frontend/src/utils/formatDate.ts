/**
 * Format an ISO date string for table display (e.g. "Jan 20, 2025 09:12 AM").
 */
export function formatDate(isoString: string | null | undefined): string {
  if (!isoString) return '—'
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return '—'

  const month   = date.toLocaleString(undefined, { month: 'short' })
  const day     = date.getDate()
  const year    = date.getFullYear()
  const hours24 = date.getHours()
  const hours12 = hours24 === 0 ? 12 : (hours24 > 12 ? hours24 - 12 : hours24)
  const ampm   = hours24 < 12 ? 'AM' : 'PM'
  const hours   = String(hours12).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')

  return `${month} ${day}, ${year} ${hours}:${minutes} ${ampm}`
}
