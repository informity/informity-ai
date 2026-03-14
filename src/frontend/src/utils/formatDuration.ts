/**
 * Format seconds as human-readable duration (e.g. "2m 30s", "1h 5m").
 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || typeof seconds !== 'number' || seconds < 0) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  const remainMins = mins % 60
  return remainMins > 0 ? `${hours}h ${remainMins}m` : `${hours}h`
}
