/**
 * Format bytes as human-readable size (e.g. "1.2 MB", "456 KB").
 */
export function formatFileSize(bytes: number | null | undefined): string {
  if (bytes == null || typeof bytes !== 'number' || bytes < 0) return '—'
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const val = bytes / Math.pow(1024, i)
  const suffix = units[Math.min(i, units.length - 1)]
  return val >= 10 || val % 1 === 0 ? `${Math.round(val)} ${suffix}` : `${val.toFixed(1)} ${suffix}`
}
