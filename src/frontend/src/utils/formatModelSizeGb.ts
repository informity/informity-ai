/**
 * Format model size as decimal gigabytes (GB) with two fractional digits.
 */
export function formatModelSizeGb(bytes: number | null | undefined): string {
  if (!Number.isFinite(bytes) || (bytes ?? 0) <= 0) return '--'
  const gb = Number(bytes) / 1_000_000_000
  return `${gb.toFixed(2)} GB`
}
