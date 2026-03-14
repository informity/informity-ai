/**
 * Informity AI — API error logging
 * Logs API errors to console for debugging. Use in .catch() instead of silent swallow.
 */
export function logApiError(err: unknown, context = ''): void {
  const prefix = context ? `[${context}]` : ''
  if (err instanceof Error && err.name === 'AbortError') return
  const msg = (err as { detail?: string; message?: string })?.detail ??
    (err instanceof Error ? err.message : String(err))
  console.warn(`${prefix} API error:`, msg, err)
}
