import { ApiError } from '../api'

const NETWORK_ERROR_PATTERNS = [
  'failed to fetch',
  'fetch failed',
  'networkerror',
  'network error',
  'load failed',
]

export const SERVICE_UNAVAILABLE_MESSAGE =
  'Service unavailable. Start or restart Informity AI, then try again.'

export function isBackendConnectionError(error: unknown): boolean {
  if (error instanceof ApiError) return false
  if (!(error instanceof Error)) return false
  const message = (error.message || '').toLowerCase()
  return NETWORK_ERROR_PATTERNS.some((pattern) => message.includes(pattern))
}
