import { ApiError } from '../api'

export function extractErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    const detail = `${error.detail || ''}`.trim()
    if (detail) return detail
    return `HTTP ${error.status}`
  }
  if (error instanceof Error) {
    const message = `${error.message || ''}`.trim()
    if (message) return message
  }
  return fallback
}
