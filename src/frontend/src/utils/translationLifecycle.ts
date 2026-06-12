import type { TranslateJobStatus } from '../api'

export type TranslationJobStatus = TranslateJobStatus['status']

export const TRANSLATION_JOB_ACTIVE_STATUSES: TranslationJobStatus[] = ['queued', 'running']
export const TRANSLATION_JOB_RECOVERABLE_STATUSES: TranslationJobStatus[] = ['queued', 'running', 'stalled']

export function isTranslationJobActive(status: TranslationJobStatus | null | undefined): boolean {
  return !!status && TRANSLATION_JOB_ACTIVE_STATUSES.includes(status)
}

export function isTranslationJobRecoverable(status: TranslationJobStatus | null | undefined): boolean {
  return !!status && TRANSLATION_JOB_RECOVERABLE_STATUSES.includes(status)
}

export function saveSessionJson(key: string, value: unknown): void {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
  } catch {
    // ignore storage errors
  }
}

export function loadSessionJson<T>(key: string): T | null {
  try {
    const raw = window.sessionStorage.getItem(key)
    return raw ? JSON.parse(raw) as T : null
  } catch {
    return null
  }
}

export function clearSessionValue(key: string): void {
  try {
    window.sessionStorage.removeItem(key)
  } catch {
    // ignore storage errors
  }
}
