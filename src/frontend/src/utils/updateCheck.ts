import { getUpdateCheck } from '../api'
export const UPDATE_CHECK_LAST_CHECKED_KEY = 'informity.update.last_checked_at'
export const UPDATE_CHECK_EVENT = 'informity:update-check'

export interface UpdateCheckResult {
  currentVersion: string
  latestVersion: string | null
  updateAvailable: boolean
  metadata: {
    version: string
    release_notes: string
    download_url: string
    published_at?: string | null
  } | null
  checkedAtIso: string
}

export async function checkForUpdates(): Promise<UpdateCheckResult> {
  const response = await getUpdateCheck()
  return {
    currentVersion: response.current_version,
    latestVersion: response.latest_version,
    updateAvailable: response.update_available,
    metadata: response.metadata,
    checkedAtIso: response.checked_at_iso,
  }
}

export function persistUpdateCheckResult(result: UpdateCheckResult): void {
  try {
    localStorage.setItem(UPDATE_CHECK_LAST_CHECKED_KEY, result.checkedAtIso)
  } catch {
    // Ignore storage errors in restricted environments.
  }
}

export function readLastCheckedAt(): string | null {
  try {
    const value = localStorage.getItem(UPDATE_CHECK_LAST_CHECKED_KEY)
    return value && value.trim() ? value : null
  } catch {
    return null
  }
}
