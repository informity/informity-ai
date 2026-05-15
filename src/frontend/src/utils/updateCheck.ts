import { getHealth } from '../api'

export const UPDATE_METADATA_URL = 'https://raw.githubusercontent.com/informity/informity-ai/develop/latest.json'
export const UPDATE_CHECK_LAST_CHECKED_KEY = 'informity.update.last_checked_at'
export const UPDATE_CHECK_EVENT = 'informity:update-check'

const REQUEST_TIMEOUT_MS = 8000

export interface UpdateMetadata {
  version: string
  release_notes: string
  download_url: string
  published_at?: string
}

export interface UpdateCheckResult {
  currentVersion: string
  latestVersion: string | null
  updateAvailable: boolean
  metadata: UpdateMetadata | null
  checkedAtIso: string
}

interface HealthResponse {
  version?: string
}

function parseSemver(version: string): [number, number, number] | null {
  const raw = String(version || '').trim()
  const normalized = raw.startsWith('v') ? raw.slice(1) : raw
  const match = normalized.match(/^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$/)
  if (!match) return null
  return [Number.parseInt(match[1], 10), Number.parseInt(match[2], 10), Number.parseInt(match[3], 10)]
}

function compareSemver(a: string, b: string): number {
  const pa = parseSemver(a)
  const pb = parseSemver(b)
  if (!pa || !pb) return a.localeCompare(b)
  for (let i = 0; i < 3; i += 1) {
    if (pa[i] > pb[i]) return 1
    if (pa[i] < pb[i]) return -1
  }
  return 0
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { signal: controller.signal, cache: 'no-store' })
  } finally {
    window.clearTimeout(timer)
  }
}

export async function checkForUpdates(): Promise<UpdateCheckResult> {
  const checkedAtIso = new Date().toISOString()
  const health = (await getHealth()) as HealthResponse
  const currentVersion = String(health?.version || '').trim()
  if (!currentVersion) {
    throw new Error('Current app version is unavailable')
  }

  const response = await fetchWithTimeout(UPDATE_METADATA_URL, REQUEST_TIMEOUT_MS)
  if (!response.ok) {
    throw new Error(`Update metadata request failed (${response.status})`)
  }
  const json = await response.json() as Partial<UpdateMetadata>
  const latestVersion = String(json.version || '').trim()
  const releaseNotes = String(json.release_notes || '').trim()
  const downloadUrl = String(json.download_url || '').trim()
  const publishedAt = typeof json.published_at === 'string' ? json.published_at : undefined
  if (!latestVersion || !downloadUrl) {
    throw new Error('Update metadata is missing required fields')
  }

  const metadata: UpdateMetadata = {
    version: latestVersion,
    release_notes: releaseNotes,
    download_url: downloadUrl,
    published_at: publishedAt,
  }
  const updateAvailable = compareSemver(latestVersion, currentVersion) > 0

  return {
    currentVersion,
    latestVersion,
    updateAvailable,
    metadata,
    checkedAtIso,
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
