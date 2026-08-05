import { afterEach, describe, expect, it, vi } from 'vitest'
import { getUpdateCheck } from '../api'
import {
  checkForUpdates,
  persistUpdateCheckResult,
  readLastCheckedAt,
  UPDATE_CHECK_LAST_CHECKED_KEY,
} from './updateCheck'

vi.mock('../api', () => ({
  getUpdateCheck: vi.fn(),
}))

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('updateCheck', () => {
  it('returns update_available=true when latest version is newer', async () => {
    vi.mocked(getUpdateCheck).mockResolvedValue({
      current_version: '0.12.0',
      latest_version: '9.9.9',
      update_available: true,
      metadata: {
        version: '9.9.9',
        release_notes: 'Test release',
        download_url: 'https://www.informity.ai/download/Informity_AI_latest_aarch64.dmg',
        published_at: '2026-05-14T12:00:00.000Z',
      },
      checked_at_iso: '2026-05-14T12:00:00.000Z',
    })

    const result = await checkForUpdates()
    expect(result.currentVersion).toBe('0.12.0')
    expect(result.latestVersion).toBe('9.9.9')
    expect(result.updateAvailable).toBe(true)
    expect(result.metadata?.download_url).toContain('Informity_AI_latest_aarch64.dmg')
  })

  it('returns up_to_date when latest version equals current version', async () => {
    vi.mocked(getUpdateCheck).mockResolvedValue({
      current_version: '0.12.0',
      latest_version: '0.12.0',
      update_available: false,
      metadata: {
        version: '0.12.0',
        release_notes: '',
        download_url: 'https://www.informity.ai/download/Informity_AI_latest_aarch64.dmg',
        published_at: '2026-05-14T12:00:00.000Z',
      },
      checked_at_iso: '2026-05-14T12:00:00.000Z',
    })

    const result = await checkForUpdates()
    expect(result.updateAvailable).toBe(false)
    expect(result.latestVersion).toBe('0.12.0')
  })

  it('propagates backend update-check errors', async () => {
    vi.mocked(getUpdateCheck).mockRejectedValue(new Error('Update metadata request failed (404)'))

    await expect(checkForUpdates()).rejects.toThrow('Update metadata request failed (404)')
  })

  it('persists and reads last checked timestamp', () => {
    const result = {
      currentVersion: '0.12.0',
      latestVersion: '9.9.9',
      updateAvailable: true,
      metadata: {
        version: '9.9.9',
        release_notes: '',
        download_url: 'https://www.informity.ai/download/Informity_AI_latest_aarch64.dmg',
      },
      checkedAtIso: '2026-05-14T12:00:00.000Z',
    }

    persistUpdateCheckResult(result)

    expect(localStorage.getItem(UPDATE_CHECK_LAST_CHECKED_KEY)).toBe('2026-05-14T12:00:00.000Z')
    expect(readLastCheckedAt()).toBe('2026-05-14T12:00:00.000Z')
  })
})
