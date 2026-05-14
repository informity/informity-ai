import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  checkForUpdates,
  persistUpdateCheckResult,
  readLastCheckedAt,
  UPDATE_CHECK_LAST_CHECKED_KEY,
  UPDATE_METADATA_URL,
} from './updateCheck'

vi.mock('../api', () => ({
  getHealth: vi.fn(async () => ({ version: '0.12.0' })),
}))

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('updateCheck', () => {
  it('returns update_available=true when latest version is newer', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      expect(url).toBe(UPDATE_METADATA_URL)
      return new Response(
        JSON.stringify({
          version: '9.9.9',
          release_notes: 'Test release',
          download_url: 'https://www.informity.ai/download/Informity_AI_latest_aarch64.dmg',
        }),
        { status: 200 },
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await checkForUpdates()
    expect(result.currentVersion).toBe('0.12.0')
    expect(result.latestVersion).toBe('9.9.9')
    expect(result.updateAvailable).toBe(true)
    expect(result.metadata?.download_url).toContain('Informity_AI_latest_aarch64.dmg')
  })

  it('returns up_to_date when latest version equals current version', async () => {
    const fetchMock = vi.fn(async () => new Response(
      JSON.stringify({
        version: '0.12.0',
        release_notes: '',
        download_url: 'https://www.informity.ai/download/Informity_AI_latest_aarch64.dmg',
      }),
      { status: 200 },
    ))
    vi.stubGlobal('fetch', fetchMock)

    const result = await checkForUpdates()
    expect(result.updateAvailable).toBe(false)
    expect(result.latestVersion).toBe('0.12.0')
  })

  it('throws when metadata endpoint is not successful', async () => {
    const fetchMock = vi.fn(async () => new Response('missing', { status: 404 }))
    vi.stubGlobal('fetch', fetchMock)

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
