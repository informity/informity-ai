import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { formatMessageContent } from './formatMessageContent'

interface MalformedFixture {
  id: string
  raw: string
  backend_cleaned: string
  backend_reasoning_only: boolean
  frontend_cleaned: string
}

const testFilePath = fileURLToPath(import.meta.url)
const sharedFixturesPath = resolve(dirname(testFilePath), '../../../../tests/fixtures/malformed_output_fixtures.json')
const fixtures = JSON.parse(readFileSync(sharedFixturesPath, 'utf-8')) as MalformedFixture[]

describe('formatMessageContent', () => {
  it('matches shared malformed-output fixtures', () => {
    for (const fixture of fixtures) {
      const cleaned = formatMessageContent(fixture.raw)
      expect(cleaned, fixture.id).toBe(fixture.frontend_cleaned)
    }
  })

  it('stays in parity with backend cleaned output except reasoning-only fallback', () => {
    for (const fixture of fixtures) {
      const cleaned = formatMessageContent(fixture.raw)
      if (fixture.backend_reasoning_only) {
        expect(cleaned, fixture.id).toBe('')
        expect(fixture.backend_cleaned, fixture.id).not.toBe(cleaned)
        continue
      }
      expect(cleaned, fixture.id).toBe(fixture.backend_cleaned)
    }
  })

  it('preserves leading indentation for nested markdown bullets', () => {
    const raw = '- Parent\n  - Child\n    - Grandchild'
    const cleaned = formatMessageContent(raw)
    expect(cleaned).toBe(raw)
  })
})
