import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { lintOutputFormatting, type OutputLintIssue } from '../../utils/outputLint'
import { ChatMessage } from './ChatMessage'
import type { DisplayBlock } from '../../types/api'

interface RenderingFixture {
  id: string
  content: string
  display_blocks: DisplayBlock[]
  expected_text: string[]
  expected_lint_issues: OutputLintIssue[]
}

const testFilePath = fileURLToPath(import.meta.url)
const fixturesPath = resolve(dirname(testFilePath), '../../../../../tests/fixtures/chat_rendering_fixtures.json')
const fixtures = JSON.parse(readFileSync(fixturesPath, 'utf-8')) as RenderingFixture[]

describe('ChatMessage rendering fixtures', () => {
  afterEach(() => {
    cleanup()
  })

  it('matches rendering fixtures and formatting lint expectations', () => {
    for (const fixture of fixtures) {
      const lintIssues = lintOutputFormatting(fixture.content)
      expect(lintIssues.sort(), `${fixture.id} lint`).toEqual([...fixture.expected_lint_issues].sort())

      const { container, unmount } = render(
        <ChatMessage
          role="assistant"
          content={fixture.content}
          displayBlocks={fixture.display_blocks}
          isStreaming={false}
        />,
      )

      const inner = container.querySelector('.chat-message__inner')
      expect(inner, `${fixture.id} inner`).not.toBeNull()
      for (const expected of fixture.expected_text) {
        expect(inner?.textContent || '', fixture.id).toContain(expected)
      }
      expect(inner, fixture.id).toMatchSnapshot()
      unmount()
    }
  })
})
