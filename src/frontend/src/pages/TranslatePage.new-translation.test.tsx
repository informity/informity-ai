import { MemoryRouter, useNavigate } from 'react-router-dom'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useEffect } from 'react'

const { getSettingsMock } = vi.hoisted(() => ({
  getSettingsMock: vi.fn(),
}))
const { createTranslateJobMock } = vi.hoisted(() => ({
  createTranslateJobMock: vi.fn(),
}))
const { getTranslateJobMock } = vi.hoisted(() => ({
  getTranslateJobMock: vi.fn(),
}))
const { streamTranslateJobMock } = vi.hoisted(() => ({
  streamTranslateJobMock: vi.fn(async () => undefined),
}))

vi.mock('../api', () => ({
  cancelTranslateJob: vi.fn(),
  createTranslateJob: createTranslateJobMock,
  deleteTranslateUpload: vi.fn(),
  estimateTranslateJob: vi.fn(async () => ({ estimated_minutes: 1, exceeds_soft_limit: false })),
  getSettings: getSettingsMock,
  getTranslateJob: getTranslateJobMock,
  streamTranslateJob: streamTranslateJobMock,
  uploadTranslateFile: vi.fn(),
}))

vi.mock('../context/useBackendStatus', () => ({
  useBackendStatus: () => ({ offline: false }),
}))

vi.mock('../context/useChatContext', () => ({
  useChatContext: () => ({ isStreaming: false, stopStreaming: vi.fn() }),
}))

import { TranslateProvider } from '../context/TranslateProvider'
import { useTranslateContext } from '../context/useTranslateContext'
import { TranslatePage } from './TranslatePage'

function TranslateProbe() {
  const { targetLanguage, tone, resultLanguage } = useTranslateContext()
  return (
    <div>
      <span data-testid="language">{targetLanguage}</span>
      <span data-testid="tone">{tone}</span>
      <span data-testid="result-language">{resultLanguage ?? ''}</span>
    </div>
  )
}

function TranslateHarness() {
  const { setFile, setTargetLanguage } = useTranslateContext()

  useEffect(() => {
    setFile({ id: 42, name: 'consulting-agreement.docx', pageCount: 1, isUpload: false })
  }, [setFile])

  return (
    <div>
      <button type="button" onClick={() => setTargetLanguage('French')}>Force French</button>
    </div>
  )
}

function TranslateRouteHarness() {
  const navigate = useNavigate()

  return (
    <div>
      <button
        type="button"
        onClick={() => navigate('/translate', {
          state: {
            scopedFileId: 84,
            scopedFileName: 'updated-contract.pdf',
          },
        })}
      >
        Select updated file
      </button>
    </div>
  )
}

function renderPage(includeHarness = true) {
  return render(
        <MemoryRouter>
          <TranslateProvider>
            <TranslatePage />
            {includeHarness && <TranslateHarness />}
            <TranslateRouteHarness />
            <TranslateProbe />
          </TranslateProvider>
        </MemoryRouter>,
  )
}

describe('TranslatePage new translation reset', () => {
  afterEach(() => {
    cleanup()
    sessionStorage.clear()
    getSettingsMock.mockReset()
    createTranslateJobMock.mockReset()
    getTranslateJobMock.mockReset()
    streamTranslateJobMock.mockReset()
    streamTranslateJobMock.mockResolvedValue(undefined)
  })

  it('resets language and tone to the configured defaults when starting a new translation', async () => {
    getSettingsMock.mockResolvedValue({
      translate_default_language: 'German',
      translate_default_tone: 'formal',
      translate_pinned_languages: ['Spanish', 'German', 'French'],
    })
    renderPage(false)

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('German'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('formal'))

    fireEvent.click(screen.getByRole('button', { name: 'Tone' }))
    fireEvent.click(screen.getByRole('button', { name: 'Natural' }))
    fireEvent.click(screen.getByRole('button', { name: 'German' }))
    fireEvent.click(screen.getByRole('button', { name: 'Spanish' }))

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('Spanish'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('natural'))

    fireEvent.click(screen.getByRole('button', { name: 'New Translation' }))

    await waitFor(() => expect(screen.getByTestId('result-language')).toHaveTextContent(''))
    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('German'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('formal'))
    expect(screen.queryByTestId('translate-run-language-0')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Spanish' })).toBeNull()
  })

  it('shows the configured primary language in the translation dropdown alongside additional languages alphabetically', async () => {
    getSettingsMock.mockResolvedValue({
      translate_default_language: 'German',
      translate_default_tone: 'formal',
      translate_pinned_languages: ['Spanish', 'French'],
    })
    renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('German'))

    fireEvent.click(screen.getByRole('button', { name: 'German' }))
    const menu = screen.getByRole('menu')
    const optionLabels = within(menu)
      .getAllByRole('button')
      .map((button) => button.textContent?.trim())

    expect(optionLabels).toEqual(['French', 'German', 'Spanish'])
  })

  it('refreshes new-translation defaults when settings change after mount', async () => {
    getSettingsMock
      .mockResolvedValueOnce({
        translate_default_language: 'German',
        translate_default_tone: 'formal',
        translate_pinned_languages: ['Spanish', 'German', 'French'],
      })
      .mockResolvedValueOnce({
        translate_default_language: 'French',
        translate_default_tone: 'literal',
        translate_pinned_languages: ['Spanish', 'French'],
      })

    renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('German'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('formal'))

    fireEvent.click(screen.getByRole('button', { name: 'Tone' }))
    fireEvent.click(screen.getByRole('button', { name: 'Natural' }))
    fireEvent.click(screen.getByRole('button', { name: 'German' }))
    fireEvent.click(screen.getByRole('button', { name: 'Spanish' }))

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('Spanish'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('natural'))

    fireEvent.click(screen.getByRole('button', { name: 'New Translation' }))

    await waitFor(() => expect(screen.getByTestId('result-language')).toHaveTextContent(''))
    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('French'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('literal'))
    expect(screen.queryByTestId('translate-run-language-0')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Spanish' })).toBeNull()
  })

  it('restores the completed translation footer language after a reload even if settings change later', async () => {
    getSettingsMock
      .mockResolvedValueOnce({
        translate_default_language: 'Spanish',
        translate_default_tone: 'natural',
        translate_pinned_languages: ['French', 'Spanish'],
      })
      .mockResolvedValueOnce({
        translate_default_language: 'Portuguese',
        translate_default_tone: 'natural',
        translate_pinned_languages: ['French', 'Portuguese', 'Spanish'],
      })
    createTranslateJobMock.mockResolvedValue({ job_id: 'job-1' })
    streamTranslateJobMock.mockImplementationOnce((async (...args: unknown[]) => {
      const callbacks = args[1] as {
        onSectionsReady?: (count: number) => void
        onSectionDone?: (section: { section_index: number; section_title: string | null; text: string }) => void
        onJobDone?: (completed: number, failed: number, elapsedSeconds?: number | null) => void
      }
      callbacks.onSectionsReady?.(1)
      callbacks.onSectionDone?.({ section_index: 0, section_title: null, text: 'hola' })
      callbacks.onJobDone?.(1, 0, 7)
    }) as never)

    const { unmount } = renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('Spanish'))
    fireEvent.click(screen.getByRole('button', { name: /Translate/i }))
    await waitFor(() => expect(screen.getByTestId('translate-run-language-0')).toHaveTextContent('Spanish'))

    unmount()
    cleanup()

    renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('Portuguese'))
    await waitFor(() => expect(screen.getByTestId('translate-run-language-0')).toHaveTextContent('Spanish'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('natural'))
    await waitFor(() => expect(screen.getByText('0m 7s')).toBeInTheDocument())
  })

  it('clears the previous translation screen when a different file is selected from route state', async () => {
    getSettingsMock.mockResolvedValue({
      translate_default_language: 'German',
      translate_default_tone: 'formal',
      translate_pinned_languages: ['Spanish', 'German', 'French'],
    })
    sessionStorage.setItem(
      'informity_completed_translate_runs',
      JSON.stringify([{
        sections: [{ section_index: 0, section_title: null, text: 'Old translation result' }],
        language: 'Spanish',
        tone: 'natural',
        completedAt: Date.now(),
        totalSections: 1,
        elapsedSeconds: 4,
        fileLabel: 'old-file.pdf',
      }]),
    )

    renderPage()

    await waitFor(() => expect(screen.getByText('Old translation result')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Select updated file' }))

    await waitFor(() => expect(screen.getByText('updated-contract.pdf')).toBeInTheDocument())
    await waitFor(() => expect(screen.queryByText('Old translation result')).toBeNull())
  })

  it('preserves the saved translation language in the footer after live target settings change', async () => {
    getSettingsMock.mockResolvedValue({
      translate_default_language: 'Spanish',
      translate_default_tone: 'natural',
      translate_pinned_languages: ['French', 'Spanish'],
    })
    createTranslateJobMock.mockResolvedValue({ job_id: 'job-1' })
    streamTranslateJobMock.mockImplementationOnce((async (...args: unknown[]) => {
      const callbacks = args[1] as {
        onSectionsReady?: (count: number) => void
        onSectionDone?: (section: { section_index: number; section_title: string | null; text: string }) => void
        onJobDone?: (completed: number, failed: number, elapsedSeconds?: number | null) => void
      }
      callbacks.onSectionsReady?.(1)
      callbacks.onSectionDone?.({ section_index: 0, section_title: null, text: 'hola' })
      callbacks.onJobDone?.(1, 0, 7)
    }) as never)

    renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('Spanish'))
    fireEvent.click(screen.getByRole('button', { name: /Translate/i }))

    await waitFor(() => expect(screen.getByTestId('result-language')).toHaveTextContent('Spanish'))
    await waitFor(() => expect(screen.getByTestId('translate-run-language-0')).toHaveTextContent('Spanish'))
    expect(screen.getByRole('button', { name: 'Spanish' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Force French' }))

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('French'))
    expect(screen.getByTestId('result-language')).toHaveTextContent('Spanish')
    expect(screen.getByTestId('translate-run-language-0')).toHaveTextContent('Spanish')

    const languageButton = screen.getByRole('button', { name: 'French' })
    expect(languageButton).not.toBeDisabled()
    fireEvent.click(languageButton)
    expect(screen.getByRole('menu')).toBeInTheDocument()
  })
})
