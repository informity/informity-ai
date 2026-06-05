import { MemoryRouter } from 'react-router-dom'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { getSettingsMock } = vi.hoisted(() => ({
  getSettingsMock: vi.fn(),
}))
const { getTranslateJobMock } = vi.hoisted(() => ({
  getTranslateJobMock: vi.fn(),
}))
const { streamTranslateJobMock } = vi.hoisted(() => ({
  streamTranslateJobMock: vi.fn(async () => undefined),
}))

vi.mock('../api', () => ({
  cancelTranslateJob: vi.fn(),
  createTranslateJob: vi.fn(),
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
  const { targetLanguage, tone } = useTranslateContext()
  return (
    <div>
      <span data-testid="language">{targetLanguage}</span>
      <span data-testid="tone">{tone}</span>
    </div>
  )
}

function renderPage() {
  return render(
    <MemoryRouter>
      <TranslateProvider>
        <TranslatePage />
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

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('German'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('formal'))
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

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('French'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('literal'))
  })

  it('falls back to a valid pinned language when a restored translation uses a removed language', async () => {
    sessionStorage.setItem('informity_active_translate_job', JSON.stringify({
      jobId: 'job-1',
      fileId: 42,
      fileName: 'consulting-agreement.docx',
      pageCount: 4,
      isUpload: false,
      targetLanguage: 'German',
      tone: 'formal',
    }))
    getSettingsMock.mockResolvedValue({
      translate_default_language: 'Spanish',
      translate_default_tone: 'natural',
      translate_pinned_languages: ['French', 'Spanish'],
    })
    getTranslateJobMock.mockResolvedValue({ status: 'done' })

    renderPage()

    await waitFor(() => expect(screen.getByTestId('language')).toHaveTextContent('French'))
    await waitFor(() => expect(screen.getByTestId('tone')).toHaveTextContent('formal'))
  })
})
