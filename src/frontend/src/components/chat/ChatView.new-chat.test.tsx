import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatView } from './ChatView'
import { ChatProvider } from '../../context/ChatProvider'
import { ConfirmProvider } from '../../context/ConfirmProvider'
import { CHAT_FILE_SCOPE_MAP_STORAGE_KEY, CHAT_MODE_STORAGE_KEY, CHAT_SPECIALIZATION_ID_STORAGE_KEY, CHAT_TRANSLATION_REQUEST_STORAGE_KEY, FORCE_NEW_CHAT_KEY } from '../../utils/storageKeys'

const {
  getFilesMock,
  getChatMock,
  getCurrentChatMock,
  exportChatMarkdownMock,
  getSpecializationsMock,
  listChatUploadsMock,
  getMessageRawMock,
  translateChatMessageMock,
  getSettingsMock,
  streamChatMock,
  updateSettingsMock,
  updateCurrentChatMock,
} = vi.hoisted(() => ({
  getFilesMock: vi.fn(),
  getChatMock: vi.fn(),
  getCurrentChatMock: vi.fn(),
  exportChatMarkdownMock: vi.fn(),
  getSpecializationsMock: vi.fn(),
  listChatUploadsMock: vi.fn(),
  getMessageRawMock: vi.fn(),
  translateChatMessageMock: vi.fn(),
  getSettingsMock: vi.fn(),
  streamChatMock: vi.fn(),
  updateSettingsMock: vi.fn(),
  updateCurrentChatMock: vi.fn(),
}))

vi.mock('../../api', () => {
  class MockApiError extends Error {
    status: number
    detail: string

    constructor(message: string, status: number, detail: string) {
      super(message)
      this.name = 'ApiError'
      this.status = status
      this.detail = detail
    }
  }

  return {
    ApiError: MockApiError,
    getFiles: getFilesMock,
    getChat: getChatMock,
    getCurrentChat: getCurrentChatMock,
    exportChatMarkdown: exportChatMarkdownMock,
    getSpecializations: getSpecializationsMock,
    listChatUploads: listChatUploadsMock,
    getMessageRaw: getMessageRawMock,
    translateChatMessage: translateChatMessageMock,
    getSettings: getSettingsMock,
    streamChat: streamChatMock,
    updateSettings: updateSettingsMock,
    updateCurrentChat: updateCurrentChatMock,
  }
})

describe('ChatView new chat behavior', () => {
  function createDeferred<T>() {
    let resolve!: (value: T | PromiseLike<T>) => void
    let reject!: (reason?: unknown) => void
    const promise = new Promise<T>((res, rej) => {
      resolve = res
      reject = rej
    })
    return { promise, resolve, reject }
  }
  beforeAll(() => {
    Object.defineProperty(window.HTMLElement.prototype, 'scrollIntoView', {
      value: vi.fn(),
      writable: true,
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    getFilesMock.mockResolvedValue({ files: [] })
    listChatUploadsMock.mockResolvedValue({ chat_id: 'test-chat', attachments: [] })
    getSpecializationsMock.mockResolvedValue([])
    exportChatMarkdownMock.mockResolvedValue({
      chat_id: 'test-chat',
      scope: 'full_chat',
      filename: 'test-chat.md',
      markdown: '# Export',
      template: 'full_transcript',
      include_frontmatter: true,
    })
  })

  afterEach(() => {
    cleanup()
    window.localStorage.removeItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY)
    window.localStorage.removeItem(CHAT_MODE_STORAGE_KEY)
    window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
    window.localStorage.removeItem(FORCE_NEW_CHAT_KEY)
    window.sessionStorage.removeItem(FORCE_NEW_CHAT_KEY)
    window.sessionStorage.removeItem(CHAT_TRANSLATION_REQUEST_STORAGE_KEY)
  })

  it('does not reselect initial history chat after New Chat', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 101,
          role: 'assistant',
          content: 'Loaded history answer',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-history-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.getByText('Loaded history answer')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Start New Chat' }))

    await waitFor(() => {
      expect(screen.queryByText('Start a chat by typing a question below.')).not.toBeInTheDocument()
    })
    await waitFor(() => expect(getChatMock).toHaveBeenCalledTimes(1))
    const inputArea = screen.getByLabelText('Chat message input').closest('.chat-view__input-area')
    expect(inputArea).toHaveClass('chat-view__input-area--centered')
    expect(updateCurrentChatMock).toHaveBeenCalledWith(null)
  })

  it('honors initial history chat even when force-new-chat flag is set', async () => {
    window.localStorage.setItem(FORCE_NEW_CHAT_KEY, '1')
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 111,
          role: 'assistant',
          content: 'Loaded from history despite force-new-chat',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-history-force-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-history-force-1'))
    expect(await screen.findByText('Loaded from history despite force-new-chat')).toBeInTheDocument()
  })

  it('disables New Chat while a response is streaming', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-streaming-1')
      await new Promise(() => {})
    })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    const input = screen.getByLabelText('Chat message input')
    fireEvent.change(input, { target: { value: 'Start streaming answer' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => {
      const newChatButton = screen.getByRole('button', { name: 'Start New Chat' })
      expect(newChatButton).toBeDisabled()
      expect(newChatButton).toHaveAttribute('title', 'Stop current response to start a new chat')
    })
  })

  it('sends selected chat mode in stream payload', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Assistant' }))
    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Hello assistant mode' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ mode: 'assistant' })
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ requestId: expect.any(String) })
  })

  it('clears draft role when switching from assistant back to researcher', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getSpecializationsMock).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Assistant' }))
    expect(screen.getByRole('button', { name: 'Specialization: General Assistant' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Specialization: General Assistant' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Legal' }))
    expect(screen.getByRole('button', { name: 'Specialization: Legal' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Researcher' }))
    expect(screen.queryByRole('button', { name: /Role:/i })).not.toBeInTheDocument()
    expect(window.localStorage.getItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Assistant' }))
    expect(screen.getByRole('button', { name: 'Specialization: General Assistant' })).toBeInTheDocument()
  })

  it('accepts trailing inline selection on Tab without sending message', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    const input = screen.getByLabelText('Chat message input') as HTMLTextAreaElement
    fireEvent.change(input, { target: { value: 'hello world' } })
    input.focus()
    input.setSelectionRange(6, 11)

    const dispatchResult = fireEvent.keyDown(input, { key: 'Tab' })
    expect(dispatchResult).toBe(false)
    expect(input.selectionStart).toBe(11)
    expect(input.selectionEnd).toBe(11)
    expect(streamChatMock).not.toHaveBeenCalled()
  })

  it('keeps focus in composer on Tab when no inline selection is active', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    const input = screen.getByLabelText('Chat message input') as HTMLTextAreaElement
    fireEvent.change(input, { target: { value: 'hello world' } })
    input.focus()
    input.setSelectionRange(11, 11)

    const dispatchResult = fireEvent.keyDown(input, { key: 'Tab' })
    expect(dispatchResult).toBe(false)
    expect(document.activeElement).toBe(input)
    expect(streamChatMock).not.toHaveBeenCalled()
  })

  it('retains file scope when opening a scoped chat from history and uses it on send', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onDone?.({ elapsed_seconds: 0.2, message_id: 5001 })
    })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 102,
          role: 'assistant',
          content: 'History response',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })
    window.localStorage.setItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY, JSON.stringify({
      'chat-history-file-1': { fileId: 77, filename: 'The Ethics of Aristotle.txt' },
    }))

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-history-file-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-history-file-1'))
    expect(await screen.findByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Summarize this file' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ fileId: 77 })
  })

  it('clears file scope with x-out and sends next message against full corpus', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onDone?.({ elapsed_seconds: 0.2, message_id: 5002 })
    })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialScopedFile={{ fileId: 42, filename: 'focused-file.txt' }} />
        </ChatProvider>
      </ConfirmProvider>,
    )

    expect(await screen.findByTitle('focused-file.txt')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Scoped question' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ fileId: 42 })

    fireEvent.click(screen.getByRole('button', { name: 'Clear file scope' }))
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Clear file scope' })).not.toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Now use full corpus' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(2))
    expect(streamChatMock.mock.calls[1][3]).toMatchObject({ fileId: null })
  })

  it('keeps per-chat scope independent when clearing one chat and opening another', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockImplementation(async (chatId: string) => ({
      messages: [
        {
          id: chatId === 'chat-a' ? 301 : 302,
          role: 'assistant',
          content: chatId === 'chat-a' ? 'A response' : 'B response',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    }))

    window.localStorage.setItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY, JSON.stringify({
      'chat-a': { fileId: 11, filename: 'File-A.txt' },
      'chat-b': { fileId: 22, filename: 'File-B.txt' },
    }))

    const view = render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-a" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-a'))
    expect(screen.getByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()
    expect(screen.getByText('File-A.txt')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Clear file scope' }))
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Clear file scope' })).not.toBeInTheDocument()
    })

    view.rerender(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-b" />
        </ChatProvider>
      </ConfirmProvider>,
    )
    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-b'))
    expect(screen.getByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()
    expect(screen.getByText('File-B.txt')).toBeInTheDocument()
  })

  it('does not infer scope for corpus-wide history chats from sources alone', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 410,
          role: 'assistant',
          content: 'Corpus-wide response',
          sources: [
            {
              filename: 'LegacyFile.txt',
              path: '/corpus/LegacyFile.txt',
              chunk_preview: '...',
              relevance_score: 0.9,
            },
            {
              filename: 'LegacyFile.txt',
              path: '/corpus/LegacyFile.txt',
              chunk_preview: '...',
              relevance_score: 0.8,
            },
          ],
          retrieval_scope_kind: 'indexed_corpus',
          retrieval_scope_key: 'indexed_corpus',
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="legacy-chat-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('legacy-chat-1'))
    expect(getFilesMock).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Clear file scope' })).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Continue corpus-wide' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ fileId: null, mode: 'researcher', specializationId: null })
  })

  it('locks mode to Researcher when file scope is active', async () => {
    getSettingsMock.mockResolvedValue({
      enable_raw_output_control: false,
      enabled_specialization_ids: ['legal'],
    })
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialScopedFile={{ fileId: 13, filename: 'Scoped Doc.txt' }} />
        </ChatProvider>
      </ConfirmProvider>,
    )

    expect(await screen.findByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Researcher')
    expect(screen.getByRole('button', { name: 'Specialization: General Assistant' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    const assistantOption = screen.getByRole('menuitemradio', { name: 'Assistant' })
    expect(assistantOption).toBeDisabled()
  })

  it('hides role selector in Researcher mode without scoped/uploaded documents', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getSpecializationsMock).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Researcher')
    expect(screen.queryByRole('button', { name: /Role:/i })).not.toBeInTheDocument()
  })

  it('forces General role for corpus-wide Researcher first turn even when a role is stored', async () => {
    window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, 'researcher')
    window.localStorage.setItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY, 'legal')
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    streamChatMock.mockResolvedValue(undefined)

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getSpecializationsMock).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Corpus question' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ specializationId: null, mode: 'researcher' })
  })

  it('restores mode and locked role selection when opening a chat from history', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 701,
          role: 'user',
          content: 'Review this agreement',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 702,
          role: 'assistant',
          content: 'Here is a legal review.',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:05.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-role-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-role-1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Assistant')
    })
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toBeDisabled()

    const roleButton = screen.getByRole('button', { name: 'Specialization: Legal' })
    expect(roleButton).toBeDisabled()
  })

  it('keeps history role when specializations load after chat history', async () => {
    const specializationsDeferred = createDeferred<Array<{ id: string; name: string; description: string; icon: string }>>()
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockReturnValue(specializationsDeferred.promise)
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 801,
          role: 'user',
          content: 'Review this agreement',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 802,
          role: 'assistant',
          content: 'Here is a legal review.',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:05.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-role-delayed-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-role-delayed-1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Assistant')
    })
    expect(screen.getByRole('button', { name: 'Specialization: Legal' })).toBeInTheDocument()

    specializationsDeferred.resolve([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])

    await waitFor(() => expect(getSpecializationsMock).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: 'Specialization: Legal' })).toBeInTheDocument()
  })

  it('restores locked role from chat-level payload when message specialization_id is missing', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    getChatMock.mockResolvedValue({
      chat_mode: 'assistant',
      specialization_id: 'legal',
      messages: [
        {
          id: 901,
          role: 'user',
          content: 'Review this agreement',
          specialization_id: null,
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 902,
          role: 'assistant',
          content: 'Here is a legal review.',
          specialization_id: null,
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:05.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-role-fallback-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-role-fallback-1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Assistant')
    })
    expect(screen.getByRole('button', { name: 'Specialization: Legal' })).toBeDisabled()
  })

  it('locks role selector for history chat even when role is General', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    getChatMock.mockResolvedValue({
      chat_mode: 'assistant',
      specialization_id: null,
      messages: [
        {
          id: 9301,
          role: 'user',
          content: 'General assistant question',
          specialization_id: null,
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 9302,
          role: 'assistant',
          content: 'General assistant answer.',
          specialization_id: null,
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:02.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-general-locked-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-general-locked-1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Assistant')
    })
    expect(screen.getByRole('button', { name: 'Specialization: General Assistant' })).toBeDisabled()
  })

  it('keeps send and upload controls active for history chats while mode/role remain locked', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    streamChatMock.mockResolvedValue(undefined)
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 9401,
          role: 'user',
          content: 'Researcher history question',
          specialization_id: null,
          chat_mode: 'researcher',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 9402,
          role: 'assistant',
          content: 'Researcher history answer.',
          specialization_id: null,
          chat_mode: 'researcher',
          sources: [],
          created_at: '2026-02-23T12:00:02.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-history-researcher-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-history-researcher-1'))
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Upload files' })).toBeEnabled()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Follow-up in same history chat' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ mode: 'researcher' })
  })

  it('keeps history file-scope controls active while mode remains locked', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([])
    streamChatMock.mockResolvedValue(undefined)
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 9501,
          role: 'user',
          content: 'Scoped history question',
          specialization_id: null,
          chat_mode: 'researcher',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 9502,
          role: 'assistant',
          content: 'Scoped history answer.',
          specialization_id: null,
          chat_mode: 'researcher',
          sources: [],
          created_at: '2026-02-23T12:00:02.000Z',
        },
      ],
    })
    window.localStorage.setItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY, JSON.stringify({
      'chat-history-scope-1': { fileId: 77, filename: 'The Ethics of Aristotle.txt' },
    }))

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-history-scope-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-history-scope-1'))
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Clear file scope' })).toBeEnabled()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Scoped follow-up' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ fileId: 77, mode: 'researcher' })
  })

  it('locks assistant mode and legal role across send and history reopen', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_specializations: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-flow-locked-1')
      callbacks.onDone?.({ elapsed_seconds: 0.2, message_id: 9101, chat_mode: 'assistant' })
    })
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 9100,
          role: 'user',
          content: 'First legal assistant question',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 9101,
          role: 'assistant',
          content: 'Legal assistant response.',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:02.000Z',
        },
      ],
    })

    const firstRender = render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Assistant' }))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Specialization: General Assistant' })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Specialization: General Assistant' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Legal' }))
    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'First legal assistant question' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ mode: 'assistant', specializationId: 'legal' })

    firstRender.unmount()

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-flow-locked-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-flow-locked-1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Assistant')
    })
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Specialization: Legal' })).toBeDisabled()
  })

  it('preserves locked role on existing chat even when role is unchecked in settings', async () => {
    getSettingsMock.mockResolvedValue({
      enable_raw_output_control: false,
      enabled_specialization_ids: [],
    })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getSpecializationsMock.mockResolvedValue([
      {
        id: 'legal',
        name: 'Legal',
        description: 'Legal role',
        icon: 'ri-scales-3-line',
      },
    ])
    streamChatMock.mockResolvedValue(undefined)
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 9200,
          role: 'user',
          content: 'Follow-up legal question',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 9201,
          role: 'assistant',
          content: 'Legal response.',
          specialization_id: 'legal',
          chat_mode: 'assistant',
          sources: [],
          created_at: '2026-02-23T12:00:02.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-role-unchecked-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-role-unchecked-1'))
    const roleButton = await screen.findByRole('button', { name: 'Specialization: Legal' })
    expect(roleButton).toBeDisabled()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Follow-up legal question' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ mode: 'assistant', specializationId: 'legal' })
  })

  it('shows edit control only on the latest non-internal user message after streaming completes', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-edit-latest')
      callbacks.onDone?.({ elapsed_seconds: 0.1, message_id: Date.now(), completion_mode: 'complete', next_action: 'none' })
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'First prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Second prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(2))

    await waitFor(() => {
      const editButtons = screen.getAllByRole('button', { name: 'Edit message' })
      expect(editButtons.length).toBe(1)
    })
  })

  it('hides edit control while streaming is active', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({ messages: [] })
    const streamDeferred = createDeferred<void>()
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-edit-streaming')
      await streamDeferred.promise
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Streaming prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole('button', { name: 'Edit message' })).toBeNull()
    streamDeferred.resolve()
  })

  it('submits edited text through normal stream send flow', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockImplementation(async (chatId?: string) => {
      if (chatId === 'chat-edit-submit') {
        return {
          messages: [
            {
              id: 9300,
              role: 'user',
              content: 'Original prompt',
              sources: [],
              created_at: '2026-02-23T12:00:00.000Z',
            },
            {
              id: 9301,
              role: 'assistant',
              content: 'Original answer',
              sources: [],
              created_at: '2026-02-23T12:00:02.000Z',
            },
          ],
          chat_mode: 'researcher',
        }
      }
      return { messages: [] }
    })
    streamChatMock.mockImplementation(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-edit-submit')
      callbacks.onDone?.({ elapsed_seconds: 0.1, message_id: Date.now(), completion_mode: 'complete', next_action: 'none' })
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView />
        </ChatProvider>
      </ConfirmProvider>,
    )

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Original prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    fireEvent.change(screen.getByLabelText('Edit message'), { target: { value: 'Edited prompt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit edited message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(2))
    expect(streamChatMock.mock.calls[1]?.[0]).toBe('Edited prompt')
  })

  it('exports full chat markdown from header action', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 222,
          role: 'assistant',
          content: 'Ready to export',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-export-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-export-1'))
    // Export is now a dropdown: click the header button to open it, then click Markdown
    fireEvent.click(screen.getByRole('button', { name: 'Export chat' }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Markdown' })[0])
    await waitFor(() => expect(exportChatMarkdownMock).toHaveBeenCalledWith('chat-export-1', expect.objectContaining({
      scope: 'full_chat',
      template: 'full_transcript',
      includeFrontmatter: false,
    })))
  })

  it('exports assistant answer markdown from message action', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 333,
          role: 'assistant',
          content: 'Message export target',
          sources: [],
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-export-2" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-export-2'))
    // Per-message export is now a dropdown: click the Export icon, then click Markdown
    fireEvent.click(screen.getByRole('button', { name: 'Export' }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Markdown' })[0])
    await waitFor(() => expect(exportChatMarkdownMock).toHaveBeenCalledWith('chat-export-2', expect.objectContaining({
      scope: 'current_answer',
      messageId: 333,
      template: 'concise_summary',
      includeFrontmatter: false,
    })))
  })

  it('translates an assistant reply and stacks the translated reply under the source', async () => {
    const translateDeferred = createDeferred<{
      chat_id: string
      source_message_id: number
      translated_message_id: number
      target_language: string
      tone: string
      reused_existing_translation: boolean
      translated_message: {
        id: number
        role: 'assistant'
        content: string
        sources: []
        display_blocks: Array<{ type: 'text'; markdown: string }>
        chat_mode: 'assistant'
        translated_from_message_id: number
        translation_language: string
        translation_tone: string
        translation_source_hash: string
        translation_is_stale: boolean
        created_at: string
      }
    }>()
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 444,
          role: 'assistant',
          content: 'Hello world.',
          sources: [],
          chat_mode: 'assistant',
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })
    translateChatMessageMock.mockReturnValue(translateDeferred.promise)

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-translate-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-translate-1'))
    expect(await screen.findByText('Hello world.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Translate reply' }))

    await waitFor(() => {
      expect(translateChatMessageMock).toHaveBeenCalledWith(
        'chat-translate-1',
        444,
        { target_language: null, tone: null },
      )
    })
    expect(screen.getByText(/Translating to selected language/i)).toBeInTheDocument()

    translateDeferred.resolve({
      chat_id: 'chat-translate-1',
      source_message_id: 444,
      translated_message_id: 555,
      target_language: 'French',
      tone: 'natural',
      reused_existing_translation: false,
      translated_message: {
        id: 555,
        role: 'assistant',
        content: 'Bonjour le monde.',
        sources: [],
        display_blocks: [{ type: 'text', markdown: 'Bonjour le monde.' }],
        chat_mode: 'assistant',
        translated_from_message_id: 444,
        translation_language: 'French',
        translation_tone: 'natural',
        translation_source_hash: 'abc123',
        translation_is_stale: false,
        created_at: '2026-02-23T12:00:01.000Z',
      },
    })
    expect(await screen.findByText('Bonjour le monde.')).toBeInTheDocument()
    expect(screen.getByText('French')).toBeInTheDocument()
  })

  it('reloads translated replies from history and keeps them stacked under the source', async () => {
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 601,
          role: 'assistant',
          content: 'Original answer.',
          sources: [],
          chat_mode: 'assistant',
          created_at: '2026-02-23T12:00:00.000Z',
        },
        {
          id: 602,
          role: 'assistant',
          content: 'Réponse originale.',
          sources: [],
          display_blocks: [{ type: 'text', markdown: 'Réponse originale.' }],
          chat_mode: 'assistant',
          translated_from_message_id: 601,
          translation_language: 'French',
          translation_tone: 'natural',
          translation_source_hash: 'def456',
          translation_is_stale: false,
          created_at: '2026-02-23T12:00:01.000Z',
        },
      ],
    })

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-translate-history-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-translate-history-1'))
    expect(await screen.findByText('Original answer.')).toBeInTheDocument()
    expect(await screen.findByText('Réponse originale.')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Translate reply' })).toHaveLength(1)
    expect(screen.getByText('French')).toBeInTheDocument()
  })

  it('recovers an in-flight translation after reload using the persisted request', async () => {
    const translateDeferred = createDeferred<{
      chat_id: string
      source_message_id: number
      translated_message_id: number
      target_language: string
      tone: string
      reused_existing_translation: boolean
      translated_message: {
        id: number
        role: 'assistant'
        content: string
        sources: []
        display_blocks: Array<{ type: 'text'; markdown: string }>
        chat_mode: 'assistant'
        translated_from_message_id: number
        translation_language: string
        translation_tone: string
        translation_source_hash: string
        translation_is_stale: boolean
        created_at: string
      }
    }>()
    window.sessionStorage.setItem(CHAT_TRANSLATION_REQUEST_STORAGE_KEY, JSON.stringify({
      chatId: 'chat-translate-reload-1',
      sourceMessageId: 701,
      targetLanguage: 'French',
      tone: 'natural',
      startedAt: Date.now(),
    }))
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    streamChatMock.mockResolvedValue(undefined)
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 701,
          role: 'assistant',
          content: 'Recover me.',
          sources: [],
          chat_mode: 'assistant',
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })
    translateChatMessageMock.mockReturnValue(translateDeferred.promise)

    render(
      <ConfirmProvider>
        <ChatProvider>
          <ChatView initialChatId="chat-translate-reload-1" />
        </ChatProvider>
      </ConfirmProvider>,
    )

    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-translate-reload-1'))
    await waitFor(() => {
      expect(translateChatMessageMock).toHaveBeenCalledWith(
        'chat-translate-reload-1',
        701,
        { target_language: 'French', tone: 'natural' },
      )
    })
    expect(screen.getByText(/Translating to French/i)).toBeInTheDocument()

    translateDeferred.resolve({
      chat_id: 'chat-translate-reload-1',
      source_message_id: 701,
      translated_message_id: 702,
      target_language: 'French',
      tone: 'natural',
      reused_existing_translation: false,
      translated_message: {
        id: 702,
        role: 'assistant',
        content: 'Récupère-moi.',
        sources: [],
        display_blocks: [{ type: 'text', markdown: 'Récupère-moi.' }],
        chat_mode: 'assistant',
        translated_from_message_id: 701,
        translation_language: 'French',
        translation_tone: 'natural',
        translation_source_hash: 'recover-hash',
        translation_is_stale: false,
        created_at: '2026-02-23T12:00:01.000Z',
      },
    })

    expect(await screen.findByText('Récupère-moi.')).toBeInTheDocument()
    expect(screen.getByText('French')).toBeInTheDocument()
    expect(window.sessionStorage.getItem(CHAT_TRANSLATION_REQUEST_STORAGE_KEY)).toBeNull()
  })
})
