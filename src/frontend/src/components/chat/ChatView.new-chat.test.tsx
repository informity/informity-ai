import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatView } from './ChatView'
import { ChatProvider } from '../../context/ChatProvider'
import { ConfirmProvider } from '../../context/ConfirmProvider'
import { CHAT_FILE_SCOPE_MAP_STORAGE_KEY } from '../../utils/storageKeys'

  const {
    getFilesMock,
    getChatMock,
    getCurrentChatMock,
    getRolesMock,
    listChatUploadsMock,
    getMessageRawMock,
  getSettingsMock,
  streamChatMock,
  updateSettingsMock,
  updateCurrentChatMock,
} = vi.hoisted(() => ({
  getFilesMock: vi.fn(),
  getChatMock: vi.fn(),
  getCurrentChatMock: vi.fn(),
  getRolesMock: vi.fn(),
  listChatUploadsMock: vi.fn(),
  getMessageRawMock: vi.fn(),
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
    getRoles: getRolesMock,
    listChatUploads: listChatUploadsMock,
    getMessageRaw: getMessageRawMock,
    getSettings: getSettingsMock,
    streamChat: streamChatMock,
    updateSettings: updateSettingsMock,
    updateCurrentChat: updateCurrentChatMock,
  }
})

describe('ChatView new chat behavior', () => {
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
    getRolesMock.mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
    window.localStorage.removeItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY)
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

  it('recovers scope for legacy chats from single-file source history when map is missing', async () => {
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
          content: 'Legacy scoped response',
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
          created_at: '2026-02-23T12:00:00.000Z',
        },
      ],
    })
    getFilesMock.mockResolvedValue({
      files: [
        {
          id: 91,
          path: '/corpus/LegacyFile.txt',
          filename: 'LegacyFile.txt',
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
    await waitFor(() => expect(getFilesMock).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Continue on legacy file' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ fileId: 91 })
  })

  it('locks mode to Researcher when file scope is active', async () => {
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
          <ChatView initialScopedFile={{ fileId: 13, filename: 'Scoped Doc.txt' }} />
        </ChatProvider>
      </ConfirmProvider>,
    )

    expect(await screen.findByRole('button', { name: 'Clear file scope' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select chat mode' })).toHaveTextContent('Researcher')

    fireEvent.click(screen.getByRole('button', { name: 'Select chat mode' }))
    const assistantOption = screen.getByRole('menuitemradio', { name: 'Assistant' })
    expect(assistantOption).toBeDisabled()
  })
})
