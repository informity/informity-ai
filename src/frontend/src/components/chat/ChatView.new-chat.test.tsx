import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatView } from './ChatView'
import { ChatProvider } from '../../context/ChatProvider'
import { ConfirmProvider } from '../../context/ConfirmProvider'

const {
  getChatMock,
  getCurrentChatMock,
  getMessageRawMock,
  getSettingsMock,
  streamChatMock,
  updateSettingsMock,
  updateCurrentChatMock,
} = vi.hoisted(() => ({
  getChatMock: vi.fn(),
  getCurrentChatMock: vi.fn(),
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
    getChat: getChatMock,
    getCurrentChat: getCurrentChatMock,
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
  })

  afterEach(() => {
    cleanup()
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
      expect(screen.getByText('Start a chat by typing a question below.')).toBeInTheDocument()
    })
    await waitFor(() => expect(getChatMock).toHaveBeenCalledTimes(1))
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
})
