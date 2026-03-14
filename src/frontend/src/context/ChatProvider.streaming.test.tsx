import { useState } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

let finishStream: (() => void) | null = null

vi.mock('../api', () => {
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
    getChat: vi.fn(),
    getSettings: vi.fn(async () => ({ enable_raw_output_control: false })),
    updateCurrentChat: vi.fn(async () => ({})),
    streamChat: vi.fn(async (_message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-1')
      callbacks.onToken?.('Hello')
      callbacks.onSources?.([])
      await Promise.resolve()

      await new Promise<void>((resolve) => {
        finishStream = () => {
          callbacks.onCleaned?.('Hello final')
          callbacks.onDone?.({ elapsed_seconds: 1.25, message_id: 321 })
          resolve()
        }
      })
    }),
  }
})

import { ChatProvider } from './ChatProvider'
import { streamChat } from '../api'
import { useChatContext } from './useChatContext'

function ChatProbe() {
  const { messages, isStreaming, error, sendMessage, setCurrentChatId } = useChatContext()
  const assistant = [...messages].reverse().find((m) => m.role === 'assistant')

  return (
    <div>
      <button onClick={() => setCurrentChatId('chat-existing')} type="button">
        BindChat
      </button>
      <button onClick={() => void sendMessage('test query')} type="button">
        Send
      </button>
      <div data-testid="streaming">{isStreaming ? 'yes' : 'no'}</div>
      <div data-testid="error">{error ?? ''}</div>
      <div data-testid="assistant-content">{assistant?.content ?? ''}</div>
      <div data-testid="assistant-streaming">{assistant?.isStreaming ? 'yes' : 'no'}</div>
      <div data-testid="assistant-id">{assistant?.id ?? ''}</div>
      <div data-testid="assistant-seconds">{assistant?.generationSeconds ?? ''}</div>
    </div>
  )
}

function Harness() {
  const [showProbe, setShowProbe] = useState(true)

  return (
    <ChatProvider>
      <button onClick={() => setShowProbe((prev) => !prev)} type="button">
        ToggleChat
      </button>
      {showProbe ? <ChatProbe /> : null}
    </ChatProvider>
  )
}

describe('ChatProvider streaming lifecycle', () => {
  afterEach(() => {
    vi.useRealTimers()
    cleanup()
  })

  it('keeps stream state coherent across unmount and remount', async () => {
    finishStream = null
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(screen.getByTestId('streaming')).toHaveTextContent('yes'))
    await waitFor(() => expect(screen.getByTestId('assistant-content')).toHaveTextContent('Hello'))
    expect(screen.getByTestId('assistant-streaming')).toHaveTextContent('yes')

    // Simulate navigation away from Chat screen.
    fireEvent.click(screen.getByRole('button', { name: 'ToggleChat' }))
    expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()

    // Stream completes while chat UI is hidden.
    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })

    // Simulate returning to Chat screen.
    fireEvent.click(screen.getByRole('button', { name: 'ToggleChat' }))

    await waitFor(() => expect(screen.getByTestId('streaming')).toHaveTextContent('no'))
    expect(screen.getByTestId('assistant-content')).toHaveTextContent('Hello final')
    expect(screen.getByTestId('assistant-streaming')).toHaveTextContent('no')
    expect(screen.getByTestId('assistant-id')).toHaveTextContent('321')
    expect(screen.getByTestId('assistant-seconds')).toHaveTextContent('1.25')
  })

  it('sends the active chat id when continuing or sending in an existing thread', async () => {
    finishStream = null
    const streamChatMock = vi.mocked(streamChat)
    streamChatMock.mockClear()
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'BindChat' }))
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][1]).toBe('chat-existing')

    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })
  })

})
