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
    listChatUploads: vi.fn(async () => ({
      attachments: [
        {
          upload_id: 'up-1',
          chat_id: 'chat-uploaded',
          file_id: 42,
          filename_at_upload: 'template.docx',
          size_bytes: 2048,
          state: 'ready',
        },
      ],
    })),
    uploadChatFile: vi.fn(async () => ({
      chat_id: 'chat-uploaded',
      upload_id: 'up-1',
      file_id: 42,
      filename: 'template.docx',
      state: 'ready',
    })),
    stopChatStream: vi.fn(async () => ({ stopped: true, status: 'stopped_now' })),
    updateCurrentChat: vi.fn(async () => ({})),
    streamChat: vi.fn(async (message, _chatId, callbacks) => {
      callbacks.onChatId?.('chat-1')
      callbacks.onRequestId?.('req-stream-1')
      callbacks.onToken?.('Hello')
      if (String(message || '').toLowerCase().includes('agent')) {
        callbacks.onStatus?.({ state: 'retrieving', message: 'Retrieving evidence...' })
        callbacks.onPlanStep?.({ step_id: 1, description: 'Analyzing the request', status: 'done' })
        callbacks.onPlanStep?.({ step_id: 2, description: 'Retrieving evidence', status: 'running' })
        callbacks.onPlanStep?.({ step_id: 3, description: 'Generating answer', status: 'running' })
        callbacks.onAgentEvent?.({
          kind: 'tool_call',
          status: 'running',
          title: 'Retrieving evidence',
          tool_name: 'search_vectors',
          query: 'agent query',
        })
        callbacks.onAgentEvent?.({
          kind: 'observation',
          status: 'done',
          title: 'Retrieving evidence',
          tool_name: 'search_vectors',
          query: 'agent query',
        })
      }
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
import { stopChatStream, streamChat } from '../api'
import { useChatContext } from './useChatContext'

function ChatProbe() {
  const {
    messages,
    isStreaming,
    error,
    sendMessage,
    continueLastScope,
    setCurrentChatId,
    stopStreaming,
    uploadFiles,
    chatUploads,
  } = useChatContext()
  const assistant = [...messages].reverse().find((m) => m.role === 'assistant')

  return (
    <div>
      <button onClick={() => setCurrentChatId('chat-existing')} type="button">
        BindChat
      </button>
      <button onClick={() => void sendMessage('test query')} type="button">
        Send
      </button>
      <button onClick={() => void sendMessage('agent query', { agentMode: true })} type="button">
        SendAgent
      </button>
      <button
        onClick={() => void uploadFiles([new File(['content'], 'template.docx')])}
        type="button"
      >
        Upload
      </button>
      <button onClick={() => void continueLastScope(undefined, { mode: 'assistant' })} type="button">
        ContinueAssistant
      </button>
      <button onClick={() => void stopStreaming()} type="button">
        Stop
      </button>
      <div data-testid="streaming">{isStreaming ? 'yes' : 'no'}</div>
      <div data-testid="error">{error ?? ''}</div>
      <div data-testid="assistant-content">{assistant?.content ?? ''}</div>
      <div data-testid="assistant-streaming">{assistant?.isStreaming ? 'yes' : 'no'}</div>
      <div data-testid="assistant-agent-mode">{assistant?.agentModeUsed ? 'yes' : 'no'}</div>
      <div data-testid="assistant-id">{assistant?.id ?? ''}</div>
      <div data-testid="assistant-seconds">{assistant?.generationSeconds ?? ''}</div>
      <div data-testid="assistant-status">{assistant?.streamStatusText ?? ''}</div>
      <div data-testid="assistant-plan-steps">{assistant?.streamPlanSteps?.length ?? 0}</div>
      <div data-testid="assistant-agent-events">{assistant?.streamAgentEvents?.length ?? 0}</div>
      <div data-testid="assistant-agent-event-0-title">{assistant?.streamAgentEvents?.[0]?.title ?? ''}</div>
      <div data-testid="assistant-agent-event-0-status">{assistant?.streamAgentEvents?.[0]?.status ?? ''}</div>
      <div data-testid="assistant-agent-event-1-title">{assistant?.streamAgentEvents?.[1]?.title ?? ''}</div>
      <div data-testid="assistant-agent-event-1-status">{assistant?.streamAgentEvents?.[1]?.status ?? ''}</div>
      <div data-testid="upload-count">{chatUploads.length}</div>
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

  it('streams structured agent events alongside plan steps', async () => {
    finishStream = null
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'SendAgent' }))
    await waitFor(() => expect(screen.getByTestId('assistant-plan-steps')).toHaveTextContent('3'))
    await waitFor(() => expect(screen.getByTestId('assistant-agent-events')).toHaveTextContent('2'))
    expect(screen.getByTestId('assistant-agent-mode')).toHaveTextContent('yes')
    expect(screen.getByTestId('assistant-status')).toHaveTextContent(/Retrieving evidence/)
    expect(screen.getByTestId('assistant-agent-event-0-title')).toHaveTextContent('Retrieving evidence')
    expect(screen.getByTestId('assistant-agent-event-0-status')).toHaveTextContent('running')
    expect(screen.getByTestId('assistant-agent-event-1-title')).toHaveTextContent('Retrieving evidence')
    expect(screen.getByTestId('assistant-agent-event-1-status')).toHaveTextContent('done')

    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })
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
    expect(streamChatMock.mock.calls[0][3]).toEqual(expect.objectContaining({ mode: 'researcher' }))
    expect(typeof streamChatMock.mock.calls[0][3]?.requestId).toBe('string')

    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })
  })

  it('propagates assistant mode through continueLastScope', async () => {
    finishStream = null
    const streamChatMock = vi.mocked(streamChat)
    streamChatMock.mockClear()
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'ContinueAssistant' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][3]).toEqual(expect.objectContaining({ mode: 'assistant' }))
    expect(typeof streamChatMock.mock.calls[0][3]?.requestId).toBe('string')

    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })
  })

  it('uses upload-resolved chat id for stream request and scoped upload ids', async () => {
    finishStream = null
    const streamChatMock = vi.mocked(streamChat)
    streamChatMock.mockClear()
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'Upload' }))
    await waitFor(() => expect(screen.getByTestId('upload-count')).toHaveTextContent('1'))

    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0][1]).toBe('chat-uploaded')
    expect(streamChatMock.mock.calls[0][3]).toEqual(expect.objectContaining({ scopedUploadIds: ['up-1'] }))

    await act(async () => {
      finishStream?.()
      await Promise.resolve()
    })
  })

  it('stops using request id when stream id is not available yet', async () => {
    finishStream = null
    const stopChatStreamMock = vi.mocked(stopChatStream)
    stopChatStreamMock.mockClear()
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(screen.getByTestId('streaming')).toHaveTextContent('yes'))
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => expect(stopChatStreamMock).toHaveBeenCalledTimes(1))
    expect(stopChatStreamMock.mock.calls[0]?.[0]).toBe('chat-1')
    expect(stopChatStreamMock.mock.calls[0]?.[1]).toEqual({
      streamId: null,
      requestId: 'req-stream-1',
    })
  })

  it('keeps stop behavior intact for agent-mode sends', async () => {
    finishStream = null
    const streamChatMock = vi.mocked(streamChat)
    const stopChatStreamMock = vi.mocked(stopChatStream)
    streamChatMock.mockClear()
    stopChatStreamMock.mockClear()
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'SendAgent' }))
    await waitFor(() => expect(streamChatMock).toHaveBeenCalled())
    expect(streamChatMock.mock.calls[0]?.[3]).toEqual(expect.objectContaining({ agentMode: true }))
    await waitFor(() => expect(screen.getByTestId('streaming')).toHaveTextContent('yes'))
    await waitFor(() => expect(screen.getByTestId('assistant-plan-steps')).toHaveTextContent('3'))

    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => expect(stopChatStreamMock).toHaveBeenCalledTimes(1))
    expect(stopChatStreamMock.mock.calls[0]?.[0]).toBe('chat-1')
    expect(stopChatStreamMock.mock.calls[0]?.[1]).toEqual({
      streamId: null,
      requestId: 'req-stream-1',
    })
    await waitFor(() => expect(screen.getByTestId('assistant-plan-steps')).toHaveTextContent('0'))
    expect(screen.getByTestId('assistant-status')).toHaveTextContent('')
  })

})
