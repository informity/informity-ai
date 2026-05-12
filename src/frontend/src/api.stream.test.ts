import { describe, expect, it, vi } from 'vitest'
import { streamChat } from './api'
import type { StreamDonePayload } from './types/api'

function makeSsePayload(events: Array<{ event: string; data: string }>): string {
  return events.map((e) => `event: ${e.event}\ndata: ${e.data}\n\n`).join('')
}

describe('streamChat SSE contract', () => {
  it('enforces event ordering invariants and forwards done payload', async () => {
    const payload = makeSsePayload([
      { event: 'chat', data: JSON.stringify({ chat_id: 'chat-1' }) },
      {
        event: 'status',
        data: JSON.stringify({
          state: 'retrieving',
          message: 'Searching for relevant information...',
          section_progress: {
            completed: ['## Scope'],
            remaining: ['## Method'],
            total: 2,
          },
        }),
      },
      { event: 'token', data: 'A' },
      { event: 'sources', data: JSON.stringify([{ filename: 'f', path: '/p' }]) },
      { event: 'token', data: 'AB' }, // allowed before cleaned
      { event: 'cleaned', data: 'Final answer' },
      { event: 'token', data: 'B' }, // ignored after sources/cleaned
      {
        event: 'done',
        data: JSON.stringify({
          elapsed_seconds: 1.2,
          message_id: 123,
          completion_mode: 'partial',
          timeout_occurred: true,
        }),
      },
      { event: 'token', data: 'C' }, // ignored after done
    ])

    const fetchMock = vi.fn(async () => {
      const body = new TextEncoder().encode(payload)
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    const tokens: string[] = []
    let cleaned = ''
    let sourceCount = 0
    let done: StreamDonePayload = {}
    let statusMessage = ''
    let statusProgressTotal = 0

    await streamChat('hello', null, {
      onToken: (token) => tokens.push(token),
      onSources: (sources) => {
        sourceCount = sources.length
      },
      onCleaned: (value) => {
        cleaned = value
      },
      onDone: (data) => {
        done = data ?? {}
      },
      onStatus: (status) => {
        statusMessage = status?.message ?? ''
        statusProgressTotal = status?.section_progress?.total ?? 0
      },
    })

    expect(tokens).toEqual(['A', 'AB'])
    expect(cleaned).toBe('Final answer')
    expect(sourceCount).toBe(1)
    expect(done.message_id).toBe(123)
    expect(done.completion_mode).toBe('partial')
    expect(statusMessage).toBe('Searching for relevant information...')
    expect(statusProgressTotal).toBe(2)

    vi.unstubAllGlobals()
  })

  it('sends researcher mode by default and allows assistant override', async () => {
    const payload = makeSsePayload([
      { event: 'chat', data: JSON.stringify({ chat_id: 'chat-2' }) },
      { event: 'done', data: JSON.stringify({ elapsed_seconds: 0.1 }) },
    ])

    const fetchMock = vi.fn(async () => {
      const body = new TextEncoder().encode(payload)
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    await streamChat('hello', null, {})
    await streamChat('hello', null, {}, { mode: 'assistant' })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit | undefined]>
    const firstBody = JSON.parse(String(calls[0]?.[1]?.body || '{}'))
    const secondBody = JSON.parse(String(calls[1]?.[1]?.body || '{}'))
    expect(firstBody.mode).toBe('researcher')
    expect(secondBody.mode).toBe('assistant')
    expect(firstBody.scoped_file_ids).toBeNull()
    expect(secondBody.scoped_file_ids).toBeNull()

    vi.unstubAllGlobals()
  })

  it('maps file scope to scoped_file_ids payload', async () => {
    const payload = makeSsePayload([
      { event: 'chat', data: JSON.stringify({ chat_id: 'chat-3' }) },
      { event: 'done', data: JSON.stringify({ elapsed_seconds: 0.1 }) },
    ])

    const fetchMock = vi.fn(async () => {
      const body = new TextEncoder().encode(payload)
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    await streamChat('hello', null, {}, { fileId: 42 })
    const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit | undefined]>
    const body = JSON.parse(String(calls[0]?.[1]?.body || '{}'))
    expect(body.scoped_file_ids).toEqual([42])
    expect(body.file_id).toBeUndefined()

    vi.unstubAllGlobals()
  })

  it('maps uploaded attachment scope to scoped_upload_ids payload', async () => {
    const payload = makeSsePayload([
      { event: 'chat', data: JSON.stringify({ chat_id: 'chat-4' }) },
      { event: 'done', data: JSON.stringify({ elapsed_seconds: 0.1 }) },
    ])

    const fetchMock = vi.fn(async () => {
      const body = new TextEncoder().encode(payload)
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    await streamChat('compare these files', null, {}, { scopedUploadIds: ['up-1', 'up-2'] })
    const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit | undefined]>
    const body = JSON.parse(String(calls[0]?.[1]?.body || '{}'))
    expect(body.scoped_upload_ids).toEqual(['up-1', 'up-2'])
    expect(body.scoped_file_ids).toBeNull()

    vi.unstubAllGlobals()
  })
})
