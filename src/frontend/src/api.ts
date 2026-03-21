/**
 * Informity AI — API Client
 * Single place for all fetch() calls. Components never call fetch directly.
 */

import type { PlanStepPayload, StreamChatCallbacks, StreamDonePayload } from './types/api'

function getApiBase(): string {
  return window.__INFORMITY_API_BASE__ || import.meta.env.VITE_API_URL || 'http://localhost:8420'
}

function getSessionToken(): string | null {
  return window.__INFORMITY_API_TOKEN__ || null
}

// -----------------------------------------------------------------------------
// ApiError — typed error for API failures
// -----------------------------------------------------------------------------

export class ApiError extends Error {
  status: number
  detail: string

  constructor(message: string, status: number, detail: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

// -----------------------------------------------------------------------------
// Base fetch wrapper
// -----------------------------------------------------------------------------

interface RequestConfig {
  body?: unknown
  params?: Record<string, string | number | boolean | undefined | string[]>
  headers?: Record<string, string>
  keepalive?: boolean
}

async function request<T = unknown>(
  method: string,
  path: string,
  options: RequestConfig = {},
): Promise<T> {
  const url = path.startsWith('http') ? path : `${getApiBase()}${path}`
  const { body, params } = options

  let fullUrl = url
  if (params && Object.keys(params).length > 0) {
    const search = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) {
        if (Array.isArray(v)) {
          v.forEach((item) => search.append(k, String(item)))
        } else {
          search.append(k, String(v))
        }
      }
    }
    fullUrl += (url.includes('?') ? '&' : '?') + search.toString()
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...options.headers,
  }
  const sessionToken = getSessionToken()
  if (sessionToken) {
    headers['X-Informity-Session'] = sessionToken
  }

  const config: RequestInit = {
    method,
    headers,
    cache: 'no-store',
    ...(options.keepalive === true ? { keepalive: true } : {}),
    ...(body != null ? { body: JSON.stringify(body) } : {}),
  }

  const response = await fetch(fullUrl, config)

  if (!response.ok) {
    let detail = response.statusText
    try {
      const err = await response.json() as { detail?: string; error?: string }
      detail = err.detail || err.error || detail
    } catch {
      // ignore
    }
    throw new ApiError(detail || `HTTP ${response.status}`, response.status, detail)
  }

  const contentType = response.headers.get('content-type')
  if (contentType && contentType.includes('application/json')) {
    return response.json() as Promise<T>
  }
  return response.text() as unknown as Promise<T>
}

// -----------------------------------------------------------------------------
// Scan
// -----------------------------------------------------------------------------

export async function scanFiles(
  directories?: string[] | null,
  force = false,
): Promise<unknown> {
  return request('POST', '/api/scan', {
    body: { directories: directories || undefined, force },
  })
}

export async function getScanStatus(): Promise<unknown> {
  return request('GET', '/api/scan/status')
}

export async function cancelScan(): Promise<unknown> {
  return request('POST', '/api/scan/cancel')
}

// -----------------------------------------------------------------------------
// Files
// -----------------------------------------------------------------------------

export interface GetFilesParams {
  category?: string
  extension?: string | string[]
  search?: string
  tag?: string
  sort?: string
  order?: string
  offset?: number
  limit?: number
}

export async function getFiles(params: GetFilesParams = {}): Promise<unknown> {
  const {
    category,
    extension,
    search,
    tag,
    sort = 'indexed_at',
    order = 'desc',
    offset = 0,
    limit = 50,
  } = params
  return request('GET', '/api/files', {
    params: { category, extension, search, tag, sort, order, offset, limit } as Record<string, string | number | undefined>,
  })
}

export async function getFile(id: number): Promise<unknown> {
  return request('GET', `/api/files/${id}`)
}

export async function reindexFile(id: number): Promise<unknown> {
  return request('POST', `/api/files/${id}/reindex`)
}

export async function removeFile(id: number): Promise<unknown> {
  return request('DELETE', `/api/files/${id}`)
}

export async function openFile(path: string): Promise<unknown> {
  return request('POST', '/api/files/open', { body: { path } })
}

export async function getFileTypes(): Promise<unknown> {
  return request('GET', '/api/file-types')
}

// -----------------------------------------------------------------------------
// Search
// -----------------------------------------------------------------------------

export async function search(
  query: string,
  params: { limit?: number; category?: string; file_types?: string[] } = {},
): Promise<unknown> {
  const { limit = 20, category, file_types } = params
  return request('POST', '/api/search', {
    body: { query, limit, category, file_types },
  })
}

// -----------------------------------------------------------------------------
// Index
// -----------------------------------------------------------------------------

export async function getIndexStatus(): Promise<unknown> {
  return request('GET', '/api/index/status')
}

export async function rebuildIndex(force = false): Promise<unknown> {
  return request('POST', '/api/index/rebuild', { body: { force } })
}

export async function resetIndex(force = false): Promise<unknown> {
  return request('POST', '/api/index/reset', {
    params: { force },
  })
}

// -----------------------------------------------------------------------------
// Chat (SSE streaming)
// -----------------------------------------------------------------------------

export async function streamChat(
  message: string,
  chatId: string | null,
  callbacks: StreamChatCallbacks,
): Promise<void> {
  const { onToken, onChatId, onStreamId, onRequestId, onSources, onDone, onError, onCleaned, onStatus, onPlanStep, signal } = callbacks
  let doneData: StreamDonePayload | null = null
  const streamState = { seenSources: false, seenCleaned: false, seenDone: false }
  const url = `${getApiBase()}/api/chat`
  const sessionToken = getSessionToken()
  const body = JSON.stringify({
    message: message.trim(),
    chat_id: chatId || null,
  })

  try {
    const response = await fetch(url, {
      method:  'POST',
      headers:  {
        'Content-Type': 'application/json',
        ...(sessionToken ? { 'X-Informity-Session': sessionToken } : {}),
      },
      body,
      signal,
    })

    if (!response.ok) {
      let detail = response.statusText
      try {
        const err = await response.json() as { detail?: string; error?: string }
        detail = err.detail || err.error || detail
      } catch {
        // ignore
      }
      throw new ApiError(detail || `HTTP ${response.status}`, response.status, detail)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new ApiError('No response body', 502, 'No response body')
    }
    const decoder = new TextDecoder()
    let buffer = ''
    let currentEvent = 'token'
    const currentData: string[] = []

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const clean = line.replace(/\r$/, '')

        if (clean.startsWith('event:')) {
          if (currentData.length > 0) {
            const data = currentData.join('\n').trim()
            if (data) {
              const result = handleEvent(currentEvent, data, {
                onToken,
                onChatId,
                onStreamId,
                onRequestId,
                onSources,
                onCleaned,
                onStatus,
                onPlanStep,
              }, streamState)
              if (currentEvent === 'done' && result) doneData = result
            }
            currentData.length = 0
          }
          currentEvent = clean.slice(6).trim()
        } else if (clean.startsWith('data:')) {
          currentData.push(clean.slice(5).replace(/^ /, ''))
        } else if (clean === '') {
          if (currentData.length > 0) {
            const data = currentData.join('\n').trim()
            if (data) {
              const result = handleEvent(currentEvent, data, {
                onToken,
                onChatId,
                onStreamId,
                onRequestId,
                onSources,
                onCleaned,
                onStatus,
                onPlanStep,
              }, streamState)
              if (currentEvent === 'done' && result) doneData = result
            }
            currentData.length = 0
          }
        }
      }
    }

    if (currentData.length > 0) {
      const data = currentData.join('\n').trim()
      if (data) {
        const result = handleEvent(currentEvent, data, {
          onToken,
          onChatId,
          onStreamId,
          onRequestId,
          onSources,
          onCleaned,
          onStatus,
          onPlanStep,
        }, streamState)
        if (currentEvent === 'done' && result) doneData = result
      }
    }

    onDone?.(doneData || {})
  } catch (err) {
    onError?.(err as Error)
  }
}

function handleEvent(
  event: string,
  data: string,
  callbacks: Pick<StreamChatCallbacks, 'onToken' | 'onChatId' | 'onStreamId' | 'onRequestId' | 'onSources' | 'onCleaned' | 'onStatus' | 'onPlanStep'>,
  state: { seenSources: boolean; seenCleaned: boolean; seenDone: boolean },
): StreamDonePayload | undefined {
  if (state.seenDone && event !== 'done') return undefined

  const { onToken, onChatId, onStreamId, onRequestId, onSources, onCleaned, onStatus, onPlanStep } = callbacks
  switch (event) {
    case 'token':
      if (state.seenCleaned || state.seenSources) return undefined
      if (data != null && typeof data === 'string') {
        onToken?.(data)
      }
      break
    case 'chat': {
      try {
        const parsed = JSON.parse(data) as { chat_id?: string; stream_id?: string; request_id?: string }
        if (parsed.chat_id) onChatId?.(parsed.chat_id)
        if (parsed.stream_id) onStreamId?.(parsed.stream_id)
        if (parsed.request_id) onRequestId?.(parsed.request_id)
      } catch {
        // ignore
      }
      break
    }
    case 'sources': {
      if (state.seenSources) return undefined
      state.seenSources = true
      try {
        const parsed = JSON.parse(data)
        onSources?.(Array.isArray(parsed) ? parsed : [])
      } catch {
        onSources?.([])
      }
      break
    }
    case 'cleaned':
      state.seenCleaned = true
      // Backend sends the final display-safe answer (model artifacts stripped)
      // Replace the accumulated token stream with this cleaned version
      if (data != null && typeof data === 'string') {
        onCleaned?.(data)
      }
      break
    case 'status': {
      try {
        const parsed = JSON.parse(data)
        onStatus?.(parsed)
      } catch {
        onStatus?.({})
      }
      break
    }
    case 'plan_step': {
      try {
        const parsed = JSON.parse(data) as PlanStepPayload
        onPlanStep?.(parsed)
      } catch {
        // ignore
      }
      break
    }
    case 'done': {
      state.seenDone = true
      try {
        return JSON.parse(data) as StreamDonePayload
      } catch {
        return {}
      }
    }
    case 'timeout': {
      try {
        return JSON.parse(data) as StreamDonePayload
      } catch {
        return {}
      }
    }
    default:
      break
  }
  return undefined
}

// -----------------------------------------------------------------------------
// Chats
// -----------------------------------------------------------------------------

export interface GetChatsParams {
  limit?: number
  offset?: number
  search?: string
}

export async function getChats(params: GetChatsParams = {}): Promise<unknown> {
  const { limit = 50, offset = 0, search } = params
  return request('GET', '/api/chat/chats', {
    params: { limit, offset, search: search || undefined } as Record<string, string | number | undefined>,
  })
}

export async function getChat(chatId: string): Promise<unknown> {
  return request('GET', `/api/chat/chats/${chatId}`)
}

export async function stopChatStream(chatId: string, streamId: string): Promise<{ stopped: boolean; stream_id: string }> {
  return request('POST', '/api/chat/stop', {
    body: { chat_id: chatId, stream_id: streamId },
  }) as Promise<{ stopped: boolean; stream_id: string }>
}

export async function getMessageRaw(messageId: number): Promise<{ content: string }> {
  return request('GET', `/api/chat/messages/${messageId}/raw`) as Promise<{ content: string }>
}

export async function setChatTitle(chatId: string, title: string): Promise<unknown> {
  return request('PUT', `/api/chat/chats/${chatId}/title`, {
    params: { title } as Record<string, string>,
  })
}

export async function deleteChat(chatId: string): Promise<unknown> {
  return request('DELETE', `/api/chat/chats/${chatId}`)
}

// -----------------------------------------------------------------------------
// Settings
// -----------------------------------------------------------------------------

export async function getSettings(): Promise<unknown> {
  return request('GET', '/api/settings')
}

export async function getModelProfile(modelFilename: string): Promise<unknown> {
  return request('GET', '/api/settings/model-profile', {
    params: { model_filename: modelFilename },
  })
}

export async function updateSettings(updates: Record<string, unknown>): Promise<unknown> {
  return request('PUT', '/api/settings', { body: updates })
}

export async function resetSettings(): Promise<unknown> {
  return request('POST', '/api/settings/reset')
}

export async function getCurrentChat(): Promise<unknown> {
  return request('GET', '/api/settings/current-chat')
}

export async function updateCurrentChat(currentChatId: string | null): Promise<unknown> {
  return request('PUT', '/api/settings/current-chat', {
    body: { current_chat_id: currentChatId },
    // Persist "new chat" intent even if user reloads immediately after clicking New Chat.
    keepalive: currentChatId == null,
  })
}

// -----------------------------------------------------------------------------
// Configuration (env vars + reference)
// -----------------------------------------------------------------------------

export async function getEnvVars(): Promise<unknown> {
  return request('GET', '/api/config/env-vars')
}

export async function getConfigReference(): Promise<unknown> {
  return request('GET', '/api/config/reference')
}

// -----------------------------------------------------------------------------
// Health
// -----------------------------------------------------------------------------

export async function getHealth(): Promise<unknown> {
  return request('GET', '/api/health')
}
