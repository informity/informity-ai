/**
 * Informity AI — Chat provider
 * Owns chat stream/session state so navigation does not interrupt streaming.
 */
import { useState, useCallback, useEffect, useRef, type ReactNode } from 'react'
import { ChatContext } from './chatContext'
import {
  ApiError,
  deleteChatUpload,
  getChat,
  getSettings,
  listChatUploads,
  stopChatStream,
  streamChat,
  updateChatPreferences,
  updateCurrentChat,
  uploadChatFile,
} from '../api'
import { showToast } from './useToast'
import { logApiError } from '../utils/logApiError'
import { extractErrorMessage } from '../utils/errorMessages'
import { CHAT_FILE_SCOPE_MAP_STORAGE_KEY, FORCE_NEW_CHAT_KEY, MESSAGE_MODE_MAP_STORAGE_KEY } from '../utils/storageKeys'
import type {
  ChatFileScope,
  ChatMode,
  ChatMessageApi,
  ChatMessageDisplay,
  ChatUploadAttachment,
  DisplayBlock,
  NextAction,
  NextActionReason,
  PlanStepPayload,
  StreamDonePayload,
} from '../types/api'
import { isChatMode } from '../types/api'

interface ChatProviderProps {
  children: ReactNode
}

interface GetChatResponse {
  messages?: ChatMessageApi[]
  chat_mode?: ChatMode
  role_id?: string | null
  chat_web_search_enabled?: boolean
  chat_web_search_privacy_override?: boolean
}

const CLEANED_REVEAL_INTERVAL_MS = 14
const CLEANED_REVEAL_CHARS_PER_TICK = 8
const CONTINUE_SCOPED_PROMPT = 'Continue with the remaining sections from your last answer. Keep the same structure and avoid repeating completed sections.'
const ACTIVE_GENERATION_REJECT_MESSAGE = 'Please wait for the current answer to finish or press Stop.'
// Keep watchdog well above backend generation hard limits.
// This timer is only a dead-connection guard.
const STREAM_INACTIVITY_TIMEOUT_MS = 20 * 60 * 1000
const STREAM_WATCHDOG_TIMEOUT_MESSAGE = 'Connection lost while waiting for response. Please try again.'
const STREAM_WATCHDOG_INTERRUPTED_MESSAGE = 'Response was interrupted due to connection inactivity.'
const STOP_ACK_TIMEOUT_MS = 1500
const STREAM_STATUS_TIMER_INTERVAL_MS = 1000
const STREAM_STATUS_LABELS: Record<string, string> = {
  classifying: 'Analyzing your request...',
  retrieving: 'Searching for relevant information...',
  searching: 'Searching the web...',
  generating: 'Generating response...',
  continuing: 'Continuing response...',
  finalizing: 'Finalizing answer...',
}

function isTransientFetchFailure(err: unknown): boolean {
  const msg = String((err as { message?: unknown })?.message || '').trim().toLowerCase()
  return msg.includes('failed to fetch') || msg.includes('networkerror') || msg.includes('load failed')
}

function normalizeCompletionMode(
  value: unknown,
): 'complete' | 'partial' | 'scoped_complete' | 'stopped' {
  return value === 'partial' || value === 'scoped_complete' || value === 'stopped'
    ? value
    : 'complete'
}

function getStreamStatusLabel(state: string): string | undefined {
  return STREAM_STATUS_LABELS[state]
}

function shouldShowLatencyHint(state: string): boolean {
  return ['classifying', 'retrieving', 'searching', 'generating', 'continuing', 'finalizing'].includes(state)
}

function formatStreamStatusWithLatency(
  baseMessage: string,
  state: string,
  elapsedSeconds: number,
): string {
  const base = baseMessage.trim()
  if (!base || !shouldShowLatencyHint(state)) return base
  const elapsed = Math.max(0, Math.floor(elapsedSeconds))
  const normalizedBase = base.replace(/\.+$/, '')
  if (elapsed >= 20) return `${normalizedBase}, still working... ${elapsed}s`
  if (elapsed > 0) return `${normalizedBase}... ${elapsed}s`
  return base
}

function setForceNewChatFlag(enabled: boolean): void {
  try {
    if (enabled) {
      window.localStorage.setItem(FORCE_NEW_CHAT_KEY, '1')
      window.sessionStorage.setItem(FORCE_NEW_CHAT_KEY, '1')
      return
    }
    window.localStorage.removeItem(FORCE_NEW_CHAT_KEY)
    window.sessionStorage.removeItem(FORCE_NEW_CHAT_KEY)
  } catch {
    // ignore storage failures
  }
}

function buildRecoveryCallout(
  nextAction: 'none' | 'continue' | 'regenerate' | 'assistant_switch',
  nextActionReason?: 'stopped' | 'timeout' | 'unresolved_content' | 'budget_exhausted' | 'stalled' | 'out_of_corpus' | null,
): DisplayBlock | null {
  if (nextAction === 'none') return null
  if (nextActionReason === 'stalled') {
    return {
      type: 'callout',
      tone: 'warning',
      text: 'I could not make additional progress on the remaining scope. Refine your request and try again.',
    }
  }
  if (nextActionReason === 'budget_exhausted') {
    return {
      type: 'callout',
      tone: 'warning',
      text: 'I reached the continuation pass budget before finishing all requested sections.',
    }
  }
  if (nextActionReason === 'timeout') {
    return {
      type: 'callout',
      tone: 'warning',
      text: 'This response hit the time limit before all requested content was complete.',
    }
  }
  if (nextActionReason === 'stopped') {
    return {
      type: 'callout',
      tone: 'info',
      text: 'Generation was stopped before completion.',
    }
  }
  if (nextActionReason === 'unresolved_content') {
    return {
      type: 'callout',
      tone: 'info',
      text: 'I completed the highest-priority sections first and left some requested scope unresolved.',
    }
  }
  return null
}

function readStoredMessageModes(): Record<string, ChatMode> {
  try {
    const raw = window.localStorage.getItem(MESSAGE_MODE_MAP_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const normalized: Record<string, ChatMode> = {}
    for (const [key, value] of Object.entries(parsed || {})) {
      if (isChatMode(value)) {
        normalized[key] = value
      }
    }
    return normalized
  } catch {
    return {}
  }
}

function storeMessageMode(messageId: number, mode: ChatMode): void {
  if (!Number.isInteger(messageId)) return
  try {
    const map = readStoredMessageModes()
    map[String(messageId)] = mode
    window.localStorage.setItem(MESSAGE_MODE_MAP_STORAGE_KEY, JSON.stringify(map))
  } catch {
    // ignore storage failures
  }
}

function readStoredChatFileScopes(): Record<string, ChatFileScope> {
  try {
    const raw = window.localStorage.getItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const normalized: Record<string, ChatFileScope> = {}
    for (const [chatId, scopeRaw] of Object.entries(parsed || {})) {
      if (!chatId || chatId === '__draft__') continue
      if (!scopeRaw || typeof scopeRaw !== 'object') continue
      const scopeObj = scopeRaw as { fileId?: unknown; filename?: unknown }
      const fileId = Number(scopeObj.fileId)
      if (!Number.isFinite(fileId) || fileId <= 0) continue
      const normalizedName = String(scopeObj.filename || '').trim()
      normalized[chatId] = {
        fileId: Math.trunc(fileId),
        filename: normalizedName || `File ${Math.trunc(fileId)}`,
      }
    }
    return normalized
  } catch {
    return {}
  }
}

function persistStoredChatFileScopes(scopes: Record<string, ChatFileScope>): void {
  try {
    const persisted: Record<string, ChatFileScope> = {}
    for (const [chatId, scope] of Object.entries(scopes || {})) {
      if (!chatId || chatId === '__draft__') continue
      const fileId = Number(scope?.fileId)
      if (!Number.isFinite(fileId) || fileId <= 0) continue
      const normalizedName = String(scope?.filename || '').trim()
      persisted[chatId] = {
        fileId: Math.trunc(fileId),
        filename: normalizedName || `File ${Math.trunc(fileId)}`,
      }
    }
    window.localStorage.setItem(CHAT_FILE_SCOPE_MAP_STORAGE_KEY, JSON.stringify(persisted))
  } catch {
    // ignore storage failures
  }
}

function createChatRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

async function resolveFileScopeFromHistory(messages: ChatMessageApi[]): Promise<ChatFileScope | null> {
  const indexedFileScopeKeys = new Set(
    messages
      .map((message) => ({
        kind: String(message.retrieval_scope_kind || '').trim(),
        key: String(message.retrieval_scope_key || '').trim(),
      }))
      .filter((item) => item.kind === 'indexed_files' && item.key.length > 0)
      .map((item) => item.key),
  )
  if (indexedFileScopeKeys.size !== 1) return null
  const [scopeKey] = [...indexedFileScopeKeys]
  const fileIds = scopeKey
    .split(',')
    .map((value) => Number.parseInt(value.trim(), 10))
    .filter((value) => Number.isFinite(value) && value > 0)
  if (fileIds.length !== 1) return null

  const fileId = Math.trunc(fileIds[0])
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]
    if (message.role !== 'assistant' || !Array.isArray(message.sources) || message.sources.length === 0) continue
    for (const source of message.sources) {
      if (Number(source?.file_id) !== fileId) continue
      const filename = String(source?.filename || '').trim()
      if (filename) {
        return { fileId, filename }
      }
    }
  }
  return { fileId, filename: `File ${fileId}` }
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [currentChatId, setCurrentChatIdState] = useState<string | null>(null)
  const [currentChatLockedMode, setCurrentChatLockedMode] = useState<ChatMode | null>(null)
  const [currentChatLockedRoleId, setCurrentChatLockedRoleId] = useState<string | null>(null)
  const [activeGenerationChatId, setActiveGenerationChatId] = useState<string | null>(null)
  const [activeGenerationRequestId, setActiveGenerationRequestId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessageDisplay[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [loadingChat, setLoadingChat] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [enableRawOutputControl, setEnableRawOutputControl] = useState(false)
  const [chatWebSearchEnabled, setChatWebSearchEnabled] = useState(false)
  const [chatWebSearchPrivacyOverride, setChatWebSearchPrivacyOverride] = useState(false)
  const [chatFileScope, setChatFileScope] = useState<ChatFileScope | null>(null)
  const [chatUploads, setChatUploads] = useState<ChatUploadAttachment[]>([])
  const abortControllerRef = useRef<AbortController | null>(null)
  const streamContentRef = useRef('')
  const streamThrottleRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const sendInFlightRef = useRef(false)
  const streamSessionRef = useRef(0)
  const chatLoadSessionRef = useRef(0)
  const streamIdRef = useRef<string | null>(null)
  const streamChatIdRef = useRef<string | null>(null)
  const streamRequestIdRef = useRef<string | null>(null)
  const streamDraftRef = useRef<ChatMessageDisplay | null>(null)
  const streamRevealTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const streamRevealActiveRef = useRef(false)
  const streamPendingDoneRef = useRef<StreamDonePayload | null>(null)
  const streamCleanedContentRef = useRef<string | null>(null)
  const streamRevealCharIndexRef = useRef(0)
  const streamWatchdogTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const streamWatchdogTimedOutRef = useRef(false)
  const streamStatusTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const streamStatusBaseMessageRef = useRef<string>('')
  const streamStatusStateRef = useRef<string>('generating')
  const streamStatusStartMsRef = useRef<number>(0)
  const streamPlanStepsRef = useRef<Array<{ step_id: number; description: string; status: 'running' | 'done' | 'empty' }>>([])
  const streamStopRequestedRef = useRef(false)
  const currentChatIdRef = useRef<string | null>(null)
  const isStreamingRef = useRef(false)
  const messagesCountRef = useRef(0)
  const lastAutoContinuedMessageIdRef = useRef<number | null>(null)
  const chatPrefsRef = useRef<Record<string, { enabled: boolean; privacyOverride: boolean }>>({})
  const chatFileScopesRef = useRef<Record<string, ChatFileScope>>(readStoredChatFileScopes())

  useEffect(() => {
    currentChatIdRef.current = currentChatId
  }, [currentChatId])

  useEffect(() => {
    isStreamingRef.current = isStreaming
  }, [isStreaming])
  useEffect(() => {
    messagesCountRef.current = messages.length
  }, [messages.length])

  const setCurrentChatId = useCallback((id: string | null) => {
    setCurrentChatIdState(id)
  }, [])

  const isViewingGeneratingChat = useCallback((): boolean => {
    const streamChatId = streamChatIdRef.current
    if (!streamChatId) return false
    return currentChatIdRef.current === streamChatId
  }, [])

  const applyStreamDraftToVisibleMessages = useCallback(() => {
    const draft = streamDraftRef.current
    if (!draft || !isViewingGeneratingChat()) return
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      if (last?.role === 'assistant' && last.isStreaming) {
        next[next.length - 1] = { ...last, ...draft }
        return next
      }
      return [...next, { ...draft }]
    })
  }, [isViewingGeneratingChat])

  const clearRevealTimer = useCallback(() => {
    if (streamRevealTimerRef.current) {
      clearTimeout(streamRevealTimerRef.current)
      streamRevealTimerRef.current = null
    }
  }, [])

  const clearStreamWatchdog = useCallback(() => {
    if (streamWatchdogTimerRef.current) {
      clearTimeout(streamWatchdogTimerRef.current)
      streamWatchdogTimerRef.current = null
    }
  }, [])

  const clearStreamStatusTimer = useCallback(() => {
    if (streamStatusTimerRef.current) {
      clearInterval(streamStatusTimerRef.current)
      streamStatusTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    getSettings()
      .then((s) => {
        const data = s as { enable_raw_output_control?: boolean }
        setEnableRawOutputControl(!!data?.enable_raw_output_control)
      })
      .catch((err) => logApiError(err, 'ChatProvider.getSettings'))
  }, [])

  const clearError = useCallback(() => setError(null), [])

  const resolveDraftOrChatFileScope = useCallback((chatId: string | null): ChatFileScope | null => {
    const scope = chatId
      ? chatFileScopesRef.current[chatId]
      : chatFileScopesRef.current.__draft__
    if (!scope) return null
    if (!Number.isFinite(scope.fileId) || scope.fileId <= 0) return null
    const normalizedName = String(scope.filename || '').trim()
    return {
      fileId: Math.trunc(scope.fileId),
      filename: normalizedName || `File ${Math.trunc(scope.fileId)}`,
    }
  }, [])

  const clearChatFileScope = useCallback(() => {
    const activeChatId = currentChatIdRef.current
    if (activeChatId) {
      delete chatFileScopesRef.current[activeChatId]
      persistStoredChatFileScopes(chatFileScopesRef.current)
    } else {
      delete chatFileScopesRef.current.__draft__
    }
    setChatFileScope(null)
  }, [])

  const normalizeChatUploads = useCallback((attachments: ChatUploadAttachment[]): ChatUploadAttachment[] => {
    const sorted = [...(attachments || [])]
    sorted.sort((a, b) => {
      const aTime = String(a?.uploaded_at || a?.updated_at || '')
      const bTime = String(b?.uploaded_at || b?.updated_at || '')
      if (aTime === bTime) return 0
      return aTime < bTime ? -1 : 1
    })
    return sorted
  }, [])

  const refreshChatUploads = useCallback(async (chatId: string | null) => {
    if (!chatId) {
      setChatUploads([])
      return
    }
    try {
      const data = await listChatUploads(chatId)
      const nextAttachments = normalizeChatUploads(Array.isArray(data?.attachments) ? data.attachments : [])
      setChatUploads(nextAttachments)
    } catch (err) {
      logApiError(err, 'ChatProvider.refreshChatUploads')
      setChatUploads([])
    }
  }, [normalizeChatUploads])

  const uploadFiles = useCallback(async (
    files: File[],
    options?: { onChatResolved?: (chatId: string) => void },
  ) => {
    const validFiles = files.filter((item) => item instanceof File)
    if (validFiles.length === 0) return
    let targetChatId = currentChatIdRef.current
    let refreshTargetChatId = targetChatId
    let uploadError: unknown = null
    try {
      for (const file of validFiles) {
        try {
          const response = await uploadChatFile(file, targetChatId)
          const responseChatId = String(response?.chat_id || '').trim()
          if (responseChatId && responseChatId !== currentChatIdRef.current) {
            currentChatIdRef.current = responseChatId
            setCurrentChatIdState(responseChatId)
            await updateCurrentChat(responseChatId).catch((err) => logApiError(err, 'ChatProvider.uploadFiles.updateCurrentChat'))
          }
          if (responseChatId) {
            refreshTargetChatId = responseChatId
            options?.onChatResolved?.(responseChatId)
          }
          targetChatId = responseChatId || targetChatId
        } catch (err) {
          uploadError = err
          break
        }
      }
    } catch (err) {
      uploadError = err
    } finally {
      await refreshChatUploads(refreshTargetChatId || currentChatIdRef.current)
    }
    if (uploadError != null) {
      const msg = extractErrorMessage(uploadError, 'Upload failed')
      setError(msg)
      showToast('error', msg)
      throw uploadError
    }
  }, [refreshChatUploads])

  const removeUploadedFile = useCallback(async (uploadId: string) => {
    const chatId = currentChatIdRef.current
    if (!chatId) return
    try {
      const result = await deleteChatUpload(uploadId, chatId)
      if (typeof result?.toast_message === 'string' && result.toast_message.trim().length > 0) {
        showToast('info', result.toast_message)
      }
      await refreshChatUploads(chatId)
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to remove uploaded file')
      setError(msg)
      showToast('error', msg)
      throw err
    }
  }, [refreshChatUploads])

  const setChatWebSearchPreferences = useCallback(async (
    prefs: { enabled: boolean; privacyOverride: boolean; persist?: boolean },
  ) => {
    const next = {
      enabled: !!prefs.enabled,
      privacyOverride: !!prefs.privacyOverride,
    }
    const chatKey = currentChatIdRef.current ?? '__draft__'
    chatPrefsRef.current[chatKey] = next
    setChatWebSearchEnabled(next.enabled)
    setChatWebSearchPrivacyOverride(next.privacyOverride)
    if (prefs.persist !== false && currentChatIdRef.current) {
      await updateChatPreferences(currentChatIdRef.current, {
        chat_web_search_enabled: next.enabled,
        chat_web_search_privacy_override: next.privacyOverride,
      }).catch((err) => logApiError(err, 'ChatProvider.setChatWebSearchPreferences'))
    }
  }, [])

  const selectChat = useCallback(async (selectedChatId: string) => {
    if (selectedChatId === currentChatId && messagesCountRef.current > 0) return

    const sessionId = ++chatLoadSessionRef.current
    setError(null)
    setLoadingChat(true)
    setMessages([])
    try {
      let data: GetChatResponse
      try {
        data = (await getChat(selectedChatId)) as GetChatResponse
      } catch (firstErr) {
        if (!isTransientFetchFailure(firstErr)) throw firstErr
        await new Promise((resolve) => setTimeout(resolve, 250))
        data = (await getChat(selectedChatId)) as GetChatResponse
      }
      if (chatLoadSessionRef.current !== sessionId) return
      const historyMessages = data.messages || []
      const lockedChatMode = isChatMode(data.chat_mode) ? data.chat_mode : undefined
      const lockedRoleId = typeof data.role_id === 'string' && data.role_id.trim().length > 0
        ? data.role_id.trim()
        : null
      setCurrentChatLockedMode(lockedChatMode ?? null)
      setCurrentChatLockedRoleId(lockedRoleId)
      const resolvedPrefs = {
        enabled: data.chat_web_search_enabled === true,
        privacyOverride: data.chat_web_search_privacy_override === true,
      }
      chatPrefsRef.current[selectedChatId] = resolvedPrefs
      setChatWebSearchEnabled(resolvedPrefs.enabled)
      setChatWebSearchPrivacyOverride(resolvedPrefs.privacyOverride)
      let resolvedFileScope = resolveDraftOrChatFileScope(selectedChatId)
      if (!resolvedFileScope) {
        const inferredScope = await resolveFileScopeFromHistory(historyMessages)
        if (chatLoadSessionRef.current !== sessionId) return
        if (inferredScope) {
          chatFileScopesRef.current[selectedChatId] = inferredScope
          persistStoredChatFileScopes(chatFileScopesRef.current)
          resolvedFileScope = inferredScope
        }
      }
      setChatFileScope(resolvedFileScope)
      try {
        await refreshChatUploads(selectedChatId)
      } catch (uploadErr) {
        // Non-fatal for history open; keep the chat readable even if uploads refresh
        // had a transient transport error.
        if (!isTransientFetchFailure(uploadErr)) throw uploadErr
      }
      const storedModes = readStoredMessageModes()
      const mapped: ChatMessageDisplay[] = historyMessages.map((m, index) => {
        const completionMode = normalizeCompletionMode(m.completion_mode)
        const previousMessage = index > 0 ? historyMessages[index - 1] : undefined
        const hasRemainingScope = m.has_remaining_scope
          ?? (completionMode === 'partial' || completionMode === 'scoped_complete' || completionMode === 'stopped')
        const stoppedByUser = m.stopped_by_user ?? (completionMode === 'stopped')
        const historyBlocks = Array.isArray(m.display_blocks) ? m.display_blocks : undefined
        const nextAction: NextAction = m.next_action ?? 'none'
        const nextActionReason: NextActionReason | null = m.next_action_reason ?? null
        const recoveryCallout = buildRecoveryCallout(nextAction, nextActionReason)
        const nextDisplayBlocks = recoveryCallout
          ? [...(historyBlocks || []), recoveryCallout]
          : historyBlocks
        const storedMode = typeof m.id === 'number' ? storedModes[String(m.id)] : undefined
        const explicitMessageMode = isChatMode(m.chat_mode) ? m.chat_mode : undefined
        const inferredAssistantMode: ChatMode | undefined = (
          m.role === 'assistant'
            ? (
                storedMode
                ?? explicitMessageMode
                ?? lockedChatMode
                ?? (nextActionReason === 'out_of_corpus' ? 'researcher' : undefined)
                ?? ((m.sources?.length || 0) > 0 ? 'researcher' : undefined)
              )
            : (explicitMessageMode ?? lockedChatMode)
        )
        return {
          id: m.id,
          role: m.role,
          content: m.content || '',
          isInternal: !!m.is_internal,
          isContinuation: (
            m.role === 'assistant'
            && previousMessage?.role === 'user'
            && !!previousMessage.is_internal
          ),
          sources: m.sources || [],
          displayBlocks: nextDisplayBlocks,
          isPartial: completionMode === 'partial',
          hasRemainingScope,
          completionMode,
          stoppedByUser,
          nextAction,
          nextActionReason,
          continueLabel: 'Continue',
          createdAt: m.created_at,
          generationSeconds: m.generation_seconds,
          chatMode: inferredAssistantMode,
          roleId: (
            typeof m.role_id === 'string' && m.role_id.trim().length > 0
              ? m.role_id.trim()
              : lockedRoleId
          ),
          scopedFileName: m.role === 'assistant' ? resolvedFileScope?.filename ?? null : null,
        }
      })
      currentChatIdRef.current = selectedChatId
      setCurrentChatIdState(selectedChatId)
      setForceNewChatFlag(false)
      lastAutoContinuedMessageIdRef.current = null
      if (isStreamingRef.current && streamChatIdRef.current === selectedChatId && streamDraftRef.current) {
        setMessages([...mapped, { ...streamDraftRef.current }])
      } else {
        setMessages(mapped)
      }
      updateCurrentChat(selectedChatId).catch((err) => logApiError(err, 'ChatProvider.selectChat.updateCurrentChat'))
    } catch (err) {
      if (chatLoadSessionRef.current !== sessionId) return
      const is404 = err instanceof ApiError && err.status === 404
      if (is404) {
        delete chatFileScopesRef.current[selectedChatId]
        persistStoredChatFileScopes(chatFileScopesRef.current)
        currentChatIdRef.current = null
        setCurrentChatIdState(null)
        setCurrentChatLockedMode(null)
        setCurrentChatLockedRoleId(null)
        setChatWebSearchEnabled(false)
        setChatWebSearchPrivacyOverride(false)
        setChatFileScope(null)
        setChatUploads([])
        setMessages([])
        setError(null)
        updateCurrentChat(null).catch((e) => logApiError(e, 'ChatProvider.selectChat.clearCurrentChat'))
      } else {
        const msg = extractErrorMessage(err, 'Failed to load chat')
        setError(msg)
        showToast('error', msg)
      }
    } finally {
      if (chatLoadSessionRef.current === sessionId) {
        setLoadingChat(false)
      }
    }
  }, [currentChatId, refreshChatUploads, resolveDraftOrChatFileScope])

  const goToGeneratingChat = useCallback(async () => {
    const generatingChatId = streamChatIdRef.current ?? activeGenerationChatId
    if (!generatingChatId) return
    await selectChat(generatingChatId)
  }, [activeGenerationChatId, selectChat])

  const stopStreamingInternal = useCallback(async (): Promise<boolean> => {
    if (!isStreamingRef.current) return false
    clearStreamWatchdog()
    streamWatchdogTimedOutRef.current = false
    const streamId = streamIdRef.current
    const requestId = streamRequestIdRef.current ?? activeGenerationRequestId
    const chatId = streamChatIdRef.current ?? currentChatIdRef.current
    streamStopRequestedRef.current = true
    if (!streamId && !requestId) {
      abortControllerRef.current?.abort()
      return true
    }

    const stopAckPromise = stopChatStream(chatId ?? null, {
      streamId,
      requestId,
    }).catch((err) => {
      logApiError(err, 'ChatProvider.stopStreaming.stopChatStream')
      return null
    })

    // Always interrupt local stream immediately so Stop feels responsive.
    abortControllerRef.current?.abort()

    try {
      const res = await Promise.race<
        Awaited<ReturnType<typeof stopChatStream>> | null
      >([
        stopAckPromise,
        new Promise<null>((resolve) => {
          setTimeout(() => resolve(null), STOP_ACK_TIMEOUT_MS)
        }),
      ])
      // Backend stop acknowledgement can lag behind model initialization.
      // Local stream is already aborted above; treat timeout as successful stop.
      if (res == null) {
        return true
      }
      if (res.status === 'already_terminal' || res.status === 'not_found') {
        return true
      }
      return !!res.stopped
    } catch {
      return true
    }
  }, [activeGenerationRequestId, clearStreamWatchdog])

  const sendMessage = useCallback(async (
    text: string,
    options?: {
      isInternal?: boolean
      mode?: ChatMode
      roleId?: string | null
      fileScope?: ChatFileScope | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => {
    const message = text.trim()
    const isInternalMessage = !!options?.isInternal
    const chatMode: ChatMode = options?.mode ?? 'researcher'
    const roleId = typeof options?.roleId === 'string' && options.roleId.trim().length > 0
      ? options.roleId.trim()
      : null
    const providedScope = options?.fileScope ?? null
    const chatWebSearchEnabled = !!options?.chatWebSearchEnabled
    const chatWebSearchPrivacyOverride = !!options?.chatWebSearchPrivacyOverride
    if (!message || sendInFlightRef.current) return
    if (isStreamingRef.current) {
      if (!isInternalMessage) {
        setError(ACTIVE_GENERATION_REJECT_MESSAGE)
        showToast('warning', ACTIVE_GENERATION_REJECT_MESSAGE)
      }
      return
    }
    const hasActiveUploads = chatUploads.some((item) => ['uploading', 'indexing', 'ready'].includes(String(item.state)))
    let requestChatId = currentChatIdRef.current
    const readyUploads = chatUploads.filter((item) => item.state === 'ready')
    const readyUploadChatIds = Array.from(
      new Set(
        readyUploads
          .map((item) => String(item.chat_id || '').trim())
          .filter((id) => id.length > 0),
      ),
    )
    if (!requestChatId && readyUploadChatIds.length === 1) {
      requestChatId = readyUploadChatIds[0]
      currentChatIdRef.current = requestChatId
      setCurrentChatIdState(requestChatId)
      updateCurrentChat(requestChatId).catch((err) => logApiError(err, 'ChatProvider.sendMessage.resolveRequestChatId'))
    }
    const effectiveFileScope = providedScope ?? resolveDraftOrChatFileScope(requestChatId)
    if (chatMode !== 'researcher' && (hasActiveUploads || effectiveFileScope)) {
      const modeError = 'Document-scoped chat is available only in Researcher mode.'
      setError(modeError)
      showToast('warning', modeError)
      return
    }
    if (!isInternalMessage) {
      setForceNewChatFlag(false)
      lastAutoContinuedMessageIdRef.current = null
    }
    sendInFlightRef.current = true
    const readyUploadIdSet = new Set(
      readyUploads
        .filter((item) => !requestChatId || String(item.chat_id || '').trim() === requestChatId)
        .map((item) => String(item.upload_id)),
    )
    const effectiveScopedUploadIds = effectiveFileScope
      ? []
      : [...readyUploadIdSet]

    setError(null)
    const now = new Date().toISOString()
    setMessages((prev) => [...prev, {
      role: 'user',
      content: message,
      isInternal: isInternalMessage,
      sources: [],
      isPartial: false,
      createdAt: now,
    }])
    setIsStreaming(true)
    const assistantDraft: ChatMessageDisplay = {
      role: 'assistant',
      content: '',
      sources: [],
      chatMode,
      roleId,
      scopedFileName: effectiveFileScope?.filename ?? null,
      isStreaming: true,
      isContinuation: isInternalMessage,
      streamStatusText: isInternalMessage
        ? getStreamStatusLabel('continuing')
        : 'Generating response...',
      isPartial: false,
      streamSectionProgress: undefined,
      createdAt: now,
    }
    streamDraftRef.current = { ...assistantDraft }
    setMessages((prev) => [...prev, assistantDraft])

    abortControllerRef.current = new AbortController()
    streamContentRef.current = ''
    streamCleanedContentRef.current = null
    streamRevealCharIndexRef.current = 0
    streamRevealActiveRef.current = false
    streamPendingDoneRef.current = null
    streamPlanStepsRef.current = []
    clearRevealTimer()
    streamSessionRef.current += 1
    const sessionId = streamSessionRef.current
    streamWatchdogTimedOutRef.current = false
    streamIdRef.current = null
    streamChatIdRef.current = null
    const requestId = createChatRequestId()
    streamRequestIdRef.current = requestId
    streamStopRequestedRef.current = false
    streamStatusBaseMessageRef.current = String(assistantDraft.streamStatusText || 'Generating response...')
    streamStatusStateRef.current = isInternalMessage ? 'continuing' : 'generating'
    streamStatusStartMsRef.current = Date.now()
    clearStreamStatusTimer()
    if (!requestChatId) {
      chatPrefsRef.current.__draft__ = {
        enabled: chatWebSearchEnabled,
        privacyOverride: chatWebSearchPrivacyOverride,
      }
    } else {
      chatPrefsRef.current[requestChatId] = {
        enabled: chatWebSearchEnabled,
        privacyOverride: chatWebSearchPrivacyOverride,
      }
    }
    if (effectiveFileScope) {
      if (requestChatId) {
        chatFileScopesRef.current[requestChatId] = effectiveFileScope
        persistStoredChatFileScopes(chatFileScopesRef.current)
      } else {
        chatFileScopesRef.current.__draft__ = effectiveFileScope
      }
      setChatFileScope(effectiveFileScope)
    }
    setChatWebSearchEnabled(chatWebSearchEnabled)
    setChatWebSearchPrivacyOverride(chatWebSearchPrivacyOverride)
    setActiveGenerationChatId(requestChatId)
    setActiveGenerationRequestId(requestId)

    try {
      const touchStreamWatchdog = () => {
        if (streamSessionRef.current !== sessionId) return
        clearStreamWatchdog()
        streamWatchdogTimerRef.current = setTimeout(() => {
          if (streamSessionRef.current !== sessionId || !isStreamingRef.current) return
          streamWatchdogTimedOutRef.current = true
          abortControllerRef.current?.abort()
        }, STREAM_INACTIVITY_TIMEOUT_MS)
      }

      const refreshStreamStatusMessage = () => {
        if (streamSessionRef.current !== sessionId || !isStreamingRef.current) return
        const draft = streamDraftRef.current
        if (!draft || !draft.isStreaming) return
        const base = String(streamStatusBaseMessageRef.current || '').trim()
        const state = String(streamStatusStateRef.current || 'generating').trim().toLowerCase()
        if (!base || !state) return
        const elapsedSeconds = (Date.now() - streamStatusStartMsRef.current) / 1000
        const nextStatusText = formatStreamStatusWithLatency(base, state, elapsedSeconds)
        if (nextStatusText === (draft.streamStatusText || '')) return
        streamDraftRef.current = {
          ...draft,
          streamStatusText: nextStatusText,
        }
        applyStreamDraftToVisibleMessages()
      }

      const reconcileFinalAssistantMessage = async (
        finalizedMessageId: number | undefined,
        finalizedChatId: string | null,
      ): Promise<void> => {
        if (typeof finalizedMessageId !== 'number' || !Number.isFinite(finalizedMessageId) || !finalizedChatId) {
          return
        }
        try {
          const data = (await getChat(finalizedChatId)) as GetChatResponse
          const historyMessages = Array.isArray(data?.messages) ? data.messages : []
          const persisted = historyMessages.find(
            (m) => m.role === 'assistant' && typeof m.id === 'number' && m.id === finalizedMessageId,
          )
          if (!persisted) return
          if (currentChatIdRef.current !== finalizedChatId) return

          const persistedMode = isChatMode(persisted.chat_mode) ? persisted.chat_mode : undefined
          const persistedCompletionMode = normalizeCompletionMode(persisted.completion_mode)
          const persistedHasRemainingScope = persisted.has_remaining_scope
            ?? (
              persistedCompletionMode === 'partial'
              || persistedCompletionMode === 'scoped_complete'
              || persistedCompletionMode === 'stopped'
            )
          const persistedStoppedByUser = persisted.stopped_by_user ?? (persistedCompletionMode === 'stopped')
          const persistedNextAction: NextAction = persisted.next_action ?? 'none'
          const persistedNextActionReason: NextActionReason | null = persisted.next_action_reason ?? null
          const persistedRecoveryCallout = buildRecoveryCallout(persistedNextAction, persistedNextActionReason)
          const persistedBlocks = Array.isArray(persisted.display_blocks) ? persisted.display_blocks : undefined
          const persistedDisplayBlocks = persistedRecoveryCallout
            ? [...(persistedBlocks || []), persistedRecoveryCallout]
            : persistedBlocks

          setMessages((prev) => prev.map((msg) => {
            if (msg.role !== 'assistant' || msg.id !== finalizedMessageId || msg.isStreaming) return msg
            return {
              ...msg,
              content: persisted.content || msg.content || '',
              sources: persisted.sources || msg.sources || [],
              displayBlocks: persistedDisplayBlocks ?? msg.displayBlocks,
              completionMode: persistedCompletionMode,
              isPartial: persistedCompletionMode === 'partial',
              hasRemainingScope: persistedHasRemainingScope,
              stoppedByUser: persistedStoppedByUser,
              chatMode: persistedMode ?? msg.chatMode,
              roleId: (
                typeof persisted.role_id === 'string' && persisted.role_id.trim().length > 0
                  ? persisted.role_id.trim()
                  : msg.roleId
              ),
              generationSeconds: persisted.generation_seconds ?? msg.generationSeconds,
              nextAction: persistedNextAction,
              nextActionReason: persistedNextActionReason,
            }
          }))
        } catch (err) {
          logApiError(err, 'ChatProvider.reconcileFinalAssistantMessage')
        }
      }

      const finalizeDone = (data: StreamDonePayload | null | undefined) => {
        clearStreamWatchdog()
        streamWatchdogTimedOutRef.current = false
        clearRevealTimer()
        streamRevealActiveRef.current = false
        streamPendingDoneRef.current = null
        setIsStreaming(false)
        const elapsed = data?.elapsed_seconds
        const messageId = data?.message_id
        const completionMode = normalizeCompletionMode(data?.completion_mode)
        const isPartial = completionMode === 'partial'
        const stoppedByUser = data?.stopped_by_user ?? (completionMode === 'stopped')
        const hasRemainingScope = data?.has_remaining_scope
          ?? (isPartial || completionMode === 'scoped_complete' || completionMode === 'stopped')
        const continuationPasses = typeof data?.continuation_passes === 'number'
          ? data.continuation_passes
          : 0
        const nextAction: 'none' | 'continue' | 'regenerate' | 'assistant_switch' = data?.next_action ?? 'none'
        const nextActionReason: 'stopped' | 'timeout' | 'unresolved_content' | 'budget_exhausted' | 'stalled' | 'out_of_corpus' | null =
          data?.next_action_reason ?? null
        const chatMode = isChatMode(data?.chat_mode) ? data.chat_mode : undefined
        const webSearchUsed = data?.web_search_used === true
          || (data?.budget_metrics != null
            && typeof data.budget_metrics === 'object'
            && (data.budget_metrics as Record<string, unknown>).web_search_used === true)
        const recoveryCallout = buildRecoveryCallout(nextAction, nextActionReason)
        const displayBlocks = Array.isArray(data?.display_blocks) ? data?.display_blocks : undefined
        const extraBlocks: DisplayBlock[] = [recoveryCallout].filter((v): v is DisplayBlock => v != null)
        const nextDisplayBlocks = extraBlocks.length > 0
          ? [...(displayBlocks || []), ...extraBlocks]
          : displayBlocks

        if (
          nextAction === 'continue'
          && continuationPasses > 0
          && typeof messageId === 'number'
        ) {
          lastAutoContinuedMessageIdRef.current = messageId
        }
        const continueLabel: 'Continue' | 'Continue Again' = (
          nextAction === 'continue'
          && typeof messageId === 'number'
          && lastAutoContinuedMessageIdRef.current === messageId
        ) ? 'Continue Again' : 'Continue'
        if (typeof messageId === 'number' && chatMode) {
          storeMessageMode(messageId, chatMode)
        }

        streamDraftRef.current = {
          ...(streamDraftRef.current || assistantDraft),
          id: messageId,
          content: streamContentRef.current,
          displayBlocks: nextDisplayBlocks,
          scopedFileName: effectiveFileScope?.filename ?? null,
          isStreaming: false,
          streamStatusText: undefined,
          streamSectionProgress: undefined,
          streamPlanSteps: undefined,
          isPartial,
          hasRemainingScope,
          completionMode,
          stoppedByUser,
          generationSeconds: elapsed,
          chatMode,
          roleId,
          nextAction,
          nextActionReason,
          continuationPasses,
          continueLabel,
          webSearchUsed,
        }
        if (isViewingGeneratingChat()) {
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last?.role === 'assistant' && last.isStreaming) {
              next[next.length - 1] = {
                ...last,
                id: messageId ?? last.id,
                content: streamContentRef.current,
                displayBlocks: nextDisplayBlocks ?? last.displayBlocks,
                isStreaming: false,
                streamStatusText: undefined,
                streamSectionProgress: undefined,
                streamPlanSteps: undefined,
                isPartial,
                hasRemainingScope,
                completionMode,
                stoppedByUser,
                generationSeconds: elapsed ?? last.generationSeconds,
                chatMode: chatMode ?? last.chatMode,
                roleId: roleId ?? last.roleId,
                scopedFileName: effectiveFileScope?.filename ?? last.scopedFileName ?? null,
                nextAction,
                nextActionReason,
                continuationPasses,
                continueLabel,
                webSearchUsed,
              }
            }
            return next
          })
        }
        streamIdRef.current = null
        streamChatIdRef.current = null
        streamRequestIdRef.current = null
        streamStopRequestedRef.current = false
        streamDraftRef.current = null
        setActiveGenerationChatId(null)
        setActiveGenerationRequestId(null)

        const finalizedChatId = streamChatIdRef.current ?? requestChatId ?? currentChatIdRef.current
        if (typeof messageId === 'number' && finalizedChatId) {
          void reconcileFinalAssistantMessage(messageId, finalizedChatId)
        }
      }

      const runCleanedReveal = () => {
        if (streamSessionRef.current !== sessionId) return
        const cleaned = streamCleanedContentRef.current || ''
        if (!cleaned) {
          streamRevealActiveRef.current = false
          if (streamPendingDoneRef.current) {
            finalizeDone(streamPendingDoneRef.current)
          }
          return
        }
        if (streamRevealCharIndexRef.current >= cleaned.length) {
          streamContentRef.current = cleaned
          streamRevealActiveRef.current = false
          if (streamPendingDoneRef.current) {
            finalizeDone(streamPendingDoneRef.current)
          }
          return
        }
        streamRevealCharIndexRef.current = Math.min(
          cleaned.length,
          streamRevealCharIndexRef.current + CLEANED_REVEAL_CHARS_PER_TICK,
        )
        streamContentRef.current = cleaned.slice(0, streamRevealCharIndexRef.current)
        streamDraftRef.current = {
          ...(streamDraftRef.current || assistantDraft),
          content: streamContentRef.current,
          isStreaming: true,
          streamStatusText: undefined,
          streamSectionProgress: undefined,
          streamPlanSteps: undefined,
        }
        applyStreamDraftToVisibleMessages()
        streamRevealTimerRef.current = setTimeout(runCleanedReveal, CLEANED_REVEAL_INTERVAL_MS)
      }

      touchStreamWatchdog()
      streamStatusTimerRef.current = setInterval(
        refreshStreamStatusMessage,
        STREAM_STATUS_TIMER_INTERVAL_MS,
      )
      await streamChat(message, requestChatId, {
        signal: abortControllerRef.current.signal,
        onToken: (token) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          // Keep collecting raw stream tokens for fallback and debugging, but do
          // not render half-baked drafts in the UI.
          streamContentRef.current += token
        },
        onChatId: (id) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          streamChatIdRef.current = id
          setActiveGenerationChatId(id)
          // Keep newly created chat selected for this send operation only.
          if (currentChatIdRef.current == null) {
            currentChatIdRef.current = id
            setCurrentChatIdState(id)
          }
          if (chatPrefsRef.current[id] == null && chatPrefsRef.current.__draft__) {
            chatPrefsRef.current[id] = chatPrefsRef.current.__draft__
          }
          if (chatFileScopesRef.current[id] == null && chatFileScopesRef.current.__draft__) {
            chatFileScopesRef.current[id] = chatFileScopesRef.current.__draft__
            setChatFileScope(chatFileScopesRef.current[id])
            persistStoredChatFileScopes(chatFileScopesRef.current)
          }
          delete chatPrefsRef.current.__draft__
          delete chatFileScopesRef.current.__draft__
          updateCurrentChat(id).catch((err) => logApiError(err, 'ChatProvider.sendMessage.onChatId'))
        },
        onStreamId: (id) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          streamIdRef.current = id
        },
        onRequestId: (id) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          streamRequestIdRef.current = id
          setActiveGenerationRequestId(id)
        },
        onCleaned: (cleanedAnswer) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          clearRevealTimer()
          streamCleanedContentRef.current = cleanedAnswer
          streamRevealCharIndexRef.current = 0
          streamRevealActiveRef.current = true
          streamContentRef.current = ''
          streamPlanStepsRef.current = []
          streamDraftRef.current = {
            ...(streamDraftRef.current || assistantDraft),
            content: '',
            isStreaming: true,
            streamStatusText: undefined,
            streamSectionProgress: undefined,
            streamPlanSteps: undefined,
          }
          if (streamThrottleRef.current) {
            clearTimeout(streamThrottleRef.current)
            streamThrottleRef.current = null
          }
          applyStreamDraftToVisibleMessages()
          runCleanedReveal()
        },
        onSources: (sources) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          if (streamThrottleRef.current) {
            clearTimeout(streamThrottleRef.current)
            streamThrottleRef.current = null
          }
          streamDraftRef.current = {
            ...(streamDraftRef.current || assistantDraft),
            content: streamContentRef.current,
            sources: sources || [],
            isStreaming: true,
          }
          applyStreamDraftToVisibleMessages()
        },
        onStatus: (status) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          const nextStatusText = typeof status?.message === 'string' && status.message.trim()
            ? status.message.trim()
            : (status?.state ? getStreamStatusLabel(status.state) : undefined)
          if (nextStatusText) {
            streamStatusBaseMessageRef.current = nextStatusText
          }
          if (typeof status?.state === 'string' && status.state.trim()) {
            const nextState = status.state.trim().toLowerCase()
            if (nextState !== streamStatusStateRef.current) {
              streamStatusStartMsRef.current = Date.now()
            }
            streamStatusStateRef.current = nextState
          }
          const sectionProgressPayload = status?.section_progress
          const normalizedSectionProgress =
            sectionProgressPayload
            && Array.isArray(sectionProgressPayload.completed)
            && Array.isArray(sectionProgressPayload.remaining)
            && typeof sectionProgressPayload.total === 'number'
              ? {
                  completed: sectionProgressPayload.completed.filter((v): v is string => typeof v === 'string'),
                  remaining: sectionProgressPayload.remaining.filter((v): v is string => typeof v === 'string'),
                  total: sectionProgressPayload.total,
                }
              : undefined
          if (!nextStatusText && !normalizedSectionProgress) return
          streamDraftRef.current = {
            ...(streamDraftRef.current || assistantDraft),
            streamStatusText: nextStatusText
              ? formatStreamStatusWithLatency(
                  nextStatusText,
                  streamStatusStateRef.current,
                  (Date.now() - streamStatusStartMsRef.current) / 1000,
                )
              : undefined,
            streamSectionProgress: normalizedSectionProgress,
            isStreaming: true,
          }
          applyStreamDraftToVisibleMessages()
        },
        onPlanStep: (payload: PlanStepPayload) => {
          if (streamSessionRef.current !== sessionId) return
          touchStreamWatchdog()
          const { step_id, description, status } = payload
          if (typeof step_id !== 'number' || typeof description !== 'string') return
          const stepStatus = status === 'done' || status === 'empty' ? status : 'running'
          const existing = streamPlanStepsRef.current
          const idx = existing.findIndex(s => s.step_id === step_id)
          if (idx >= 0) {
            const updated = [...existing]
            updated[idx] = { step_id, description, status: stepStatus }
            streamPlanStepsRef.current = updated
          } else {
            streamPlanStepsRef.current = [...existing, { step_id, description, status: stepStatus }]
          }
          streamDraftRef.current = {
            ...(streamDraftRef.current || assistantDraft),
            streamPlanSteps: [...streamPlanStepsRef.current],
            isStreaming: true,
          }
          applyStreamDraftToVisibleMessages()
        },
        onDone: (data) => {
          if (streamSessionRef.current !== sessionId) return
          clearStreamWatchdog()
          clearStreamStatusTimer()
          streamWatchdogTimedOutRef.current = false
          if (streamThrottleRef.current) {
            clearTimeout(streamThrottleRef.current)
            streamThrottleRef.current = null
          }
          if (streamRevealActiveRef.current) {
            streamPendingDoneRef.current = data as StreamDonePayload
            return
          }
          finalizeDone(data as StreamDonePayload)
        },
        onError: (err) => {
          if (streamSessionRef.current !== sessionId) return
          clearStreamWatchdog()
          clearStreamStatusTimer()
          clearRevealTimer()
          streamRevealActiveRef.current = false
          streamPendingDoneRef.current = null
          if (streamThrottleRef.current) {
            clearTimeout(streamThrottleRef.current)
            streamThrottleRef.current = null
          }
          setIsStreaming(false)
          const content = streamContentRef.current
          const isAbort = err.name === 'AbortError'
          const userRequestedStop = streamStopRequestedRef.current
          if (isAbort) {
            if (streamWatchdogTimedOutRef.current) {
              const timeoutMsg = STREAM_WATCHDOG_TIMEOUT_MESSAGE
              setError(timeoutMsg)
              showToast('error', timeoutMsg)
              if (isViewingGeneratingChat()) {
                setMessages((prev) => {
                  const next = [...prev]
                  const last = next[next.length - 1]
                  if (last?.role === 'assistant') {
                    next[next.length - 1] = {
                      ...last,
                      content: content || STREAM_WATCHDOG_INTERRUPTED_MESSAGE,
                      isStreaming: false,
                      streamStatusText: undefined,
                      streamSectionProgress: undefined,
                      streamPlanSteps: undefined,
                      completionMode: 'partial',
                      hasRemainingScope: true,
                      nextAction: 'regenerate',
                      nextActionReason: 'timeout',
                    }
                  }
                  return next
                })
              }
              streamIdRef.current = null
              streamChatIdRef.current = null
              streamRequestIdRef.current = null
              streamCleanedContentRef.current = null
              streamDraftRef.current = null
              streamStopRequestedRef.current = false
              setActiveGenerationChatId(null)
              setActiveGenerationRequestId(null)
              streamWatchdogTimedOutRef.current = false
              return
            }
            streamDraftRef.current = null
            if (isViewingGeneratingChat()) {
              setMessages((prev) => {
                const next = [...prev]
                const last = next[next.length - 1]
                if (last?.role === 'assistant') {
                  const stoppedCallout = userRequestedStop ? buildRecoveryCallout('regenerate', 'stopped') : null
                  next[next.length - 1] = {
                    ...last,
                    content,
                    displayBlocks: stoppedCallout
                      ? [...(last.displayBlocks || []), stoppedCallout]
                      : last.displayBlocks,
                    isStreaming: false,
                    streamStatusText: undefined,
                    streamSectionProgress: undefined,
                    streamPlanSteps: undefined,
                    completionMode: userRequestedStop ? 'stopped' : 'partial',
                    stoppedByUser: userRequestedStop,
                    hasRemainingScope: true,
                    nextAction: 'regenerate',
                    nextActionReason: userRequestedStop ? 'stopped' : 'timeout',
                    continueLabel: 'Continue',
                  }
                }
                return next
              })
            }
            streamIdRef.current = null
            streamChatIdRef.current = null
            streamRequestIdRef.current = null
            streamCleanedContentRef.current = null
            streamStopRequestedRef.current = false
            setActiveGenerationChatId(null)
            setActiveGenerationRequestId(null)
            streamWatchdogTimedOutRef.current = false
            return
          }
          // Keep explicit 429 handling here: backend uses this code when another
          // generation is active, and we intentionally surface a stable UX message.
          const msg = err instanceof ApiError
            ? (err.status === 429 ? ACTIVE_GENERATION_REJECT_MESSAGE : err.detail)
            : (err.message || 'Failed to send message')
          setError(msg)
          showToast('error', msg)
          const errContent = streamContentRef.current || 'Response was interrupted.'
          if (isViewingGeneratingChat()) {
            setMessages((prev) => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.role === 'assistant') {
                next[next.length - 1] = {
                  ...last,
                  content: errContent,
                  sources: errContent ? last.sources : [],
                  isStreaming: false,
                  streamStatusText: undefined,
                  streamSectionProgress: undefined,
                  streamPlanSteps: undefined,
                }
              }
              return next
            })
          }
          streamIdRef.current = null
          streamChatIdRef.current = null
          streamRequestIdRef.current = null
          streamCleanedContentRef.current = null
          streamStopRequestedRef.current = false
          streamDraftRef.current = null
          setActiveGenerationChatId(null)
          setActiveGenerationRequestId(null)
          streamWatchdogTimedOutRef.current = false
        },
      }, {
        mode: chatMode,
        roleId,
        requestId,
        fileId: effectiveFileScope?.fileId ?? null,
        scopedUploadIds: effectiveScopedUploadIds.length > 0 ? effectiveScopedUploadIds : null,
        chatWebSearchEnabled,
        chatWebSearchPrivacyOverride,
      })
    } finally {
      clearStreamWatchdog()
      clearStreamStatusTimer()
      sendInFlightRef.current = false
    }
  }, [
    applyStreamDraftToVisibleMessages,
    chatUploads,
    clearRevealTimer,
    clearStreamStatusTimer,
    clearStreamWatchdog,
    isViewingGeneratingChat,
    resolveDraftOrChatFileScope,
  ])

  const continueLastScope = useCallback(async (
    anchorMessageId?: number,
    options?: {
      mode?: ChatMode
      roleId?: string | null
      fileScope?: ChatFileScope | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => {
    if (typeof anchorMessageId === 'number') {
      lastAutoContinuedMessageIdRef.current = anchorMessageId
    }
    await sendMessage(
      CONTINUE_SCOPED_PROMPT,
      {
        isInternal: true,
        mode: options?.mode ?? 'researcher',
        roleId: options?.roleId ?? null,
        fileScope: options?.fileScope ?? null,
        chatWebSearchEnabled: options?.chatWebSearchEnabled ?? false,
        chatWebSearchPrivacyOverride: options?.chatWebSearchPrivacyOverride ?? false,
      },
    )
  }, [sendMessage])

  const stopStreaming = useCallback(async (): Promise<boolean> => {
    return stopStreamingInternal()
  }, [stopStreamingInternal])

  const newChat = useCallback(async () => {
    setForceNewChatFlag(true)
    chatLoadSessionRef.current += 1
    setLoadingChat(false)
    currentChatIdRef.current = null
    setCurrentChatIdState(null)
    setCurrentChatLockedMode(null)
    setCurrentChatLockedRoleId(null)
    setChatWebSearchEnabled(false)
    setChatWebSearchPrivacyOverride(false)
    setChatFileScope(null)
    setChatUploads([])
    chatPrefsRef.current.__draft__ = { enabled: false, privacyOverride: false }
    delete chatFileScopesRef.current.__draft__
    lastAutoContinuedMessageIdRef.current = null
    setMessages([])
    setError(null)
    await updateCurrentChat(null).catch((err) => logApiError(err, 'ChatProvider.newChat'))
  }, [])

  const startScopedChat = useCallback(async (scope: ChatFileScope) => {
    const normalizedScope: ChatFileScope = {
      fileId: Math.trunc(Number(scope.fileId)),
      filename: String(scope.filename || '').trim() || `File ${Math.trunc(Number(scope.fileId))}`,
    }
    if (!Number.isFinite(normalizedScope.fileId) || normalizedScope.fileId <= 0) return
    await newChat()
    chatFileScopesRef.current.__draft__ = normalizedScope
    setChatFileScope(normalizedScope)
  }, [newChat])

  useEffect(() => {
    return () => {
      if (streamThrottleRef.current) {
        clearTimeout(streamThrottleRef.current)
      }
      if (streamRevealTimerRef.current) {
        clearTimeout(streamRevealTimerRef.current)
      }
      if (streamWatchdogTimerRef.current) {
        clearTimeout(streamWatchdogTimerRef.current)
      }
      if (streamStatusTimerRef.current) {
        clearInterval(streamStatusTimerRef.current)
      }
      abortControllerRef.current?.abort()
    }
  }, [])

  return (
    <ChatContext.Provider
      value={{
        currentChatId,
        currentChatLockedMode,
        currentChatLockedRoleId,
        setCurrentChatId,
        activeGenerationChatId,
        activeGenerationRequestId,
        hasActiveGenerationForCurrentChat: !!(isStreaming && activeGenerationChatId && currentChatId === activeGenerationChatId),
        messages,
        isStreaming,
        loadingChat,
        error,
        enableRawOutputControl,
        chatWebSearchEnabled,
        chatWebSearchPrivacyOverride,
        chatFileScope,
        chatUploads,
        setChatWebSearchPreferences,
        startScopedChat,
        clearChatFileScope,
        uploadFiles,
        removeUploadedFile,
        selectChat,
        goToGeneratingChat,
        sendMessage,
        continueLastScope,
        stopStreaming,
        newChat,
        clearError,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}
