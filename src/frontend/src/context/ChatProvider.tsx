/**
 * Informity AI — Chat provider
 * Owns chat stream/session state so navigation does not interrupt streaming.
 */
import { useState, useCallback, useEffect, useRef, type ReactNode } from 'react'
import { ChatContext } from './chatContext'
import { ApiError, getChat, getSettings, stopChatStream, streamChat, updateChatPreferences, updateCurrentChat } from '../api'
import { showToast } from './useToast'
import { logApiError } from '../utils/logApiError'
import { FORCE_NEW_CHAT_KEY, MESSAGE_MODE_MAP_STORAGE_KEY } from '../utils/storageKeys'
import type {
  ChatMode,
  ChatMessageApi,
  ChatMessageDisplay,
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
const STREAM_STATUS_LABELS: Record<string, string> = {
  classifying: 'Analyzing your request...',
  retrieving: 'Searching for relevant information...',
  searching: 'Searching the web...',
  generating: 'Generating response...',
  continuing: 'Continuing response...',
  finalizing: 'Finalizing answer...',
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

function createChatRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [currentChatId, setCurrentChatIdState] = useState<string | null>(null)
  const [activeGenerationChatId, setActiveGenerationChatId] = useState<string | null>(null)
  const [activeGenerationRequestId, setActiveGenerationRequestId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessageDisplay[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [loadingChat, setLoadingChat] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [enableRawOutputControl, setEnableRawOutputControl] = useState(false)
  const [chatWebSearchEnabled, setChatWebSearchEnabled] = useState(false)
  const [chatWebSearchPrivacyOverride, setChatWebSearchPrivacyOverride] = useState(false)
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
  const streamPlanStepsRef = useRef<Array<{ step_id: number; description: string; status: 'running' | 'done' | 'empty' }>>([])
  const streamStopRequestedRef = useRef(false)
  const currentChatIdRef = useRef<string | null>(null)
  const isStreamingRef = useRef(false)
  const messagesCountRef = useRef(0)
  const lastAutoContinuedMessageIdRef = useRef<number | null>(null)
  const chatPrefsRef = useRef<Record<string, { enabled: boolean; privacyOverride: boolean }>>({})

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

  useEffect(() => {
    getSettings()
      .then((s) => {
        const data = s as { enable_raw_output_control?: boolean }
        setEnableRawOutputControl(!!data?.enable_raw_output_control)
      })
      .catch((err) => logApiError(err, 'ChatProvider.getSettings'))
  }, [])

  const clearError = useCallback(() => setError(null), [])

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
      const data = (await getChat(selectedChatId)) as GetChatResponse
      if (chatLoadSessionRef.current !== sessionId) return
      const historyMessages = data.messages || []
      const resolvedPrefs = {
        enabled: data.chat_web_search_enabled === true,
        privacyOverride: data.chat_web_search_privacy_override === true,
      }
      chatPrefsRef.current[selectedChatId] = resolvedPrefs
      setChatWebSearchEnabled(resolvedPrefs.enabled)
      setChatWebSearchPrivacyOverride(resolvedPrefs.privacyOverride)
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
        const inferredAssistantMode: ChatMode | undefined = (
          m.role === 'assistant'
            ? (
                storedMode
                ?? (isChatMode(m.chat_mode) ? m.chat_mode : undefined)
                ?? (nextActionReason === 'out_of_corpus' ? 'researcher' : undefined)
                ?? ((m.sources?.length || 0) > 0 ? 'researcher' : undefined)
              )
            : undefined
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
          timeoutReason: null,
          nextAction,
          nextActionReason,
          continueLabel: 'Continue',
          createdAt: m.created_at,
          generationSeconds: m.generation_seconds,
          chatMode: inferredAssistantMode,
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
        currentChatIdRef.current = null
        setCurrentChatIdState(null)
        setChatWebSearchEnabled(false)
        setChatWebSearchPrivacyOverride(false)
        setMessages([])
        setError(null)
        updateCurrentChat(null).catch((e) => logApiError(e, 'ChatProvider.selectChat.clearCurrentChat'))
      } else {
        const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to load chat'
        setError(msg)
        showToast('error', msg)
      }
    } finally {
      if (chatLoadSessionRef.current === sessionId) {
        setLoadingChat(false)
      }
    }
  }, [currentChatId])

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
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => {
    const message = text.trim()
    const isInternalMessage = !!options?.isInternal
    const chatMode: ChatMode = options?.mode ?? 'researcher'
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
    if (!isInternalMessage) {
      setForceNewChatFlag(false)
      lastAutoContinuedMessageIdRef.current = null
    }
    sendInFlightRef.current = true

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
    const requestChatId = currentChatIdRef.current
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
        const timeoutReason = data?.timeout_reason ?? null
        const continuationPasses = typeof data?.continuation_passes === 'number'
          ? data.continuation_passes
          : 0
        const continuationResolutionReason = data?.continuation_resolution_reason ?? null
        const continuationProgressState = data?.continuation_progress_state
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
          isStreaming: false,
          streamStatusText: undefined,
          streamSectionProgress: undefined,
          streamPlanSteps: undefined,
          isPartial,
          hasRemainingScope,
          completionMode,
          stoppedByUser,
          timeoutReason,
          generationSeconds: elapsed,
          chatMode,
          continuationResolutionReason,
          continuationProgressState: continuationProgressState ?? null,
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
                timeoutReason,
                generationSeconds: elapsed ?? last.generationSeconds,
                chatMode: chatMode ?? last.chatMode,
                continuationResolutionReason,
                continuationProgressState: continuationProgressState ?? last.continuationProgressState ?? null,
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
          delete chatPrefsRef.current.__draft__
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
            streamStatusText: nextStatusText,
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
        requestId,
        chatWebSearchEnabled,
        chatWebSearchPrivacyOverride,
      })
    } finally {
      clearStreamWatchdog()
      sendInFlightRef.current = false
    }
  }, [applyStreamDraftToVisibleMessages, clearRevealTimer, clearStreamWatchdog, isViewingGeneratingChat])

  const continueLastScope = useCallback(async (
    anchorMessageId?: number,
    options?: {
      mode?: ChatMode
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
    setChatWebSearchEnabled(false)
    setChatWebSearchPrivacyOverride(false)
    chatPrefsRef.current.__draft__ = { enabled: false, privacyOverride: false }
    lastAutoContinuedMessageIdRef.current = null
    setMessages([])
    setError(null)
    await updateCurrentChat(null).catch((err) => logApiError(err, 'ChatProvider.newChat'))
  }, [])

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
      abortControllerRef.current?.abort()
    }
  }, [])

  return (
    <ChatContext.Provider
      value={{
        currentChatId,
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
        setChatWebSearchPreferences,
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
