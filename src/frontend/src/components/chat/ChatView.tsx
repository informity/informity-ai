/**
 * Informity AI — Chat view
 * Full-height message list, input, SSE streaming.
 */
import { useState, useRef, useEffect, useCallback, useLayoutEffect, useMemo } from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatMessageSkeleton } from './ChatMessageSkeleton'
import { PageHeader } from '../PageHeader'
import { ServiceUnavailableState } from '../ServiceUnavailableState'
import { useChatContext } from '../../context/useChatContext'
import { useBackendStatus } from '../../context/useBackendStatus'
import { useOptionalTranslateContext } from '../../context/useTranslateContext'
import { exportChatMarkdown, getCurrentChat, getSpecializations, getSettings } from '../../api'
import { markdownToPlainText, downloadTextFile } from '../../utils/downloadHelpers'
import { showToast } from '../../context/useToast'
import {
  isChatMode,
  type ChatFileScope,
  type ChatMessageDisplay,
  type ChatMode,
  type ChatSpecializationDefinition,
} from '../../types/api'
import { logApiError } from '../../utils/logApiError'
import {
  CHAT_MODE_STORAGE_KEY,
  CHAT_SPECIALIZATION_ID_STORAGE_KEY,
  FORCE_NEW_CHAT_KEY,
} from '../../utils/storageKeys'
import { CHAT_MODE_ICONS, CHAT_MODE_LABELS } from '../../utils/chatModeConfig'
import { getFileIcon } from '../../utils/fileFormatting'
import {
  COMPOSER_TEXTAREA_MIN_HEIGHT,
  COMPOSER_TEXTAREA_MAX_HEIGHT,
  COMPOSER_SCOPED_EXTRA_HEIGHT,
} from '../../utils/composerSizing'
import './ChatView.css'
const UPLOAD_CHIP_FALLBACK_WIDTH = 180
const UPLOAD_OVERFLOW_CHIP_FALLBACK_WIDTH = 52
const UPLOAD_PENDING_CHIP_FALLBACK_WIDTH = 116
const ALL_CHAT_MODES: ChatMode[] = ['assistant', 'researcher']
const GENERAL_SPECIALIZATION_LABEL = 'General Assistant'

interface ChatViewProps {
  prefillMessage?: string
  initialChatId?: string | null
  initialScopedFile?: ChatFileScope | null
}

interface GetCurrentChatResponse {
  current_chat_id?: string
}
interface ChatSettingsResponse {
  default_chat_mode?: ChatMode
  full_privacy?: boolean
  web_search_configured?: boolean
  enabled_specialization_ids?: string[]
}

interface SettingsUpdatedEvent extends Event {
  detail?: ChatSettingsResponse
}

function triggerMarkdownDownload(filename: string, markdown: string): void {
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = String(filename || 'chat-export.md').trim() || 'chat-export.md'
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

function resolveFileIconFromFilename(filename: string): string {
  const normalized = String(filename || '').trim()
  const dotIndex = normalized.lastIndexOf('.')
  if (dotIndex <= 0 || dotIndex >= normalized.length - 1) {
    return getFileIcon(undefined)
  }
  return getFileIcon(normalized.slice(dotIndex + 1))
}

function resolveChatModeFromHistory(history: ChatMessageDisplay[]): ChatMode | null {
  for (let i = history.length - 1; i >= 0; i -= 1) {
    const candidate = history[i]?.chatMode
    if (isChatMode(candidate)) {
      return candidate
    }
  }
  return null
}

function resolveLockedMode(history: ChatMessageDisplay[]): ChatMode | null {
  return (
    history.find((msg) => msg.role === 'user' && !msg.isInternal && isChatMode(msg.chatMode))?.chatMode
    ?? history.find((msg) => msg.role === 'assistant' && isChatMode(msg.chatMode))?.chatMode
    ?? null
  )
}

function resolveLockedSpecializationId(history: ChatMessageDisplay[]): string | null {
  return (
    history.find((msg) => msg.role === 'user' && !msg.isInternal && !!msg.specializationId)?.specializationId
    ?? history.find((msg) => msg.role === 'assistant' && !!msg.specializationId)?.specializationId
    ?? null
  )
}

function formatSpecializationNameFromId(specializationId: string): string {
  return specializationId
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

export function ChatView({ prefillMessage = '', initialChatId = null, initialScopedFile = null }: ChatViewProps) {
  const { offline } = useBackendStatus()
  const translateCtx = useOptionalTranslateContext()
  const isTranslating = translateCtx?.isTranslating ?? false
  const translateTargetLanguage = translateCtx?.targetLanguage ?? null
  const translateTone = translateCtx?.tone ?? null
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const exportMenuRef = useRef<HTMLDivElement>(null)
  const {
    currentChatId: contextChatId,
    currentChatLockedMode,
    currentChatLockedSpecializationId,
    activeChatTranslation,
    isTranslatingReply,
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
    sendMessage,
    continueLastScope,
    stopStreaming,
    newChat,
    clearError,
    translateAssistantMessage,
  } = useChatContext()
  const [translationElapsedSeconds, setTranslationElapsedSeconds] = useState(0)
  const [inputValue, setInputValue] = useState(prefillMessage)
  const [chatMode, setChatMode] = useState<ChatMode>('researcher')
  const [, setDefaultChatMode] = useState<ChatMode>('researcher')
  const [fullPrivacyMode, setFullPrivacyMode] = useState(true)
  const [webSearchConfigured, setWebSearchConfigured] = useState(false)
  const [modeMenuOpen, setModeMenuOpen] = useState(false)
  const [enabledSpecializationIds, setEnabledSpecializationIds] = useState<string[]>([])
  const [hasConfiguredEnabledSpecializationIds, setHasConfiguredEnabledSpecializationIds] = useState(false)
  const [selectedSpecializationId, setSelectedSpecializationId] = useState<string | null>(null)
  const [specializations, setSpecializations] = useState<ChatSpecializationDefinition[]>([])
  const [specializationsLoaded, setSpecializationsLoaded] = useState(false)
  const [specializationMenuOpen, setSpecializationMenuOpen] = useState(false)
  const [showScrollToBottom, setShowScrollToBottom] = useState(false)
  const [animateToDocked, setAnimateToDocked] = useState(false)
  const [textareaCanScroll, setTextareaCanScroll] = useState(false)
  const [textareaHasTopScroll, setTextareaHasTopScroll] = useState(false)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const newChatRequestedRef = useRef(false)
  const consumedInitialChatIdRef = useRef<string | null>(null)
  const wasCenteredComposerRef = useRef(false)
  const dockAnimationTimerRef = useRef<number | null>(null)
  const scrollRafRef = useRef<number | null>(null)
  const autoFollowRafRef = useRef<number | null>(null)
  const showScrollRef = useRef(false)
  const isNearBottomRef = useRef(true)
  const modeMenuRef = useRef<HTMLDivElement>(null)
  const specializationMenuRef = useRef<HTMLDivElement>(null)
  const consumedInitialScopeRef = useRef<string | null>(null)
  const skipNextSelectChatIdRef = useRef<string | null>(null)
  const draftPendingAliasChatIdRef = useRef<string | null>(null)
  const uploadInputRef = useRef<HTMLInputElement>(null)
  const uploadChipsContainerRef = useRef<HTMLDivElement>(null)
  const uploadChipMeasureRefs = useRef<Record<string, HTMLSpanElement | null>>({})
  const uploadOverflowMeasureRef = useRef<HTMLSpanElement>(null)
  const uploadPendingMeasureRef = useRef<HTMLSpanElement>(null)
  const specializationSelectionExplicitRef = useRef(false)
  const [visibleUploadCount, setVisibleUploadCount] = useState(chatUploads.length)
  const [pendingUploadCountsByChat, setPendingUploadCountsByChat] = useState<Record<string, number>>({})
  const [isDragOverComposer, setIsDragOverComposer] = useState(false)
  const uploadDragDepthRef = useRef(0)
  const hasAssistantReply = useMemo(
    () => messages.some((msg) => msg.role === 'assistant' && !msg.isInternal && !!msg.content?.trim()),
    [messages],
  )

  const pendingUploadCount = (() => {
    if (!contextChatId) return pendingUploadCountsByChat.__draft__ ?? 0
    const scopedCount = pendingUploadCountsByChat[contextChatId] ?? 0
    const draftAliasCount = (
      draftPendingAliasChatIdRef.current === contextChatId
        ? (pendingUploadCountsByChat.__draft__ ?? 0)
        : 0
    )
    return Math.max(scopedCount, draftAliasCount)
  })()

  useEffect(() => {
    if (!activeChatTranslation || activeChatTranslation.chatId !== contextChatId) {
      setTranslationElapsedSeconds(0)
      return
    }
    const startedAt = Number(activeChatTranslation.startedAt)
    const updateElapsed = () => {
      const elapsed = Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
      setTranslationElapsedSeconds(elapsed)
    }
    updateElapsed()
    const intervalId = window.setInterval(updateElapsed, 1000)
    return () => window.clearInterval(intervalId)
  }, [activeChatTranslation, contextChatId])

  const renderedMessages = useMemo(() => {
    if (!activeChatTranslation || activeChatTranslation.chatId !== contextChatId) return messages
    const sourceMessageId = activeChatTranslation.sourceMessageId
    const filteredMessages = messages.filter((message) => message.translatedFromMessageId !== sourceMessageId)
    const pendingLanguage = activeChatTranslation.targetLanguage || 'selected language'
    const elapsedLabel = translationElapsedSeconds > 0 ? ` ${translationElapsedSeconds}s` : ''
    const placeholder: ChatMessageDisplay = {
      role: 'assistant',
      content: '',
      isStreaming: true,
      streamStatusText: `Translating to ${pendingLanguage}…${elapsedLabel}`,
      chatMode: currentChatLockedMode ?? 'assistant',
      specializationId: currentChatLockedSpecializationId,
      translatedFromMessageId: activeChatTranslation.sourceMessageId,
      translationLanguage: pendingLanguage,
      translationTone: activeChatTranslation.tone,
      translationIsStale: false,
      sources: [],
      displayBlocks: [],
      isInternal: false,
    }
    const next: ChatMessageDisplay[] = []
    let inserted = false
    for (const message of filteredMessages) {
      next.push(message)
      if (!inserted && message.id === sourceMessageId) {
        next.push(placeholder)
        inserted = true
      }
    }
    if (!inserted) next.push(placeholder)
    return next
  }, [activeChatTranslation, contextChatId, currentChatLockedMode, currentChatLockedSpecializationId, messages, translationElapsedSeconds])

  const isForceNewChatRequested = useCallback((): boolean => {
    try {
      return (
        window.localStorage.getItem(FORCE_NEW_CHAT_KEY) === '1'
        || window.sessionStorage.getItem(FORCE_NEW_CHAT_KEY) === '1'
      )
    } catch {
      return false
    }
  }, [])

  useEffect(() => {
    if (prefillMessage) setInputValue(prefillMessage)
  }, [prefillMessage])

  useEffect(() => {
    if (!offline && !isStreaming && chatMode === 'researcher') return
    uploadDragDepthRef.current = 0
    setIsDragOverComposer(false)
  }, [chatMode, isStreaming, offline])

  useEffect(() => {
    let cancelled = false
    let specializationsCancelled = false
    let hasStoredMode = false
    try {
      const raw = window.localStorage.getItem(CHAT_MODE_STORAGE_KEY)
      if (isChatMode(raw)) {
        hasStoredMode = true
        setChatMode(raw)
      }
      const storedSpecializationId = String(
        window.localStorage.getItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
        || ''
      ).trim()
      if (storedSpecializationId) setSelectedSpecializationId(storedSpecializationId)
    } catch {
      // ignore storage errors
    }
    getSettings()
      .then((data) => {
        if (cancelled) return
        const settings = (data as ChatSettingsResponse | null | undefined)
        const mode = settings?.default_chat_mode
        setFullPrivacyMode(!!settings?.full_privacy)
        setWebSearchConfigured(!!settings?.web_search_configured)
        if (Array.isArray(settings?.enabled_specialization_ids)) {
          setEnabledSpecializationIds(settings.enabled_specialization_ids)
          setHasConfiguredEnabledSpecializationIds(true)
        } else {
          setEnabledSpecializationIds([])
          setHasConfiguredEnabledSpecializationIds(false)
        }
        if (isChatMode(mode)) {
          setDefaultChatMode(mode)
        }
        if (!hasStoredMode && isChatMode(mode)) {
          setChatMode(mode)
          try {
            window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, mode)
          } catch {
            // ignore storage errors
          }
        }
      })
      .catch((err) => logApiError(err, 'ChatView.getSettings.default_chat_mode'))
    getSpecializations()
      .then((items) => {
        if (specializationsCancelled) return
        setSpecializations(Array.isArray(items) ? items : [])
        setSpecializationsLoaded(true)
      })
      .catch((err) => {
        if (specializationsCancelled) return
        setSpecializationsLoaded(true)
        logApiError(err, 'ChatView.getSpecializations')
      })
    return () => {
      cancelled = true
      specializationsCancelled = true
    }
  }, [])

  useEffect(() => {
    const handleSettingsUpdated = (event: Event) => {
      const detail = (event as SettingsUpdatedEvent).detail
      if (!detail) return
      if (typeof detail.full_privacy === 'boolean') {
        setFullPrivacyMode(detail.full_privacy)
      }
      if (typeof detail.web_search_configured === 'boolean') {
        setWebSearchConfigured(detail.web_search_configured)
      }
      if (Array.isArray(detail.enabled_specialization_ids)) {
        setEnabledSpecializationIds(detail.enabled_specialization_ids)
        setHasConfiguredEnabledSpecializationIds(true)
      }
      if (isChatMode(detail.default_chat_mode)) {
        setDefaultChatMode(detail.default_chat_mode)
      }
    }
    window.addEventListener('settings-updated', handleSettingsUpdated as EventListener)
    return () => {
      window.removeEventListener('settings-updated', handleSettingsUpdated as EventListener)
    }
  }, [])

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (modeMenuRef.current && !modeMenuRef.current.contains(target)) {
        setModeMenuOpen(false)
      }
      if (specializationMenuRef.current && !specializationMenuRef.current.contains(target)) {
        setSpecializationMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  useEffect(() => {
    if (!exportMenuOpen) return
    const h = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) setExportMenuOpen(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [exportMenuOpen])

  useEffect(() => {
    if (offline || isStreaming) {
      setModeMenuOpen(false)
      setSpecializationMenuOpen(false)
    }
  }, [offline, isStreaming])

  useEffect(() => {
    if (!specializationsLoaded) return
    if (!selectedSpecializationId) return
    if (messages.length > 0) return
    if (!specializations.some((specialization) => specialization.id === selectedSpecializationId)) {
      setSelectedSpecializationId(null)
      specializationSelectionExplicitRef.current = false
      try {
        window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
      } catch {
        // ignore storage errors
      }
      return
    }
    const normalizedEnabledSpecializationIds = hasConfiguredEnabledSpecializationIds
      ? enabledSpecializationIds
      : specializations.map((specialization) => specialization.id)
    if (normalizedEnabledSpecializationIds.includes(selectedSpecializationId)) return
    setSelectedSpecializationId(null)
    specializationSelectionExplicitRef.current = false
    try {
      window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
    } catch {
      // ignore storage errors
    }
  }, [specializations, selectedSpecializationId, specializationsLoaded, messages.length, enabledSpecializationIds, hasConfiguredEnabledSpecializationIds])

  useEffect(() => {
    // Route-selected chat id should be consumed once per incoming value.
    // Re-applying it on every context mismatch prevents explicit "New Chat".
    if (initialChatId) {
      if (consumedInitialChatIdRef.current !== initialChatId) {
        consumedInitialChatIdRef.current = initialChatId
        selectChat(initialChatId)
      }
      return
    }
    if (contextChatId) {
      if (skipNextSelectChatIdRef.current === contextChatId) {
        skipNextSelectChatIdRef.current = null
        return
      }
      selectChat(contextChatId)
      return
    }
    if (isForceNewChatRequested()) return
    // Skip getCurrentChat when user explicitly requested a New Chat (avoids race with
    // updateCurrentChat(null) still in flight returning stale chat id)
    if (newChatRequestedRef.current) {
      newChatRequestedRef.current = false
      return
    }
    let cancelled = false
    getCurrentChat()
      .then((data) => {
        const d = data as GetCurrentChatResponse
        if (!cancelled && d?.current_chat_id) {
          selectChat(d.current_chat_id)
        }
      })
      .catch((err) => logApiError(err, 'ChatView.getCurrentChat'))
    return () => {
      cancelled = true
    }
  }, [initialChatId, contextChatId, selectChat, isForceNewChatRequested])

  useEffect(() => {
    if (!initialScopedFile) return
    const scopeKey = `${initialScopedFile.fileId}:${initialScopedFile.filename}`
    if (consumedInitialScopeRef.current === scopeKey) return
    consumedInitialScopeRef.current = scopeKey
    setChatMode('researcher')
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, 'researcher')
    } catch {
      // ignore storage errors
    }
    clearError()
    void startScopedChat(initialScopedFile)
  }, [clearError, initialScopedFile, startScopedChat])

  const hasUploadAttachments = chatUploads.length > 0
  const hasActiveUploadAttachments = chatUploads.some((item) => ['uploading', 'indexing', 'ready'].includes(String(item.state)))
  const hasPendingUploads = pendingUploadCount > 0
  const lockedMode = currentChatLockedMode ?? resolveLockedMode(messages)
  const lockedSpecializationId = currentChatLockedSpecializationId ?? resolveLockedSpecializationId(messages)
  const effectiveChatMode: ChatMode = lockedMode ?? chatMode
  const hasUploadChipRow = effectiveChatMode === 'researcher' && (hasUploadAttachments || hasPendingUploads)
  const hasScopedInputPill = effectiveChatMode === 'researcher' && (!!chatFileScope || hasUploadChipRow)
  const hideAssistantSwitch = effectiveChatMode === 'researcher' && hasScopedInputPill
  // Existing chat threads lock only session identity controls (mode/specialization).
  // The rest of chat interactions (send, uploads, scope clear, etc.) remain active.
  const modeSpecializationSessionLocked = (
    !!contextChatId
    && messages.some((msg) => (msg.role === 'user' && !msg.isInternal) || msg.role === 'assistant')
  )
  const effectiveSpecializationId = lockedSpecializationId ?? selectedSpecializationId
  const hasSpecializationDocContext = !!chatFileScope || chatUploads.some((item) => item.state === 'ready')
  const effectiveEnabledSpecializationIds = (
    hasConfiguredEnabledSpecializationIds
      ? enabledSpecializationIds
      : specializations.map((specialization) => specialization.id)
  )
  const specializationsSelectable = effectiveEnabledSpecializationIds.length > 0
  const requestSpecializationId = (() => {
    if (lockedSpecializationId != null) return lockedSpecializationId
    if (!specializationsSelectable) return null
    if (effectiveChatMode !== 'researcher') return effectiveSpecializationId
    if (!hasSpecializationDocContext) return null
    if (specializationSelectionExplicitRef.current) return effectiveSpecializationId
    return null
  })()
  const hiddenUploadCount = Math.max(0, chatUploads.length - visibleUploadCount)
  const visibleUploads = hiddenUploadCount > 0 ? chatUploads.slice(0, visibleUploadCount) : chatUploads
  const recomputeVisibleUploadCount = useCallback(() => {
    if (chatFileScope || !hasUploadChipRow) {
      setVisibleUploadCount(chatUploads.length)
      return
    }
    const container = uploadChipsContainerRef.current
    if (!container) {
      setVisibleUploadCount(chatUploads.length)
      return
    }
    const availableWidth = container.clientWidth
    if (!Number.isFinite(availableWidth) || availableWidth <= 0) {
      setVisibleUploadCount(chatUploads.length)
      return
    }
    const styles = window.getComputedStyle(container)
    const gapPx = Number.parseFloat(styles.columnGap || styles.gap || '0') || 0
    const widths = chatUploads.map((upload) => {
      const uploadId = String(upload.upload_id)
      const measured = uploadChipMeasureRefs.current[uploadId]?.offsetWidth
      return Number.isFinite(measured) && measured && measured > 0 ? measured : UPLOAD_CHIP_FALLBACK_WIDTH
    })
    const overflowWidth = (() => {
      const measured = uploadOverflowMeasureRef.current?.offsetWidth
      return Number.isFinite(measured) && measured && measured > 0 ? measured : UPLOAD_OVERFLOW_CHIP_FALLBACK_WIDTH
    })()
    const pendingWidth = hasPendingUploads
      ? (() => {
        const measured = uploadPendingMeasureRef.current?.offsetWidth
        return Number.isFinite(measured) && measured && measured > 0 ? measured : UPLOAD_PENDING_CHIP_FALLBACK_WIDTH
      })()
      : 0
    const prefixSums: number[] = [0]
    for (const width of widths) {
      prefixSums.push(prefixSums[prefixSums.length - 1] + width)
    }
    let bestVisible = chatUploads.length
    for (let visible = chatUploads.length; visible >= 0; visible -= 1) {
      const hidden = chatUploads.length - visible
      const trailingWidth = (hidden > 0 ? overflowWidth : 0) + (hasPendingUploads ? pendingWidth : 0)
      const elementCount = visible + (hidden > 0 ? 1 : 0) + (hasPendingUploads ? 1 : 0)
      const gapsTotal = elementCount > 0 ? gapPx * Math.max(0, elementCount - 1) : 0
      const requiredWidth = prefixSums[visible] + trailingWidth + gapsTotal
      if (requiredWidth <= availableWidth) {
        bestVisible = visible
        break
      }
    }
    setVisibleUploadCount((prev) => (prev === bestVisible ? prev : bestVisible))
  }, [chatFileScope, hasUploadChipRow, chatUploads, hasPendingUploads])

  useLayoutEffect(() => {
    recomputeVisibleUploadCount()
  }, [recomputeVisibleUploadCount])

  useEffect(() => {
    const container = uploadChipsContainerRef.current
    if (!container || chatFileScope || !hasUploadChipRow) return
    const observer = new ResizeObserver(() => recomputeVisibleUploadCount())
    observer.observe(container)
    return () => observer.disconnect()
  }, [chatFileScope, hasUploadChipRow, recomputeVisibleUploadCount])

  useEffect(() => {
    if (!chatFileScope && !hasActiveUploadAttachments && !hasPendingUploads) return
    if (effectiveChatMode === 'researcher') return
    setChatMode('researcher')
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, 'researcher')
    } catch {
      // ignore storage errors
    }
  }, [chatFileScope, effectiveChatMode, hasActiveUploadAttachments, hasPendingUploads])

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const el = messagesContainerRef.current
    if (!el) return
    if (behavior === 'smooth') {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
      return
    }
    el.scrollTop = el.scrollHeight
  }, [])

  const scheduleAutoFollow = useCallback(() => {
    if (autoFollowRafRef.current != null) return
    autoFollowRafRef.current = window.requestAnimationFrame(() => {
      autoFollowRafRef.current = null
      const el = messagesContainerRef.current
      if (!el) return
      el.scrollTop = el.scrollHeight
    })
  }, [])

  const handleScroll = useCallback(() => {
    if (scrollRafRef.current != null) return
    scrollRafRef.current = window.requestAnimationFrame(() => {
      scrollRafRef.current = null
      const el = messagesContainerRef.current
      if (!el) return
      const { scrollTop, scrollHeight, clientHeight } = el
      const nearBottom = scrollHeight - scrollTop - clientHeight < 80
      isNearBottomRef.current = nearBottom
      const nextShowScroll = !nearBottom
      if (showScrollRef.current !== nextShowScroll) {
        showScrollRef.current = nextShowScroll
        setShowScrollToBottom(nextShowScroll)
      }
    })
  }, [])

  useEffect(() => {
    const el = messagesContainerRef.current
    if (!el) return
    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      el.removeEventListener('scroll', handleScroll)
      if (scrollRafRef.current != null) {
        window.cancelAnimationFrame(scrollRafRef.current)
        scrollRafRef.current = null
      }
      if (autoFollowRafRef.current != null) {
        window.cancelAnimationFrame(autoFollowRafRef.current)
        autoFollowRafRef.current = null
      }
    }
  }, [handleScroll])

  useEffect(() => {
    if (messages.length === 0) return
    scheduleAutoFollow()
  }, [messages.length, scheduleAutoFollow])

  useEffect(() => {
    if (messages.length > 0) return
    showScrollRef.current = false
    setShowScrollToBottom(false)
    isNearBottomRef.current = true
  }, [messages.length])

  useEffect(() => {
    if (!contextChatId || messages.length === 0 || loadingChat) return
    const restoredMode = resolveChatModeFromHistory(messages)
    if (!restoredMode || restoredMode === chatMode) return
    setChatMode(restoredMode)
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, restoredMode)
    } catch {
      // ignore storage errors
    }
  }, [contextChatId, messages, loadingChat, chatMode])

  useEffect(() => {
    if (!contextChatId || loadingChat || messages.length === 0) return
    const restoredRoleId = (
      messages.find((msg) => msg.role === 'user' && !msg.isInternal && !!msg.specializationId)?.specializationId
      ?? messages.find((msg) => msg.role === 'assistant' && !!msg.specializationId)?.specializationId
      ?? null
    )
    if (restoredRoleId == null) return
    if (restoredRoleId === selectedSpecializationId) return
    setSelectedSpecializationId(restoredRoleId)
  }, [contextChatId, messages, loadingChat, selectedSpecializationId])

  const lastMessage = messages[messages.length - 1]
  const isInitialThinkingPhase = isStreaming && lastMessage?.role === 'assistant' && !lastMessage.content
  const showOfflineEmptyState = offline && !loadingChat && messages.length === 0
  const isCenteredComposer = !offline && !loadingChat && messages.length === 0

  useEffect(() => {
    if (wasCenteredComposerRef.current && !isCenteredComposer) {
      setAnimateToDocked(true)
      if (dockAnimationTimerRef.current != null) {
        window.clearTimeout(dockAnimationTimerRef.current)
      }
      dockAnimationTimerRef.current = window.setTimeout(() => {
        setAnimateToDocked(false)
        dockAnimationTimerRef.current = null
      }, 1200)
    } else if (isCenteredComposer) {
      setAnimateToDocked(false)
      if (dockAnimationTimerRef.current != null) {
        window.clearTimeout(dockAnimationTimerRef.current)
        dockAnimationTimerRef.current = null
      }
    }
    wasCenteredComposerRef.current = isCenteredComposer
  }, [isCenteredComposer])

  useEffect(() => () => {
    if (dockAnimationTimerRef.current != null) {
      window.clearTimeout(dockAnimationTimerRef.current)
      dockAnimationTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!isNearBottomRef.current) return
    if (messages.length === 0) return
    scheduleAutoFollow()
  }, [messages, scheduleAutoFollow])

  const handleContinue = useCallback((anchorMessageId?: number) => {
    if (offline) return
    void continueLastScope(anchorMessageId, {
      mode: effectiveChatMode,
      specializationId: requestSpecializationId,
      fileScope: chatFileScope,
      chatWebSearchEnabled,
      chatWebSearchPrivacyOverride,
    })
  }, [offline, continueLastScope, effectiveChatMode, requestSpecializationId, chatFileScope, chatWebSearchPrivacyOverride, chatWebSearchEnabled])

  const handleRegenerate = useCallback((assistantMessageIndex: number) => {
    if (offline) return
    if (isStreaming) return
    const previousUser = [...messages]
      .slice(0, assistantMessageIndex)
      .reverse()
      .find((msg) => msg.role === 'user' && !msg.isInternal && !!msg.content?.trim())
    if (!previousUser) return
    void sendMessage(previousUser.content, {
      mode: effectiveChatMode,
      specializationId: requestSpecializationId,
      fileScope: chatFileScope,
      chatWebSearchEnabled,
      chatWebSearchPrivacyOverride,
    })
  }, [offline, isStreaming, messages, sendMessage, effectiveChatMode, requestSpecializationId, chatFileScope, chatWebSearchPrivacyOverride, chatWebSearchEnabled])

  const handleAskInAssistant = useCallback(async (assistantMessageIndex: number) => {
    if (offline) return
    if (isStreaming) return
    if (hasScopedInputPill) return
    const previousUser = [...messages]
      .slice(0, assistantMessageIndex)
      .reverse()
      .find((msg) => msg.role === 'user' && !msg.isInternal && !!msg.content?.trim())
    if (!previousUser) return
    await newChat()
    setChatMode('assistant')
    setSelectedSpecializationId(null)
    specializationSelectionExplicitRef.current = false
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, 'assistant')
      window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
    } catch {
      // ignore storage errors
    }
    await sendMessage(previousUser.content, {
      mode: 'assistant',
      specializationId: null,
      fileScope: null,
      chatWebSearchEnabled: false,
      chatWebSearchPrivacyOverride: false,
    })
  }, [
    offline,
    isStreaming,
    hasScopedInputPill,
    messages,
    newChat,
    sendMessage,
  ])

  const lastEditableUserMessageIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const candidate = messages[i]
      if (candidate.role === 'user' && !candidate.isInternal && String(candidate.content || '').trim()) {
        return i
      }
    }
    return -1
  })()

  const handleEditSubmit = useCallback(async (editedText: string) => {
    if (offline) return
    if (isStreaming) return
    if (loadingChat) return
    await sendMessage(editedText, {
      mode: effectiveChatMode,
      specializationId: requestSpecializationId,
      fileScope: chatFileScope,
      chatWebSearchEnabled,
      chatWebSearchPrivacyOverride,
    })
  }, [
    offline,
    isStreaming,
    loadingChat,
    sendMessage,
    effectiveChatMode,
    requestSpecializationId,
    chatFileScope,
    chatWebSearchEnabled,
    chatWebSearchPrivacyOverride,
  ])

  const handleNewChat = useCallback(() => {
    if (offline) return
    newChatRequestedRef.current = true
    setInputValue('')
    setChatMode('researcher')
    setSelectedSpecializationId(null)
    specializationSelectionExplicitRef.current = false
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, 'researcher')
      window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
    } catch {
      // ignore storage errors
    }
    void setChatWebSearchPreferences({ enabled: false, privacyOverride: false, persist: false })
    clearError()
    newChat().catch((err) => logApiError(err, 'ChatView.handleNewChat'))
  }, [offline, clearError, newChat, setChatWebSearchPreferences])

  const handleExportFullChat = useCallback(async (format: 'markdown' | 'text' = 'markdown') => {
    if (offline || isStreaming) return
    if (!contextChatId) {
      showToast('error', 'No active chat to export')
      return
    }
    setExportMenuOpen(false)
    try {
      const payload = await exportChatMarkdown(contextChatId, {
        scope: 'full_chat',
        includeFrontmatter: false,
        template: 'full_transcript',
      })
      if (format === 'text') {
        const txtFilename = String(payload.filename || 'chat-export').replace(/\.md$/, '') + '.txt'
        downloadTextFile(txtFilename, markdownToPlainText(payload.markdown))
        showToast('success', 'Chat exported as plain text')
      } else {
        triggerMarkdownDownload(payload.filename, payload.markdown)
        showToast('success', 'Chat exported as Markdown')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to export chat.'
      showToast('error', msg)
    }
  }, [contextChatId, isStreaming, offline])

  const handleExportAnswer = useCallback(async (messageId?: number) => {
    if (offline || isStreaming) return
    if (!contextChatId) {
      showToast('error', 'No active chat to export')
      return
    }
    try {
      const payload = await exportChatMarkdown(contextChatId, {
        scope: 'current_answer',
        messageId,
        includeFrontmatter: false,
        template: 'concise_summary',
      })
      triggerMarkdownDownload(payload.filename, payload.markdown)
      showToast('success', 'Answer exported as Markdown')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to export answer.'
      showToast('error', msg)
    }
  }, [contextChatId, isStreaming, offline])

  const handleExportAnswerText = useCallback(async (messageId?: number) => {
    if (offline || isStreaming) return
    if (!contextChatId) return
    try {
      const payload = await exportChatMarkdown(contextChatId, {
        scope: 'current_answer',
        messageId,
        includeFrontmatter: false,
        template: 'concise_summary',
      })
      const txtFilename = String(payload.filename || 'answer').replace(/\.md$/, '') + '.txt'
      downloadTextFile(txtFilename, markdownToPlainText(payload.markdown))
      showToast('success', 'Answer exported as plain text')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to export answer.'
      showToast('error', msg)
    }
  }, [contextChatId, isStreaming, offline])

  const handleTranslateReply = useCallback(async (messageId?: number) => {
    if (offline || isStreaming || loadingChat || isTranslating || isTranslatingReply) return
    if (!contextChatId) return
    if (!messageId || !Number.isFinite(messageId)) return
    await translateAssistantMessage(messageId, {
      targetLanguage: translateTargetLanguage,
      tone: translateTone,
    })
  }, [contextChatId, isStreaming, isTranslating, isTranslatingReply, loadingChat, offline, translateAssistantMessage, translateTargetLanguage, translateTone])

  useEffect(() => {
    const handleNewChatEvent = () => handleNewChat()
    window.addEventListener('new-chat', handleNewChatEvent)
    return () => window.removeEventListener('new-chat', handleNewChatEvent)
  }, [handleNewChat])

  const handleSend = useCallback(async () => {
    if (offline) return
    if (isTranslating) return
    if (isTranslatingReply) return
    const text = inputValue.trim()
    if (!text) return

    setInputValue('')
    await sendMessage(text, {
      mode: effectiveChatMode,
      specializationId: requestSpecializationId,
      fileScope: chatFileScope,
      chatWebSearchEnabled,
      chatWebSearchPrivacyOverride,
    })
  }, [offline, inputValue, isTranslating, isTranslatingReply, sendMessage, effectiveChatMode, requestSpecializationId, chatFileScope, chatWebSearchPrivacyOverride, chatWebSearchEnabled])

  const handleStop = useCallback(() => {
    if (offline) return
    void stopStreaming()
  }, [offline, stopStreaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (offline) return
    if (e.key === 'Tab' && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey) {
      // Keep focus in composer for plain Tab to avoid accidental focus jumps in
      // desktop WebView hosts (Tauri/WKWebView) while typing.
      e.preventDefault()
      const textarea = e.currentTarget as HTMLTextAreaElement
      const { selectionStart, selectionEnd, value } = textarea
      const hasTrailingInlineSelection = (
        selectionEnd > selectionStart
        && selectionEnd === value.length
      )
      if (hasTrailingInlineSelection) {
        // Accept inline completion selection without moving focus away from composer.
        textarea.setSelectionRange(selectionEnd, selectionEnd)
      }
      return
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const resizeTextarea = useCallback((ta: HTMLTextAreaElement) => {
    const scopedExtra = hasScopedInputPill ? COMPOSER_SCOPED_EXTRA_HEIGHT : 0
    const minHeight = COMPOSER_TEXTAREA_MIN_HEIGHT + scopedExtra
    const maxHeight = COMPOSER_TEXTAREA_MAX_HEIGHT + scopedExtra
    ta.style.height = 'auto'
    const nextHeight = Math.min(Math.max(ta.scrollHeight, minHeight), maxHeight)
    ta.style.height = `${nextHeight}px`
    const canScroll = ta.scrollHeight > maxHeight
    ta.style.overflowY = canScroll ? 'auto' : 'hidden'
    setTextareaCanScroll(canScroll)
    if (!canScroll && ta.scrollTop !== 0) {
      ta.scrollTop = 0
    }
    setTextareaHasTopScroll(ta.scrollTop > 0)
  }, [hasScopedInputPill])

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    resizeTextarea(ta)
  }, [inputValue, resizeTextarea, hasScopedInputPill])

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (offline) return
    setInputValue(e.target.value)
  }

  const handleTextareaScroll = useCallback((e: React.UIEvent<HTMLTextAreaElement>) => {
    setTextareaHasTopScroll(e.currentTarget.scrollTop > 0)
  }, [])

  const webSearchToggleLocked = offline || isStreaming
  const webSearchToggleTitle = fullPrivacyMode
    ? (chatWebSearchEnabled
      ? 'Web search override is enabled for this chat'
      : 'Enable web search override for this chat')
    : (chatWebSearchEnabled
      ? 'Web search is enabled for this chat'
      : 'Enable web search for this chat')

  const handleWebSearchToggle = useCallback(() => {
    if (webSearchToggleLocked) return
    const next = !chatWebSearchEnabled
    const nextPrivacyOverride = next ? fullPrivacyMode : false
    void setChatWebSearchPreferences({
      enabled: next,
      privacyOverride: nextPrivacyOverride,
    })
  }, [webSearchToggleLocked, chatWebSearchEnabled, fullPrivacyMode, setChatWebSearchPreferences])

  const enabledSpecializations = specializations.filter((specialization) => effectiveEnabledSpecializationIds.includes(specialization.id))
  const selectedSpecialization = specializations.find((specialization) => specialization.id === effectiveSpecializationId) ?? (
    effectiveSpecializationId
      ? {
          id: effectiveSpecializationId,
          name: formatSpecializationNameFromId(effectiveSpecializationId),
          description: '',
          icon: null,
        }
      : null
  )
  const modeSelectorDisabled = offline || isStreaming || modeSpecializationSessionLocked
  const specializationSelectorDisabled = offline || isStreaming || enabledSpecializations.length === 0 || modeSpecializationSessionLocked
  const specializationButtonLabel = selectedSpecialization?.name || GENERAL_SPECIALIZATION_LABEL
  const showSpecializationSelector = (
    (specializationsSelectable || lockedSpecializationId != null || !specializationsLoaded)
    && (enabledSpecializations.length > 0 || lockedSpecializationId != null || !specializationsLoaded)
    && (
      effectiveChatMode === 'assistant'
      || (effectiveChatMode === 'researcher' && hasSpecializationDocContext)
    )
  )
  const specializationContextTooltip = (
    effectiveChatMode === 'researcher' && !hasSpecializationDocContext
      ? 'Plugins are strongest with scoped documents'
      : `Specialization: ${specializationButtonLabel}`
  )

  useEffect(() => {
    if (showSpecializationSelector) return
    setSpecializationMenuOpen(false)
  }, [showSpecializationSelector])

  const lockedSpecializationMissingFromEnabled = (
    !!lockedSpecializationId
    && !effectiveEnabledSpecializationIds.includes(lockedSpecializationId)
    && specializations.some((specialization) => specialization.id === lockedSpecializationId)
  )

  useEffect(() => {
    if (lockedSpecializationId == null) return
    if (lockedSpecializationId === selectedSpecializationId) return
    setSelectedSpecializationId(lockedSpecializationId)
  }, [lockedSpecializationId, selectedSpecializationId])

  useEffect(() => {
    if (lockedMode == null) return
    if (lockedMode === chatMode) return
    setChatMode(lockedMode)
    try {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, lockedMode)
    } catch {
      // ignore storage errors
    }
  }, [lockedMode, chatMode])

  const handleUploadControl = useCallback(() => {
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    uploadInputRef.current?.click()
  }, [offline, isStreaming, effectiveChatMode])

  const uploadSelectedFiles = useCallback(async (selectedFiles: File[]) => {
    if (selectedFiles.length === 0) return
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    const selectedCount = selectedFiles.length
    const uploadScopeKeyAtStart = contextChatId || '__draft__'
    const uploadScopeKeyToSettle = uploadScopeKeyAtStart
    if (uploadScopeKeyAtStart === '__draft__') {
      draftPendingAliasChatIdRef.current = null
    }
    setPendingUploadCountsByChat((prev) => ({
      ...prev,
      [uploadScopeKeyAtStart]: (prev[uploadScopeKeyAtStart] ?? 0) + selectedCount,
    }))
    try {
      await uploadFiles(selectedFiles, {
        onChatResolved: (resolvedChatId: string) => {
          if (uploadScopeKeyAtStart !== '__draft__') return
          const normalizedChatId = String(resolvedChatId || '').trim()
          if (!normalizedChatId) return
          skipNextSelectChatIdRef.current = normalizedChatId
          draftPendingAliasChatIdRef.current = normalizedChatId
        },
      })
    } catch (err) {
      logApiError(err, 'ChatView.handleUploadInputChange')
    } finally {
      await new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => resolve())
      })
      setPendingUploadCountsByChat((prev) => {
        const current = prev[uploadScopeKeyToSettle] ?? 0
        const remaining = Math.max(0, current - selectedCount)
        if (remaining === current) return prev
        const next = { ...prev }
        if (remaining === 0) {
          delete next[uploadScopeKeyToSettle]
        } else {
          next[uploadScopeKeyToSettle] = remaining
        }
        return next
      })
      if (uploadScopeKeyAtStart === '__draft__') {
        draftPendingAliasChatIdRef.current = null
      }
    }
  }, [effectiveChatMode, contextChatId, isStreaming, offline, uploadFiles])

  const isFileDragEvent = useCallback((event: { dataTransfer: DataTransfer | null }): boolean => {
    const transfer = event.dataTransfer
    if (!transfer) return false
    if ((transfer.files?.length ?? 0) > 0) return true
    if (transfer.items && Array.from(transfer.items).some((item) => item.kind === 'file')) return true
    const types = Array.from(transfer.types ?? []).map((value) => value.toLowerCase())
    return types.some((value) => value === 'files' || value.includes('file') || value.includes('public.file-url'))
  }, [])

  const handleUploadInputChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files
    if (!fileList || fileList.length === 0) return
    const selectedFiles = Array.from(fileList)
    event.target.value = ''
    await uploadSelectedFiles(selectedFiles)
  }, [uploadSelectedFiles])

  const handleComposerDragEnter = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (!isFileDragEvent(event)) return
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    event.preventDefault()
    event.stopPropagation()
    uploadDragDepthRef.current += 1
    setIsDragOverComposer(true)
  }, [effectiveChatMode, isFileDragEvent, isStreaming, offline])

  const handleComposerDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (!isFileDragEvent(event)) return
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
    if (!isDragOverComposer) {
      setIsDragOverComposer(true)
    }
  }, [effectiveChatMode, isDragOverComposer, isFileDragEvent, isStreaming, offline])

  const handleComposerDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (!isFileDragEvent(event)) return
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    event.preventDefault()
    event.stopPropagation()
    uploadDragDepthRef.current = Math.max(0, uploadDragDepthRef.current - 1)
    if (uploadDragDepthRef.current === 0) {
      setIsDragOverComposer(false)
    }
  }, [effectiveChatMode, isFileDragEvent, isStreaming, offline])

  const handleComposerDrop = useCallback(async (event: React.DragEvent<HTMLDivElement>) => {
    const selectedFiles = Array.from(event.dataTransfer.files || [])
    if (selectedFiles.length === 0 && !isFileDragEvent(event)) return
    event.preventDefault()
    event.stopPropagation()
    uploadDragDepthRef.current = 0
    setIsDragOverComposer(false)
    if (offline || isStreaming || effectiveChatMode !== 'researcher') return
    await uploadSelectedFiles(selectedFiles)
  }, [effectiveChatMode, isFileDragEvent, isStreaming, offline, uploadSelectedFiles])

  const handleRemoveUpload = useCallback(async (uploadId: string) => {
    try {
      await removeUploadedFile(uploadId)
    } catch (err) {
      logApiError(err, 'ChatView.handleRemoveUpload')
    }
  }, [removeUploadedFile])

  return (
    <div className="chat-view">
      <PageHeader
        title="Chat"
        subtitle="Ask questions about your documents"
        icon="ri-chat-ai-4-line"
        className="chat-view__header"
        action={
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {hasAssistantReply && (
              <div ref={exportMenuRef} style={{ position: 'relative' }}>
                <button
                  type="button"
                  className="chat-view__new-chat"
                  onClick={() => setExportMenuOpen(v => !v)}
                  disabled={offline || isStreaming || !contextChatId}
                  title="Export chat"
                  aria-label="Export chat"
                >
                  <i className="ri-download-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                  <span>Export</span>
                </button>
                {exportMenuOpen && (
                  <div className="chat-view__mode-menu" role="menu" style={{ left: 0, right: 'auto', minWidth: '9rem', bottom: 'auto', top: 'calc(100% + 0.375rem)' }}>
                    <button type="button" className="chat-view__mode-option" onClick={() => handleExportFullChat('markdown')}>
                      <i className="ri-markdown-line" aria-hidden style={{ fontSize: '1rem' }} />
                      <span>Markdown</span>
                    </button>
                    <button type="button" className="chat-view__mode-option" onClick={() => handleExportFullChat('text')}>
                      <i className="ri-file-text-line" aria-hidden style={{ fontSize: '1rem' }} />
                      <span>Plain text</span>
                    </button>
                  </div>
                )}
              </div>
            )}
            <button
              type="button"
              className="chat-view__new-chat"
              onClick={handleNewChat}
              disabled={offline || isStreaming}
              title={
                isStreaming
                  ? 'Stop current response to start a new chat'
                  : 'New Chat (Cmd+N)'
              }
              aria-label="Start New Chat"
            >
              <i className="ri-chat-new-line" aria-hidden style={{ fontSize: '1.125rem' }} />
              <span>New Chat</span>
            </button>
          </div>
        }
      />

      <div className="chat-view__body">
        <div className="chat-view__content">
          <div className="chat-view__messages-wrapper">
            <div className={`chat-view__messages${isCenteredComposer ? ' chat-view__messages--centered-composer' : ''}`}>
              <div
                ref={messagesContainerRef}
                className={`chat-view__messages-scroll${showOfflineEmptyState ? ' chat-view__messages-scroll--state' : ''}`}
              >
                {loadingChat && (
                  <>
                    <ChatMessageSkeleton role="user" />
                    <ChatMessageSkeleton role="assistant" />
                  </>
                )}
                {showOfflineEmptyState && <ServiceUnavailableState />}
                {!loadingChat &&
                  renderedMessages.length > 0 &&
                  renderedMessages.map((msg, i) => (
                    <ChatMessage
                      key={`${msg.role}-${i}-${msg.id ?? msg.createdAt ?? ''}`}
                      id={msg.id}
                      role={msg.role}
                      content={msg.content}
                      isInternal={msg.isInternal}
                      isContinuation={msg.isContinuation}
                      chatMode={msg.chatMode}
                      sources={msg.sources}
                      displayBlocks={msg.displayBlocks}
                      fileDiscovery={msg.fileDiscovery}
                      isStreaming={msg.isStreaming}
                      streamStatusText={msg.streamStatusText}
                      streamSectionProgress={msg.streamSectionProgress}
                      streamPlanSteps={msg.streamPlanSteps}
                      scopedFileName={msg.scopedFileName}
                      translateTargetLanguage={translateTargetLanguage}
                      isPartial={msg.isPartial}
                      hasRemainingScope={msg.hasRemainingScope}
                      completionMode={msg.completionMode}
                      stoppedByUser={msg.stoppedByUser}
                      nextAction={msg.nextAction}
                      continueLabel={msg.continueLabel}
                      webSearchUsed={msg.webSearchUsed}
                      specializationName={msg.specializationId ? (specializations.find((specialization) => specialization.id === msg.specializationId)?.name || msg.specializationId) : undefined}
                      specializationIcon={msg.specializationId ? (specializations.find((specialization) => specialization.id === msg.specializationId)?.icon || undefined) : undefined}
                      createdAt={msg.createdAt}
                      generationSeconds={msg.generationSeconds}
                      enableRawOutputControl={enableRawOutputControl}
                      onContinue={handleContinue}
                      onRegenerate={() => handleRegenerate(i)}
                      onAssistantSwitch={hideAssistantSwitch ? undefined : (() => void handleAskInAssistant(i))}
                      onExport={handleExportAnswer}
                      onExportText={handleExportAnswerText}
                      canEdit={
                        i === lastEditableUserMessageIndex
                        && !offline
                        && !isStreaming
                        && !loadingChat
                      }
                      onEditSubmit={handleEditSubmit}
                      canContinue={!offline && !isStreaming}
                      canRegenerate={!offline && !isStreaming}
                      canAssistantSwitch={!offline && !isStreaming && !hideAssistantSwitch}
                      onTranslate={typeof msg.translatedFromMessageId === 'number' ? undefined : handleTranslateReply}
                      translationLanguage={msg.translationLanguage}
                      translationTone={msg.translationTone}
                      translationIsStale={msg.translationIsStale}
                      canTranslate={!offline && !isStreaming && !loadingChat && !isTranslating && !isTranslatingReply && typeof msg.translatedFromMessageId !== 'number'}
                      actionsDisabled={offline}
                    />
                  ))}
                {!showOfflineEmptyState && <div className="chat-view__messages-end" />}
              </div>

              <div
                className={
                  `chat-view__input-area composer-wrap${isCenteredComposer ? ' chat-view__input-area--centered' : ''}${animateToDocked ? ' chat-view__input-area--docking composer-wrap--docking' : ''}`
                }
              >
                {error && <div className="chat-view__error">{error}</div>}
                <div
                  className={
                    `chat-view__input-wrapper composer__input-wrapper${textareaCanScroll ? ' chat-view__input-wrapper--scrollable composer__input-wrapper--scrollable' : ''}${textareaHasTopScroll ? ' chat-view__input-wrapper--top-scrolled composer__input-wrapper--top-scrolled' : ''}${hasScopedInputPill ? ' chat-view__input-wrapper--scoped composer__input-wrapper--scoped' : ''}${isDragOverComposer ? ' chat-view__input-wrapper--drag-active composer__input-wrapper--drag-active' : ''}`
                  }
                  onDragEnter={handleComposerDragEnter}
                  onDragOver={handleComposerDragOver}
                  onDragLeave={handleComposerDragLeave}
                  onDrop={handleComposerDrop}
                >
                  <input
                    ref={uploadInputRef}
                    type="file"
                    className="chat-view__upload-input"
                    multiple
                    onChange={handleUploadInputChange}
                    tabIndex={-1}
                    aria-hidden
                  />
                  {chatFileScope && (
                    <span className="chat-view__scope-chip" title={chatFileScope.filename}>
                      <i className={resolveFileIconFromFilename(chatFileScope.filename)} aria-hidden />
                      <span>{chatFileScope.filename}</span>
                      <button
                        type="button"
                        className="chat-view__scope-clear"
                        onClick={clearChatFileScope}
                        disabled={offline || isStreaming}
                        aria-label="Clear file scope"
                        title="Clear file scope"
                      >
                        <i className="ri-close-line" aria-hidden />
                      </button>
                    </span>
                  )}
                  {!chatFileScope && hasUploadChipRow && (
                    <div
                      ref={uploadChipsContainerRef}
                      className="chat-view__upload-chips"
                      role="list"
                      aria-label="Uploaded files"
                    >
                      {visibleUploads.map((upload) => {
                        const uploadId = String(upload.upload_id)
                        const isReady = upload.state === 'ready'
                        return (
                          <span
                            key={uploadId}
                            className={`chat-view__upload-chip${isReady ? ' chat-view__upload-chip--ready' : ''}`}
                            title={upload.filename_at_upload}
                            role="listitem"
                          >
                            <span className="chat-view__upload-chip-label">
                              <i className={resolveFileIconFromFilename(upload.filename_at_upload)} aria-hidden />
                              <span>{upload.filename_at_upload}</span>
                              {!isReady && <em>{upload.state === 'failed' ? 'Failed' : 'Indexing'}</em>}
                            </span>
                            <button
                              type="button"
                              className="chat-view__upload-chip-remove"
                              onClick={() => void handleRemoveUpload(uploadId)}
                              disabled={offline || isStreaming}
                              aria-label={`Remove ${upload.filename_at_upload}`}
                              title={`Remove ${upload.filename_at_upload}`}
                            >
                              <i className="ri-close-line" aria-hidden />
                            </button>
                          </span>
                        )
                      })}
                      {hiddenUploadCount > 0 && (
                        <span
                          className="chat-view__upload-chip chat-view__upload-overflow-chip"
                          title={`${hiddenUploadCount} more file${hiddenUploadCount === 1 ? '' : 's'} not shown`}
                          role="listitem"
                          aria-label={`${hiddenUploadCount} more uploaded files`}
                        >
                          <span className="chat-view__upload-chip-label">+{hiddenUploadCount}</span>
                        </span>
                      )}
                      {hasPendingUploads && (
                        <span
                          className="chat-view__upload-chip chat-view__upload-overflow-chip chat-view__upload-overflow-chip--pending"
                          title={pendingUploadCount === 1 ? 'Uploading 1 file…' : `Uploading ${pendingUploadCount} files…`}
                          role="listitem"
                          aria-label={pendingUploadCount === 1 ? 'Uploading 1 file' : `Uploading ${pendingUploadCount} files`}
                        >
                          <span className="chat-view__upload-chip-label">
                            <i className="ri-loader-4-line chat-view__upload-chip-spinner" aria-hidden />
                          </span>
                        </span>
                      )}
                    </div>
                  )}
                  {!chatFileScope && hasUploadChipRow && (
                    <div className="chat-view__upload-measure" aria-hidden>
                      {chatUploads.map((upload) => {
                        const uploadId = String(upload.upload_id)
                        const isReady = upload.state === 'ready'
                        return (
                          <span
                            key={`measure-${uploadId}`}
                            ref={(el) => { uploadChipMeasureRefs.current[uploadId] = el }}
                            className={`chat-view__upload-chip${isReady ? ' chat-view__upload-chip--ready' : ''}`}
                          >
                            <span className="chat-view__upload-chip-label">
                              <i className={resolveFileIconFromFilename(upload.filename_at_upload)} aria-hidden />
                              <span>{upload.filename_at_upload}</span>
                              {!isReady && <em>{upload.state === 'failed' ? 'Failed' : 'Indexing'}</em>}
                            </span>
                            <span className="chat-view__upload-chip-remove" aria-hidden>
                              <i className="ri-close-line" aria-hidden />
                            </span>
                          </span>
                        )
                      })}
                      <span ref={uploadOverflowMeasureRef} className="chat-view__upload-chip chat-view__upload-overflow-chip">
                        <span className="chat-view__upload-chip-label">+999</span>
                      </span>
                      <span ref={uploadPendingMeasureRef} className="chat-view__upload-chip chat-view__upload-overflow-chip chat-view__upload-overflow-chip--pending">
                        <span className="chat-view__upload-chip-label">
                          <i className="ri-loader-4-line chat-view__upload-chip-spinner" aria-hidden />
                        </span>
                      </span>
                    </div>
                  )}
                  <textarea
                    ref={textareaRef}
                    className={`chat-view__textarea composer__textarea${hasScopedInputPill ? ' chat-view__textarea--scoped composer__textarea--scoped' : ''}`}
                    placeholder={
                      offline
                        ? 'Service unavailable'
                        : isTranslating
                          ? 'Translation in progress…'
                          : isStreaming
                            ? 'Response in progress…'
                            : (effectiveChatMode === 'assistant' && chatWebSearchEnabled
                              ? 'Search the web…'
                              : 'Ask me anything…')
                    }
                    value={inputValue}
                    onChange={handleTextareaChange}
                    onScroll={handleTextareaScroll}
                    onKeyDown={handleKeyDown}
                    aria-label="Chat message input"
                    rows={1}
                    disabled={offline || isStreaming}
                  />
                  <div className="chat-view__controls-row composer__controls-row">
                    <div className="chat-view__controls-left">
                      {effectiveChatMode === 'researcher' && !chatFileScope && (
                        <button
                          type="button"
                          className="chat-view__upload-toggle"
                          onClick={handleUploadControl}
                          disabled={offline || isStreaming}
                          aria-label="Upload files"
                        >
                          <i className="ri-add-line" aria-hidden />
                        </button>
                      )}
                      {effectiveChatMode === 'assistant' && webSearchConfigured && (
                        <button
                          type="button"
                          className={`chat-view__web-search-toggle${chatWebSearchEnabled ? ' chat-view__web-search-toggle--active' : ''}`}
                          onClick={handleWebSearchToggle}
                          disabled={webSearchToggleLocked}
                          aria-label={webSearchToggleTitle}
                          aria-pressed={chatWebSearchEnabled}
                        >
                          <i className="ri-global-line" aria-hidden />
                          {chatWebSearchEnabled && <span className="chat-view__web-search-pill">Search</span>}
                        </button>
                      )}
                      {showSpecializationSelector && (
                      <div ref={specializationMenuRef} className="chat-view__specialization-selector">
                        <button
                          type="button"
                          className="chat-view__specialization-button"
                          onClick={() => setSpecializationMenuOpen((open) => !open)}
                          disabled={specializationSelectorDisabled}
                          aria-haspopup="menu"
                          aria-expanded={specializationMenuOpen}
                          aria-label={specializationContextTooltip}
                        >
                          <i className={selectedSpecialization?.icon || 'ri-user-settings-line'} aria-hidden />
                        </button>
                        {specializationMenuOpen && (
                          <div className="chat-view__mode-menu chat-view__mode-menu--specialization" role="menu">
                            <span className="chat-view__mode-option-wrap">
                              <button
                                type="button"
                                className={`chat-view__mode-option${effectiveSpecializationId == null ? ' chat-view__mode-option--active' : ''}`}
                                role="menuitemradio"
                                aria-label={GENERAL_SPECIALIZATION_LABEL}
                                aria-checked={effectiveSpecializationId == null}
                                disabled={specializationSelectorDisabled}
                                onClick={() => {
                                  setSelectedSpecializationId(null)
                                  specializationSelectionExplicitRef.current = true
                                  setSpecializationMenuOpen(false)
                                  try {
                                    window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
                                  } catch {
                                    // ignore storage errors
                                  }
                                }}
                              >
                                <i className="ri-user-settings-line" aria-hidden />
                                <span className="chat-view__mode-option-label">{GENERAL_SPECIALIZATION_LABEL}</span>
                              </button>
                            </span>
                            {enabledSpecializations.map((specialization) => (
                              <span key={specialization.id} className="chat-view__mode-option-wrap">
                                <button
                                  type="button"
                                  className={`chat-view__mode-option${effectiveSpecializationId === specialization.id ? ' chat-view__mode-option--active' : ''}`}
                                  role="menuitemradio"
                                  aria-label={specialization.name}
                                  aria-checked={effectiveSpecializationId === specialization.id}
                                  disabled={specializationSelectorDisabled}
                                  onClick={() => {
                                    setSelectedSpecializationId(specialization.id)
                                    specializationSelectionExplicitRef.current = true
                                    setSpecializationMenuOpen(false)
                                    try {
                                      window.localStorage.setItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY, specialization.id)
                                    } catch {
                                      // ignore storage errors
                                    }
                                  }}
                                >
                                  <i className={specialization.icon || 'ri-user-settings-line'} aria-hidden />
                                  <span className="chat-view__mode-option-label">{specialization.name}</span>
                                </button>
                              </span>
                            ))}
                            {lockedSpecializationMissingFromEnabled && selectedSpecialization && (
                              <span key={`${selectedSpecialization.id}-locked`} className="chat-view__mode-option-wrap">
                                <button
                                  type="button"
                                  className="chat-view__mode-option chat-view__mode-option--disabled"
                                  role="menuitemradio"
                                  aria-checked
                                  disabled
                                  title="Disabled in Settings for new chats"
                                >
                                  <i className={selectedSpecialization.icon || 'ri-user-settings-line'} aria-hidden />
                                  <span>{selectedSpecialization.name}</span>
                                </button>
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                      )}
                    </div>
                    <div className="chat-view__controls-right">
                      <div ref={modeMenuRef} className="chat-view__mode-selector">
                        <button
                          type="button"
                          className="chat-view__mode-button"
                          onClick={() => setModeMenuOpen((open) => !open)}
                          disabled={modeSelectorDisabled}
                          aria-haspopup="menu"
                          aria-expanded={modeMenuOpen}
                          aria-label="Select chat mode"
                        >
                          <i className={CHAT_MODE_ICONS[effectiveChatMode]} aria-hidden />
                          <span>{CHAT_MODE_LABELS[effectiveChatMode]}</span>
                          <i className="ri-arrow-down-s-line" aria-hidden />
                        </button>
                        {modeMenuOpen && (
                          <div className="chat-view__mode-menu" role="menu">
                            {ALL_CHAT_MODES.map((mode) => {
                              const scopedModeLocked = (hasScopedInputPill && mode !== 'researcher')
                              return (
                                <span key={mode} className="chat-view__mode-option-wrap">
                                <button
                                  type="button"
                                  className={`chat-view__mode-option${effectiveChatMode === mode ? ' chat-view__mode-option--active' : ''}`}
                                  role="menuitemradio"
                                  aria-checked={effectiveChatMode === mode}
                                  disabled={modeSelectorDisabled || scopedModeLocked}
                                  onClick={() => {
                                    if (modeSelectorDisabled) return
                                    if (scopedModeLocked) return
                                    setChatMode(mode)
                                    if (mode === 'researcher' && lockedSpecializationId == null) {
                                      // Prevent assistant-specialization carryover into researcher drafts:
                                      // researcher should start from General unless user picks a specialization in that mode context.
                                      setSelectedSpecializationId(null)
                                      specializationSelectionExplicitRef.current = false
                                      try {
                                        window.localStorage.removeItem(CHAT_SPECIALIZATION_ID_STORAGE_KEY)
                                      } catch {
                                        // ignore storage errors
                                      }
                                    }
                                    setModeMenuOpen(false)
                                    try {
                                      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, mode)
                                    } catch {
                                      // ignore storage errors
                                    }
                                  }}
                                >
                                  <i className={CHAT_MODE_ICONS[mode]} aria-hidden />
                                  <span>{CHAT_MODE_LABELS[mode]}</span>
                                </button>
                              </span>
                              )
                            })}
                          </div>
                        )}
                      </div>
                      {isStreaming ? (
                        <button
                          type="button"
                          className="chat-view__stop"
                          onClick={handleStop}
                          disabled={offline}
                          title="Stop generating"
                          aria-label="Stop generating response"
                        >
                          <i className="ri-stop-circle-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                        </button>
                      ) : isTranslating ? (
                        <button
                          type="button"
                          className="chat-view__send"
                          onClick={() => translateCtx?.cancelTranslation()}
                          title="Stop Translation"
                          aria-label="Stop Translation"
                          style={{ width: 'auto', gap: '0.375rem', padding: '0.375rem 0.75rem' }}
                        >
                          <i className="ri-stop-circle-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                          <span style={{ fontSize: 'var(--font-size-sm)' }}>Stop Translation</span>
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="chat-view__send"
                          onClick={handleSend}
                          disabled={offline || !inputValue.trim()}
                          aria-label="Send message"
                        >
                          <i className="ri-arrow-up-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <p className="chat-view__disclaimer">Informity AI can make mistakes. Please double-check cited sources.</p>
              </div>
            </div>

            {messages.length > 0 && showScrollToBottom && !isInitialThinkingPhase && !offline && (
              <button
                type="button"
                className={`chat-view__scroll-to-bottom${hasScopedInputPill ? ' chat-view__scroll-to-bottom--scoped' : ''}`}
                onClick={() => scrollToBottom()}
                title="Scroll to bottom"
                aria-label="Scroll to bottom"
              >
                <i className="ri-arrow-down-line" aria-hidden style={{ fontSize: '1.125rem' }} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
