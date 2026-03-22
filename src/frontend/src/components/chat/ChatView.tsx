/**
 * Informity AI — Chat view
 * Full-height message list, input, SSE streaming.
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatMessageSkeleton } from './ChatMessageSkeleton'
import { PageHeader } from '../PageHeader'
import { ServiceUnavailableState } from '../ServiceUnavailableState'
import { useChatContext } from '../../context/useChatContext'
import { useBackendStatus } from '../../context/useBackendStatus'
import { getCurrentChat } from '../../api'
import { logApiError } from '../../utils/logApiError'
import './ChatView.css'

const CHAT_INPUT_MIN_HEIGHT = 104
const CHAT_INPUT_MAX_HEIGHT = 304
const FORCE_NEW_CHAT_KEY = 'informity_force_new_chat'

interface ChatViewProps {
  prefillMessage?: string
  initialChatId?: string | null
}

interface GetCurrentChatResponse {
  current_chat_id?: string
}

export function ChatView({ prefillMessage = '', initialChatId = null }: ChatViewProps) {
  const { offline } = useBackendStatus()
  const {
    currentChatId: contextChatId,
    messages,
    isStreaming,
    loadingChat,
    error,
    enableRawOutputControl,
    selectChat,
    sendMessage,
    continueLastScope,
    stopStreaming,
    newChat,
    clearError,
  } = useChatContext()
  const [inputValue, setInputValue] = useState(prefillMessage)
  const [showScrollToBottom, setShowScrollToBottom] = useState(false)
  const [animateToDocked, setAnimateToDocked] = useState(false)
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
    if (isForceNewChatRequested()) return
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
      selectChat(contextChatId)
      return
    }
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

  const lastMessage = messages[messages.length - 1]
  const streamContent = lastMessage?.role === 'assistant' ? lastMessage.content : ''
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
    if (!isStreaming || !isNearBottomRef.current) return
    scheduleAutoFollow()
  }, [isStreaming, streamContent, scheduleAutoFollow])

  const handleContinue = useCallback((anchorMessageId?: number) => {
    if (offline) return
    void continueLastScope(anchorMessageId)
  }, [offline, continueLastScope])

  const handleRegenerate = useCallback((assistantMessageIndex: number) => {
    if (offline) return
    if (isStreaming) return
    const previousUser = [...messages]
      .slice(0, assistantMessageIndex)
      .reverse()
      .find((msg) => msg.role === 'user' && !msg.isInternal && !!msg.content?.trim())
    if (!previousUser) return
    void sendMessage(previousUser.content)
  }, [offline, isStreaming, messages, sendMessage])

  const handleNewChat = useCallback(() => {
    if (offline) return
    newChatRequestedRef.current = true
    setInputValue('')
    clearError()
    newChat().catch((err) => logApiError(err, 'ChatView.handleNewChat'))
  }, [offline, clearError, newChat])

  useEffect(() => {
    const handleNewChatEvent = () => handleNewChat()
    window.addEventListener('new-chat', handleNewChatEvent)
    return () => window.removeEventListener('new-chat', handleNewChatEvent)
  }, [handleNewChat])

  const handleSend = useCallback(async () => {
    if (offline) return
    const text = inputValue.trim()
    if (!text) return

    setInputValue('')
    await sendMessage(text)
  }, [offline, inputValue, sendMessage])

  const handleStop = useCallback(() => {
    if (offline) return
    void stopStreaming()
  }, [offline, stopStreaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (offline) return
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const resizeTextarea = useCallback((ta: HTMLTextAreaElement) => {
    ta.style.height = 'auto'
    const nextHeight = Math.min(Math.max(ta.scrollHeight, CHAT_INPUT_MIN_HEIGHT), CHAT_INPUT_MAX_HEIGHT)
    ta.style.height = `${nextHeight}px`
    ta.style.overflowY = ta.scrollHeight > CHAT_INPUT_MAX_HEIGHT ? 'auto' : 'hidden'
  }, [])

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    resizeTextarea(ta)
  }, [inputValue, resizeTextarea])

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (offline) return
    setInputValue(e.target.value)
  }

  const handleMessagesWrapperWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    const container = messagesContainerRef.current
    if (!container) return

    const target = e.target as HTMLElement | null
    if (!target) return

    // Keep native wheel behavior over input controls.
    if (target.closest('.chat-view__input-area')) return

    const deltaY = e.deltaY
    if (!Number.isFinite(deltaY) || deltaY === 0) return

    const findScrollableAncestor = (node: HTMLElement): HTMLElement | null => {
      let current: HTMLElement | null = node
      while (current && current !== container) {
        const style = window.getComputedStyle(current)
        const overflowY = style.overflowY
        const scrollableY = (overflowY === 'auto' || overflowY === 'scroll') && current.scrollHeight > current.clientHeight
        if (scrollableY) return current
        current = current.parentElement
      }
      return null
    }

    const nestedScrollable = findScrollableAncestor(target)
    if (nestedScrollable) {
      const maxScrollTop = nestedScrollable.scrollHeight - nestedScrollable.clientHeight
      const atTop = nestedScrollable.scrollTop <= 0
      const atBottom = nestedScrollable.scrollTop >= maxScrollTop - 1
      const scrollingDown = deltaY > 0
      const canNestedConsume = scrollingDown ? !atBottom : !atTop
      if (canNestedConsume) return
    }

    e.preventDefault()
    const maxContainerScroll = container.scrollHeight - container.clientHeight
    const next = Math.min(maxContainerScroll, Math.max(0, container.scrollTop + deltaY))
    if (next !== container.scrollTop) {
      container.scrollTop = next
    }
  }, [])

  return (
    <div className="chat-view">
      <PageHeader
        title="Chat"
        subtitle="Ask questions about your documents"
        icon="ri-chat-ai-4-line"
        className="chat-view__header"
        action={
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
        }
      />

      <div className="chat-view__body" onWheel={handleMessagesWrapperWheel}>
        <div className="chat-view__content">
          <div className="chat-view__messages-wrapper" onWheel={handleMessagesWrapperWheel}>
            <div className={`chat-view__messages${isCenteredComposer ? ' chat-view__messages--centered-composer' : ''}`}>
              <div
                ref={messagesContainerRef}
                className={`chat-view__messages-scroll${showOfflineEmptyState ? ' chat-view__messages-scroll--state' : ''}`}
              >
                {loadingChat && (
                  <>
                    <ChatMessageSkeleton role="user" />
                    <ChatMessageSkeleton role="assistant" />
                    <ChatMessageSkeleton role="assistant" />
                  </>
                )}
                {showOfflineEmptyState && <ServiceUnavailableState />}
                {!loadingChat &&
                  messages.length > 0 &&
                  messages.map((msg, i) => (
                    <ChatMessage
                      key={`${msg.role}-${i}-${msg.id ?? msg.createdAt ?? ''}`}
                      id={msg.id}
                      role={msg.role}
                      content={msg.content}
                      isInternal={msg.isInternal}
                      isContinuation={msg.isContinuation}
                      sources={msg.sources}
                      displayBlocks={msg.displayBlocks}
                      isStreaming={msg.isStreaming}
                      streamStatusText={msg.streamStatusText}
                      streamSectionProgress={msg.streamSectionProgress}
                      streamPlanSteps={msg.streamPlanSteps}
                      isPartial={msg.isPartial}
                      hasRemainingScope={msg.hasRemainingScope}
                      completionMode={msg.completionMode}
                      nextAction={msg.nextAction}
                      nextActionReason={msg.nextActionReason}
                      continueLabel={msg.continueLabel}
                      createdAt={msg.createdAt}
                      generationSeconds={msg.generationSeconds}
                      enableRawOutputControl={enableRawOutputControl}
                      onContinue={handleContinue}
                      onRegenerate={() => handleRegenerate(i)}
                      canContinue={!offline && !isStreaming}
                      canRegenerate={!offline && !isStreaming}
                      actionsDisabled={offline}
                    />
                  ))}
                {!showOfflineEmptyState && <div className="chat-view__messages-end" />}
              </div>

              <div
                className={
                  `chat-view__input-area${isCenteredComposer ? ' chat-view__input-area--centered' : ''}${animateToDocked ? ' chat-view__input-area--docking' : ''}`
                }
              >
                {error && <div className="chat-view__error">{error}</div>}
                <div className="chat-view__input-wrapper">
                  <textarea
                    ref={textareaRef}
                    className="chat-view__textarea"
                    placeholder={
                      offline
                        ? 'Service unavailable'
                        : isStreaming
                          ? 'Response in progress...'
                          : 'Ask me anything...'
                    }
                    value={inputValue}
                    onChange={handleTextareaChange}
                    onKeyDown={handleKeyDown}
                    aria-label="Chat message input"
                    rows={1}
                    disabled={offline || isStreaming}
                  />
                  <div className="chat-view__controls-row">
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
                    ) : (
                      <button
                        type="button"
                        className="chat-view__send"
                        onClick={handleSend}
                        disabled={offline || !inputValue.trim()}
                        title="Send (Enter)"
                        aria-label="Send message"
                      >
                        <i className="ri-arrow-up-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                      </button>
                    )}
                  </div>
                </div>
                <p className="chat-view__disclaimer">Informity AI can make mistakes. Please double-check cited sources.</p>
              </div>
            </div>

            {messages.length > 0 && showScrollToBottom && !isInitialThinkingPhase && !offline && (
              <button
                type="button"
                className="chat-view__scroll-to-bottom"
                onClick={() => scrollToBottom()}
                title="Scroll to bottom"
                aria-label="Scroll to bottom"
              >
                <i className="ri-arrow-down-line" aria-hidden style={{ fontSize: '1rem' }} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
