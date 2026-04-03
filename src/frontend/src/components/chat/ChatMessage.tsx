import { Fragment, memo, useCallback, useState, type ReactElement } from 'react'
import { formatRelativeTime } from '../../utils/formatRelativeTime'
import { formatDuration } from '../../utils/formatDuration'
import { getMessageRaw } from '../../api'
import { SourceCard } from './SourceCard'
import { MessageBlocks } from './MessageBlocks'
import type { ChatMode, ChatSourceReference, DisplayBlock } from '../../types/api'
import 'highlight.js/styles/github-dark.min.css'
import './ChatMessage.css'

function stripThinkArtifactsForStreaming(text: string): string {
  if (!text) return ''
  let next = text
  // Remove complete think blocks.
  next = next.replace(/<think>[\s\S]*?<\/think>/gi, '')
  next = next.replace(/<<think>>[\s\S]*?<<\/think>>/gi, '')
  // If an opening think tag is present without close yet, treat remainder as hidden reasoning draft.
  const openIdx = next.search(/<think>|<<think>>/i)
  if (openIdx >= 0) {
    next = next.slice(0, openIdx)
  }
  return next
}

interface ChatMessageProps {
  id?: number
  role: string
  content: string
  isInternal?: boolean
  isContinuation?: boolean
  sources?: ChatSourceReference[]
  displayBlocks?: DisplayBlock[]
  chatMode?: ChatMode
  isStreaming?: boolean
  streamStatusText?: string
  streamSectionProgress?: {
    completed: string[]
    remaining: string[]
    total: number
  }
  streamPlanSteps?: Array<{ step_id: number; description: string; status: 'running' | 'done' | 'empty' }>
  isPartial?: boolean
  hasRemainingScope?: boolean
  completionMode?: 'complete' | 'partial' | 'scoped_complete' | 'stopped'
  stoppedByUser?: boolean
  nextAction?: 'none' | 'continue' | 'regenerate' | 'assistant_switch'
  nextActionReason?: 'stopped' | 'timeout' | 'unresolved_content' | 'budget_exhausted' | 'stalled' | 'out_of_corpus' | null
  continueLabel?: 'Continue' | 'Continue Again'
  createdAt?: string
  generationSeconds?: number
  enableRawOutputControl?: boolean
  onContinue?: (anchorMessageId?: number) => void
  onRegenerate?: () => void
  onAssistantSwitch?: () => void
  canContinue?: boolean
  canRegenerate?: boolean
  canAssistantSwitch?: boolean
  actionsDisabled?: boolean
}

function ChatMessageComponent({
  id: messageId,
  role,
  content,
  isInternal = false,
  isContinuation = false,
  sources = [],
  displayBlocks = [],
  chatMode,
  isStreaming = false,
  streamStatusText,
  streamSectionProgress,
  streamPlanSteps,
  isPartial = false,
  hasRemainingScope = false,
  completionMode = 'complete',
  stoppedByUser = false,
  nextAction = 'none',
  continueLabel = 'Continue',
  createdAt,
  generationSeconds,
  enableRawOutputControl = false,
  onContinue,
  onRegenerate,
  onAssistantSwitch,
  canContinue = false,
  canRegenerate = false,
  canAssistantSwitch = false,
  actionsDisabled = false,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [codeBlockCopied, setCodeBlockCopied] = useState(false)
  const [sourcesExpanded, setSourcesExpanded] = useState(false)
  const [rawExpanded, setRawExpanded] = useState(false)
  const [rawContent, setRawContent] = useState<string | null>(null)
  const [rawLoading, setRawLoading] = useState(false)
  const [rawCopied, setRawCopied] = useState(false)

  const handleRawToggle = useCallback(() => {
    if (actionsDisabled) return
    if (!messageId) return
    setRawExpanded((v) => !v)
    if (!rawContent && !rawLoading) {
      setRawLoading(true)
      getMessageRaw(messageId)
        .then((res) => setRawContent(res.content ?? ''))
        .catch(() => setRawContent(''))
        .finally(() => setRawLoading(false))
    }
  }, [actionsDisabled, messageId, rawContent, rawLoading])

  const isUser = role === 'user'
  const safeContent = content ?? ''
  const showActions = !(role === 'assistant' && isStreaming)
  const visibleContent = isStreaming
    ? stripThinkArtifactsForStreaming(safeContent)
    : safeContent
  const meaningfulAssistantContent = !isUser
    ? stripThinkArtifactsForStreaming(safeContent).trim()
    : ''
  const hasMeaningfulAssistantContent = meaningfulAssistantContent.length > 0
  const hasVisibleContent = visibleContent.trim().length > 0
  const shouldRenderDraft = !isUser && isStreaming && hasVisibleContent
  const showTrailingCursor = isStreaming && hasVisibleContent
  const showBouncingDots = isStreaming && !hasVisibleContent
  const isMutedOnly = !isUser && !hasVisibleContent && (safeContent || !isStreaming)
  const isScopedComplete = completionMode === 'scoped_complete' && hasRemainingScope
  const isStopped = !isUser && (completionMode === 'stopped' || stoppedByUser)
  const showContinue = !isUser
    && nextAction === 'continue'
    && !!onContinue
    && (!isStopped || hasMeaningfulAssistantContent)
  const showRegenerate = !isUser && nextAction === 'regenerate' && !!onRegenerate
  const showAssistantSwitch = !isUser && nextAction === 'assistant_switch' && !!onAssistantSwitch
  const showSectionProgress = !!streamSectionProgress && streamSectionProgress.total > 0
  const completedProgressText = showSectionProgress
    ? streamSectionProgress.completed.map((heading) => heading.replace(/^#{1,6}\s*/, '')).join(' · ')
    : ''
  const remainingProgressText = showSectionProgress
    ? streamSectionProgress.remaining.map((heading) => heading.replace(/^#{1,6}\s*/, '').trim()).join(' · ')
    : ''
  const showPlanSteps = !!streamPlanSteps && streamPlanSteps.length > 0
  const assistantMetaItems = [] as Array<{ key: string; node: ReactElement }>
  if (generationSeconds != null) {
    assistantMetaItems.push({
      key: 'timer',
      node: (
        <div className="chat-message__meta-item">
          <i className="ri-time-line chat-message__meta-icon" aria-hidden />
          <span>{formatDuration(generationSeconds)}</span>
        </div>
      ),
    })
  }
  if (!isUser && chatMode) {
    const modeIcon = chatMode === 'assistant' ? 'ri-robot-2-line' : 'ri-search-ai-3-line'
    const modeLabel = chatMode === 'assistant' ? 'Assistant' : 'Researcher'
    assistantMetaItems.push({
      key: 'mode',
      node: (
        <div className="chat-message__meta-item">
          <i className={`${modeIcon} chat-message__meta-icon`} aria-hidden />
          <span>{modeLabel}</span>
        </div>
      ),
    })
  }
  if (!isUser && isContinuation) {
    assistantMetaItems.push({
      key: 'continuation',
      node: (
        <div className="chat-message__meta-item">
          <i className="ri-flow-chart chat-message__meta-icon" aria-hidden />
          <span>Continuation</span>
        </div>
      ),
    })
  }
  if ((sources?.length ?? 0) > 0) {
    assistantMetaItems.push({
      key: 'sources',
      node: (
        <button
          type="button"
          className={`chat-message__sources-toggle-inline ${sourcesExpanded ? 'chat-message__sources-toggle-inline--expanded' : ''}`}
          onClick={() => setSourcesExpanded((v) => !v)}
          disabled={actionsDisabled}
          aria-label={`Toggle sources (${sources.length})`}
          aria-expanded={sourcesExpanded}
        >
          <i className="ri-file-copy-line chat-message__meta-icon" aria-hidden />
          <span>Sources ({sources.length})</span>
          <i className="ri-arrow-down-s-line chat-message__sources-chevron" aria-hidden />
        </button>
      ),
    })
  }
  if (isPartial) {
    assistantMetaItems.push({
      key: 'partial',
      node: (
        <div className="chat-message__meta-item">
          <i className="ri-error-warning-line chat-message__meta-icon" aria-hidden />
          <span>Partial</span>
        </div>
      ),
    })
  }
  if (isScopedComplete) {
    assistantMetaItems.push({
      key: 'condensed',
      node: (
        <div className="chat-message__meta-item">
          <i className="ri-scissors-cut-line chat-message__meta-icon" aria-hidden />
          <span>Condensed</span>
        </div>
      ),
    })
  }
  if (isStopped) {
    assistantMetaItems.push({
      key: 'stopped',
      node: (
        <div className="chat-message__meta-item">
          <i className="ri-stop-circle-line chat-message__meta-icon" aria-hidden />
          <span>Stopped</span>
        </div>
      ),
    })
  }
  if (showContinue) {
    assistantMetaItems.push({
      key: 'continue',
      node: (
        <button
          type="button"
          className="chat-message__continue-inline"
          onClick={() => onContinue?.(messageId)}
          disabled={actionsDisabled || !canContinue}
          aria-label={continueLabel}
        >
          <i className="ri-arrow-right-up-line chat-message__meta-icon" aria-hidden />
          <span>{continueLabel}</span>
        </button>
      ),
    })
  }
  if (showRegenerate) {
    assistantMetaItems.push({
      key: 'regenerate',
      node: (
        <button
          type="button"
          className="chat-message__continue-inline"
          onClick={() => onRegenerate?.()}
          disabled={actionsDisabled || !canRegenerate}
          aria-label="Regenerate from your last prompt"
        >
          <i className="ri-refresh-line chat-message__meta-icon" aria-hidden />
          <span>Regenerate</span>
        </button>
      ),
    })
  }
  if (showAssistantSwitch) {
    assistantMetaItems.push({
      key: 'assistant_switch',
      node: (
        <button
          type="button"
          className="chat-message__continue-inline"
          onClick={() => onAssistantSwitch?.()}
          disabled={actionsDisabled || !canAssistantSwitch}
          aria-label="Ask Assistant"
        >
          <i className="ri-robot-2-line chat-message__meta-icon" aria-hidden />
          <span>Ask Assistant</span>
        </button>
      ),
    })
  }
  if (enableRawOutputControl && messageId != null) {
    assistantMetaItems.push({
      key: 'raw_output',
      node: (
        <button
          type="button"
          className={`chat-message__sources-toggle-inline ${rawExpanded ? 'chat-message__sources-toggle-inline--expanded' : ''}`}
          onClick={handleRawToggle}
          disabled={actionsDisabled}
          aria-label="Toggle raw output"
          aria-expanded={rawExpanded}
        >
          <i className="ri-code-s-slash-line chat-message__meta-icon" aria-hidden />
          <span>Raw Output</span>
          <i className="ri-arrow-down-s-line chat-message__sources-chevron" aria-hidden />
        </button>
      ),
    })
  }

  const handleCopyFull = useCallback(async () => {
    if (actionsDisabled) return
    try {
      const toCopy = isUser ? safeContent : visibleContent
      await navigator.clipboard.writeText(toCopy)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // ignore
    }
  }, [actionsDisabled, isUser, safeContent, visibleContent])

  const handleCopyCode = useCallback(async (code: string) => {
    if (actionsDisabled) return
    try {
      await navigator.clipboard.writeText(code)
      setCodeBlockCopied(true)
      setTimeout(() => setCodeBlockCopied(false), 2000)
    } catch {
      // ignore
    }
  }, [actionsDisabled])

  const handleCopyRaw = useCallback(async () => {
    if (actionsDisabled || rawLoading) return
    if (rawContent == null) return
    try {
      await navigator.clipboard.writeText(rawContent)
      setRawCopied(true)
      setTimeout(() => setRawCopied(false), 2000)
    } catch {
      // ignore
    }
  }, [actionsDisabled, rawContent, rawLoading])

  if (isUser && isInternal) return null

  return (
    <div className={`chat-message chat-message--${role}${isMutedOnly ? ' chat-message--muted-only' : ''}`}>
      <div className="chat-message__inner">
        <div className="chat-message__content">
          {isUser ? (
            <p className="chat-message__text">{safeContent}</p>
          ) : (
            <>
              <div className="chat-message__markdown">
                {showBouncingDots ? (
                  <span className="chat-message__typing-indicator" aria-label="Thinking">
                    <span className="chat-message__typing-dot" />
                    <span className="chat-message__typing-dot" />
                    <span className="chat-message__typing-dot" />
                    {streamStatusText && (
                      <span className="chat-message__typing-status">
                        <span>{streamStatusText}</span>
                        {showSectionProgress && (
                          <span className="chat-message__typing-progress">
                            {completedProgressText ? `\u2713 ${completedProgressText}` : 'Starting sections...'}
                            {remainingProgressText ? ` | ${remainingProgressText}` : ''}
                          </span>
                        )}
                        {showPlanSteps && (
                          <span className="chat-message__plan-steps">
                            {streamPlanSteps!.map(step => (
                              <span key={step.step_id} className={`chat-message__plan-step chat-message__plan-step--${step.status}`}>
                                {step.status === 'done' ? '\u2713' : step.status === 'empty' ? '\u2212' : '\u25cc'}
                                {' '}{step.description}
                              </span>
                            ))}
                          </span>
                        )}
                      </span>
                    )}
                  </span>
                ) : (
                  <>
                    {shouldRenderDraft ? (
                      <p className="chat-message__text chat-message__text--draft">{visibleContent}</p>
                    ) : hasVisibleContent ? (
                      <MessageBlocks
                        blocks={displayBlocks}
                        fallbackMarkdown={visibleContent}
                        onCopyCode={handleCopyCode}
                        codeBlockCopied={codeBlockCopied}
                      />
                    ) : safeContent && isStreaming ? (
                      <span className="chat-message__typing-indicator" aria-label="Thinking">
                        <span className="chat-message__typing-dot" />
                        <span className="chat-message__typing-dot" />
                        <span className="chat-message__typing-dot" />
                        {streamStatusText && (
                          <span className="chat-message__typing-status">
                            <span>{streamStatusText}</span>
                            {showSectionProgress && (
                              <span className="chat-message__typing-progress">
                                {completedProgressText ? `\u2713 ${completedProgressText}` : 'Starting sections...'}
                                {remainingProgressText ? ` | ${remainingProgressText}` : ''}
                              </span>
                            )}
                          </span>
                        )}
                      </span>
                    ) : safeContent ? (
                      <p className="chat-message__text chat-message__text--muted">
                        The model produced only internal reasoning with no final answer. Try rephrasing your question or try again.
                      </p>
                    ) : isStopped && !isStreaming ? (
                      <p className="chat-message__text chat-message__text--muted">
                        Generation stopped by you.
                      </p>
                    ) : !isStreaming ? (
                      <p className="chat-message__text chat-message__text--muted">
                        No response was generated. Please try rephrasing your question.
                      </p>
                    ) : null}
                    {showTrailingCursor && <span className="chat-message__cursor" />}
                  </>
                )}
              </div>
            </>
          )}
        </div>
        {showActions && (
        <div className="chat-message__actions">
          {role === 'assistant' && assistantMetaItems.length > 0 && (
            <div className="chat-message__meta-left">
              {assistantMetaItems.map((item, index) => (
                <Fragment key={item.key}>
                  {index > 0 && <span className="chat-message__meta-sep">|</span>}
                  {item.node}
                </Fragment>
              ))}
            </div>
          )}
          <div className="chat-message__actions-right">
            {createdAt && (
              <span className="chat-message__time">{formatRelativeTime(createdAt)}</span>
            )}
            {((isUser && safeContent) || (!isUser && hasVisibleContent)) && (
              <button
                type="button"
                className="chat-message__copy-full"
                onClick={handleCopyFull}
                disabled={actionsDisabled}
                title="Copy message"
                aria-label="Copy message"
              >
                {copied ? <i className="ri-check-line" aria-hidden style={{ fontSize: '0.875rem' }} /> : <i className="ri-file-copy-line" aria-hidden style={{ fontSize: '0.875rem' }} />}
              </button>
            )}
          </div>
        </div>
        )}
        {role === 'assistant' && sources?.length > 0 && sourcesExpanded && (
          <div className="chat-message__sources">
            <div className="chat-message__sources-scroll">
              {sources.map((s, i) => (
                <SourceCard
                  key={i}
                  filename={s.filename}
                  path={s.path}
                  rankIndex={i}
                  rankTotal={sources.length}
                />
              ))}
            </div>
          </div>
        )}
        {role === 'assistant' && enableRawOutputControl && messageId != null && rawExpanded && (
          <div className="chat-message__raw">
            <div className="chat-message__raw-scroll">
              {rawLoading ? (
                <p className="chat-message__text chat-message__text--muted">Loading...</p>
              ) : rawContent != null ? (
                <>
                  <div className="chat-message__raw-header">
                    <div className="chat-message__raw-header-meta">
                      <div className="source-card__icon chat-message__raw-header-icon">
                        <i className="ri-code-s-slash-line" aria-hidden style={{ fontSize: '1rem' }} />
                      </div>
                      <span className="source-card__filename chat-message__raw-header-title">Raw Output</span>
                    </div>
                    <button
                      type="button"
                      className="chat-message__copy-full"
                      onClick={handleCopyRaw}
                      disabled={actionsDisabled || rawLoading}
                      title={rawCopied ? 'Copied' : 'Copy raw output'}
                      aria-label="Copy raw output"
                    >
                      <i className={rawCopied ? 'ri-check-line' : 'ri-file-copy-line'} aria-hidden style={{ fontSize: '0.875rem' }} />
                    </button>
                  </div>
                  <pre className="chat-message__raw-output">{rawContent}</pre>
                </>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function areChatMessagePropsEqual(prev: ChatMessageProps, next: ChatMessageProps): boolean {
  return (
    prev.id === next.id &&
    prev.role === next.role &&
    prev.content === next.content &&
    prev.isInternal === next.isInternal &&
    prev.isContinuation === next.isContinuation &&
    prev.chatMode === next.chatMode &&
    prev.sources === next.sources &&
    prev.displayBlocks === next.displayBlocks &&
    prev.isStreaming === next.isStreaming &&
    prev.streamStatusText === next.streamStatusText &&
    prev.streamSectionProgress === next.streamSectionProgress &&
    prev.streamPlanSteps === next.streamPlanSteps &&
    prev.isPartial === next.isPartial &&
    prev.hasRemainingScope === next.hasRemainingScope &&
    prev.completionMode === next.completionMode &&
    prev.stoppedByUser === next.stoppedByUser &&
    prev.nextAction === next.nextAction &&
    prev.nextActionReason === next.nextActionReason &&
    prev.continueLabel === next.continueLabel &&
    prev.createdAt === next.createdAt &&
    prev.generationSeconds === next.generationSeconds &&
    prev.enableRawOutputControl === next.enableRawOutputControl &&
    prev.onContinue === next.onContinue &&
    prev.onRegenerate === next.onRegenerate &&
    prev.onAssistantSwitch === next.onAssistantSwitch &&
    prev.canContinue === next.canContinue &&
    prev.canRegenerate === next.canRegenerate &&
    prev.canAssistantSwitch === next.canAssistantSwitch &&
    prev.actionsDisabled === next.actionsDisabled
  )
}

export const ChatMessage = memo(ChatMessageComponent, areChatMessagePropsEqual)
