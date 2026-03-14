/**
 * Informity AI — History table
 * Sortable columns, row actions (open, rename, delete). Matches FileTable appearance.
 */
import { useState, useCallback, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { formatRelativeTime } from '../../utils/formatRelativeTime'
import { formatDuration } from '../../utils/formatDuration'
import { setChatTitle, deleteChat, ApiError } from '../../api'
import { useChatContext } from '../../context/useChatContext'
import { showToast } from '../../context/useToast'
import { useConfirm } from '../../context/useConfirm'
import { SortIcon } from '../SortIcon'
import { StateMessage } from '../StateMessage'
import type { ChatListItem } from '../../types/api'
import './HistoryTable.css'

const SORT_COLUMNS = ['title', 'date']

type SortColumn = (typeof SORT_COLUMNS)[number]
type SortOrder = 'asc' | 'desc'

function getModeMeta(mode?: 'balanced' | 'analysis' | 'research'): { icon: string; label: string } | null {
  if (mode === 'research') return { icon: 'ri-search-ai-3-line', label: 'Research' }
  if (mode === 'analysis') return { icon: 'ri-flask-line', label: 'Analysis' }
  if (mode === 'balanced') return { icon: 'ri-scales-3-line', label: 'Balanced' }
  return null
}

function sortChats(chats: ChatListItem[], field: string, order: string): ChatListItem[] {
  return [...chats].sort((a, b) => {
    let valA: string
    let valB: string
    if (field === 'title') {
      valA = (a.title || a.last_message_preview || '').toLowerCase()
      valB = (b.title || b.last_message_preview || '').toLowerCase()
    } else {
      valA = a.last_message_at || a.updated_at || ''
      valB = b.last_message_at || b.updated_at || ''
    }
    if (valA < valB) return order === 'asc' ? -1 : 1
    if (valA > valB) return order === 'asc' ? 1 : -1
    return 0
  })
}

interface HistoryTableProps {
  chats?: ChatListItem[]
  total?: number
  offset?: number
  limit?: number
  sort?: SortColumn
  order?: SortOrder
  search?: string
  offline?: boolean
  onSortChange?: (col: SortColumn, order: SortOrder) => void
  onPageChange?: (offset: number) => void
  onChatDeleted?: () => void
  onChatRenamed?: () => void
}

export function HistoryTable({
  chats = [],
  total = 0,
  offset = 0,
  limit = 25,
  sort = 'date',
  order = 'desc',
  search = '',
  offline = false,
  onSortChange,
  onPageChange,
  onChatDeleted,
  onChatRenamed,
}: HistoryTableProps) {
  const navigate = useNavigate()
  const {
    currentChatId,
    activeGenerationChatId,
    isStreaming,
    selectChat,
    newChat,
    stopStreaming,
  } = useChatContext()
  const confirm = useConfirm()
  const [editingChatId, setEditingChatId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  const sorted = useMemo(() => sortChats(chats, sort, order), [chats, sort, order])
  const currentPage = Math.floor(offset / limit) + 1
  const totalPages = Math.max(1, Math.ceil(total / limit))
  const canPrev = offset > 0
  const canNext = offset + limit < total

  const handleHeaderClick = useCallback(
    (col: string) => {
      if (offline) return
      if (!SORT_COLUMNS.includes(col)) return
      const nextOrder = sort === col && order === 'desc' ? 'asc' : 'desc'
      onSortChange?.(col as SortColumn, sort === col ? (nextOrder as SortOrder) : 'desc')
    },
    [offline, sort, order, onSortChange],
  )

  const handleOpenChat = useCallback(
    (chatId: string) => {
      if (offline) return
      // Prime shared chat state immediately so selected history chat becomes active
      // even before Chat page effects run.
      void selectChat(chatId)
      navigate('/chat', { state: { chatId } })
    },
    [offline, navigate, selectChat],
  )

  const handleStartRename = useCallback((chat: ChatListItem, e?: React.MouseEvent) => {
    if (offline) return
    e?.stopPropagation()
    const title = chat.title || chat.last_message_preview?.substring(0, 60) || 'Untitled'
    setEditingChatId(chat.chat_id)
    setEditValue(title)
    setTimeout(() => editInputRef.current?.focus(), 0)
  }, [offline])

  const handleSaveRename = useCallback(
    async (chatId: string) => {
      if (offline) return
      const newTitle = editValue.trim()
      const chat = chats.find((c) => c.chat_id === chatId)
      const currentTitle = chat?.title || chat?.last_message_preview?.substring(0, 60) || 'Untitled'

      if (!newTitle || newTitle === currentTitle) {
        setEditingChatId(null)
        setEditValue('')
        return
      }

      try {
        await setChatTitle(chatId, newTitle)
        setEditingChatId(null)
        setEditValue('')
        onChatRenamed?.()
        showToast('success', 'Chat renamed')
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to rename'
        showToast('error', msg)
      }
    },
    [offline, editValue, chats, onChatRenamed],
  )

  const handleCancelRename = useCallback(() => {
    setEditingChatId(null)
    setEditValue('')
  }, [])

  const handleRenameKeyDown = useCallback(
    (e: React.KeyboardEvent, chatId: string) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleSaveRename(chatId)
      } else if (e.key === 'Escape') {
        handleCancelRename()
      }
    },
    [handleSaveRename, handleCancelRename],
  )

  const handleDelete = useCallback(
    async (chatId: string, e?: React.MouseEvent) => {
      e?.stopPropagation()
      if (offline) return
      const confirmed = await confirm({
        title:       'Delete Chat',
        message:     'Are you sure you want to delete this chat? This cannot be undone.',
        confirmLabel: 'Delete',
        cancelLabel:  'Cancel',
        variant:     'danger',
        icon:       'ri-delete-bin-line',
      })
      if (!confirmed) return

      try {
        if (isStreaming && activeGenerationChatId === chatId) {
          const stopConfirmed = await confirm({
            title: 'Stop Active Generation',
            message: 'This chat is currently generating a response. Stop generation before deleting the chat?',
            confirmLabel: 'Stop & Delete',
            cancelLabel: 'Cancel',
            variant: 'default',
            icon: 'ri-alert-line',
          })
          if (!stopConfirmed) return
          await stopStreaming()
        }
        await deleteChat(chatId)
        const deletedActiveChat = currentChatId === chatId
        if (deletedActiveChat) {
          await newChat()
        }
        onChatDeleted?.()
        window.dispatchEvent(new CustomEvent('chats-updated'))
        showToast('success', 'Chat deleted')
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to delete'
        showToast('error', msg)
      }
    },
    [offline, activeGenerationChatId, confirm, currentChatId, isStreaming, newChat, onChatDeleted, stopStreaming],
  )

  if (chats.length === 0) {
    if (offline) {
      return (
        <div className="history-table history-table--empty data-table data-table--empty">
          <StateMessage className="history-table__empty-state data-table__empty-state" />
        </div>
      )
    }

    const hasSearch = (search?.trim?.()?.length ?? 0) > 0
    return (
      <div className="history-table history-table--empty data-table data-table--empty">
        <div className="history-table__empty-state data-table__empty-state">
          <i className="ri-chat-off-line history-table__empty-icon data-table__empty-icon" aria-hidden />
          <p>{hasSearch ? 'No chats match your search.' : 'No chats.'}</p>
          <p className="history-table__empty-hint data-table__empty-hint">
            {hasSearch ? 'Try a different search term.' : 'Start chatting to create your first chat.'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={`history-table data-table${offline ? ' data-table--offline' : ''}`}>
      <div className="history-table__scroll data-table__scroll">
        <table className="history-table__table data-table__table">
          <thead>
            <tr>
              <th
                className={`history-table__th history-table__th--title history-table__th--sortable data-table__th data-table__th--sortable ${
                  sort === 'title' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('title')}
              >
                Chat Title
                <SortIcon sort={sort} order={order} column="title" />
              </th>
              <th className="history-table__th history-table__th--right data-table__th data-table__th--right">Messages</th>
              <th
                className={`history-table__th history-table__th--sortable history-table__th--right data-table__th data-table__th--sortable data-table__th--right ${
                  sort === 'date' ? 'data-table__th--sorted' : ''
                }`}
                onClick={() => handleHeaderClick('date')}
              >
                Last Activity
                <SortIcon sort={sort} order={order} column="date" />
              </th>
              <th className="history-table__th history-table__th--right data-table__th data-table__th--right">Generation Time</th>
              <th className="history-table__th history-table__th--actions data-table__th" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((chat) => {
              const preview = chat.last_message_preview?.trim() || ''
              const title =
                chat.title || (preview.length > 60 ? preview.substring(0, 60) + '…' : preview) || 'Untitled'
              const isEditing = editingChatId === chat.chat_id
              const modeMeta = getModeMeta(chat.last_response_mode_used)
              const titleIconClass = modeMeta?.icon || 'ri-chat-4-line'

              return (
                <tr
                  key={chat.chat_id}
                  className="history-table__row data-table__row"
                  onClick={() => !isEditing && !offline && handleOpenChat(chat.chat_id)}
                  tabIndex={isEditing || offline ? -1 : 0}
                  role="button"
                  aria-label={`Open chat ${title}`}
                  onKeyDown={(e) => {
                    if (isEditing || offline) return
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      handleOpenChat(chat.chat_id)
                    }
                  }}
                >
                  <td className="history-table__td history-table__td--title data-table__td">
                    <div className="history-table__title-cell">
                      {isEditing ? (
                        <input
                          ref={editInputRef}
                          type="text"
                          className="history-table__title-input"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onBlur={() => handleSaveRename(chat.chat_id)}
                          onKeyDown={(e) => handleRenameKeyDown(e, chat.chat_id)}
                          onClick={(e) => e.stopPropagation()}
                          disabled={offline}
                          maxLength={200}
                        />
                      ) : (
                        <>
                          <div className="history-table__title-row">
                            <div className="history-table__icon data-table__icon">
                              <i className={titleIconClass} aria-hidden style={{ fontSize: '1rem' }} />
                            </div>
                            <span className="history-table__title-text">{title}</span>
                          </div>
                          {chat.first_user_message && (
                            <div className="history-table__title-preview">{chat.first_user_message}</div>
                          )}
                        </>
                      )}
                    </div>
                  </td>
                  <td className="history-table__td history-table__td--right data-table__td data-table__td--right">
                    <span className="history-table__badge data-table__badge">{chat.message_count ?? 0}</span>
                  </td>
                  <td className="history-table__td history-table__td--right data-table__td data-table__td--right">
                    {chat.last_message_at ? formatRelativeTime(chat.last_message_at) : '—'}
                  </td>
                  <td className="history-table__td history-table__td--right data-table__td data-table__td--right">
                    {chat.last_generation_seconds != null
                      ? formatDuration(chat.last_generation_seconds)
                      : '—'}
                  </td>
                  <td className="history-table__td history-table__td--actions data-table__td" onClick={(e) => e.stopPropagation()}>
                    <div className="history-table__actions">
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="history-table__action-btn data-table__action-btn"
                          onClick={() => handleOpenChat(chat.chat_id)}
                          disabled={offline}
                          aria-label="Open this chat"
                        >
                          <i className="ri-eye-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">Open this chat</span>
                      </span>
                      <span className="data-table__action-wrap ui-tooltip-trigger">
                        <button
                          type="button"
                          className="history-table__action-btn data-table__action-btn"
                          onClick={(e) => handleStartRename(chat, e)}
                          aria-label="Edit chat title"
                          disabled={isEditing || offline}
                        >
                          <i className="ri-pencil-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap">Edit title</span>
                      </span>
                      <span className="data-table__action-wrap data-table__action-wrap--last ui-tooltip-trigger">
                        <button
                          type="button"
                          className="history-table__action-btn history-table__action-btn--danger data-table__action-btn"
                          onClick={(e) => handleDelete(chat.chat_id, e)}
                          disabled={offline}
                          aria-label="Delete chat"
                        >
                          <i className="ri-delete-bin-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                        </button>
                        <span className="data-table__action-tooltip ui-tooltip ui-tooltip--nowrap ui-tooltip--right-anchor">Delete chat</span>
                      </span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="history-table__pagination">
          <span className="history-table__pagination-info">
            {offset + 1}–{Math.min(offset + limit, total)} of {total}
          </span>
          <div className="history-table__pagination-buttons">
            <button
              type="button"
              className="history-table__pagination-btn data-table__icon-btn"
              disabled={offline || !canPrev}
              onClick={() => onPageChange?.(Math.max(0, offset - limit))}
            >
              <i className="ri-arrow-left-s-line" aria-hidden style={{ fontSize: '1rem' }} />
            </button>
            <span className="history-table__pagination-page">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="history-table__pagination-btn data-table__icon-btn"
              disabled={offline || !canNext}
              onClick={() => onPageChange?.(offset + limit)}
            >
              <i className="ri-arrow-right-s-line" aria-hidden style={{ fontSize: '1rem' }} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
