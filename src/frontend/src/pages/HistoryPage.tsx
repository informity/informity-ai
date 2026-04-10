/**
 * Informity AI — History page
 * Chat history with table, search filter, rename, load, delete.
 * Matches Files page appearance.
 */
import { useState, useEffect, useCallback } from 'react'
import { HistoryTable } from '../components/history/HistoryTable'
import { PageHeader } from '../components/PageHeader'
import { HistoryFilters } from '../components/history/HistoryFilters'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { CenteredState } from '../components/CenteredState'
import { getChats } from '../api'
import { showToast } from '../context/useToast'
import { useBackendStatus } from '../context/useBackendStatus'
import { useDebounce } from '../utils/useDebounce'
import { extractErrorMessage } from '../utils/errorMessages'
import { isBackendConnectionError } from '../utils/networkErrors'
import type { ChatListItem } from '../types/api'
import '../pages/PlaceholderPage.css'

const PAGE_SIZE = 25
const SEARCH_DEBOUNCE_MS = 300

interface HistoryFiltersState {
  search?: string
}

export function HistoryPage() {
  const [chats, setChats] = useState<ChatListItem[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState<'title' | 'date'>('date')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [filters, setFilters] = useState<HistoryFiltersState>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { offline } = useBackendStatus()

  const debouncedSearch = useDebounce(filters.search, SEARCH_DEBOUNCE_MS)

  const loadChats = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = (await getChats({
        limit:  PAGE_SIZE,
        offset,
        search: debouncedSearch?.trim() || undefined,
      })) as { chats?: ChatListItem[]; total?: number }
      setChats(data.chats || [])
      setTotal(data.total || 0)
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to load chats')
      const disconnected = isBackendConnectionError(err)
      setError(msg)
      setChats([])
      setTotal(0)
      if (!disconnected) {
        showToast('error', msg)
      }
    } finally {
      setLoading(false)
    }
  }, [offset, debouncedSearch])

  useEffect(() => {
    loadChats()
  }, [loadChats])

  const handleSortChange = useCallback((newSort: string, newOrder: string) => {
    setSort(newSort as 'title' | 'date')
    setOrder(newOrder as 'asc' | 'desc')
    setOffset(0)
  }, [])

  const handleFiltersChange = useCallback((newFilters: HistoryFiltersState) => {
    setFilters(newFilters)
    setOffset(0)
  }, [])

  const handleChatDeleted = useCallback(() => {
    loadChats()
  }, [loadChats])

  const handleChatRenamed = useCallback(() => {
    loadChats()
  }, [loadChats])

  const handlePageChange = useCallback((newOffset: number) => {
    setOffset(newOffset)
  }, [])

  const hasSearch = (filters.search?.trim()?.length ?? 0) > 0
  const showBaseEmptyState = !loading && chats.length === 0 && total === 0 && !hasSearch

  if (offline) {
    return (
      <div className="page page--history">
        <PageHeader
          title="History"
          subtitle="Manage your chat history"
          icon="ri-history-line"
        />
        <div className="page__scroll">
          <ServiceUnavailableState />
        </div>
      </div>
    )
  }

  return (
    <div className="page page--history">
      <PageHeader
        title="History"
        subtitle="Manage your chat history"
        icon="ri-history-line"
      />
      <div className="page__scroll">
        {error && <div className="page__error">{error}</div>}
        {showBaseEmptyState ? (
          <CenteredState
            icon="ri-chat-off-line"
            title="No chats yet."
            description="Start chatting to create your first chat."
          />
        ) : (
          <>
            <HistoryFilters filters={filters} onChange={handleFiltersChange} />
            <div className="history-page__table-wrapper">
              {loading && chats.length === 0 ? (
                <div className="history-page__loading">
                  <i className="ri-loader-4-line" aria-hidden style={{ fontSize: '1.5rem' }} />
                  <span>Loading chats...</span>
                </div>
              ) : (
                <>
                  <HistoryTable
                    chats={chats}
                    total={total}
                    offset={offset}
                    limit={PAGE_SIZE}
                    sort={sort}
                    order={order}
                    search={filters.search}
                    onSortChange={handleSortChange}
                    onPageChange={handlePageChange}
                    onChatDeleted={handleChatDeleted}
                    onChatRenamed={handleChatRenamed}
                  />
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
