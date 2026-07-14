/**
 * Informity AI — Files page
 * File browser with table, filters (TASK-050), detail panel (TASK-051).
 */
import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { FileTable } from '../components/files/FileTable'
import { FileSearchResults } from '../components/files/FileSearchResults'
import { PageHeader } from '../components/PageHeader'
import { FileTableSkeleton } from '../components/files/FileTableSkeleton'
import { FileFilters } from '../components/files/FileFilters'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { CenteredState } from '../components/CenteredState'
import { getFileReindexOperation, getFiles, listFileReindexOperations, openFile, reindexFile, removeFile, searchFiles } from '../api'
import { showToast } from '../context/useToast'
import { useConfirm } from '../context/useConfirm'
import { useBackendStatus } from '../context/useBackendStatus'
import { useDebounce } from '../utils/useDebounce'
import { extractErrorMessage } from '../utils/errorMessages'
import { isBackendConnectionError } from '../utils/networkErrors'
import type { FileReindexOperation, IndexedFile } from '../types/api'
import {
  FILE_SEARCH_MODE_STORAGE_KEY,
  FILE_SEARCH_RESULT_LIMIT_STORAGE_KEY,
  FILE_SEARCH_TEXT_STORAGE_KEY,
} from '../utils/storageKeys'
import '../pages/PlaceholderPage.css'

const PAGE_SIZE = 25
const SEARCH_DEBOUNCE_MS = 300
const DEFAULT_SEMANTIC_RESULT_LIMIT = 20

type FileSearchMode = 'standard' | 'semantic'

interface FileFiltersState {
  search?: string
  extension?: string[]
}

export function FilesPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const confirm = useConfirm()
  const [files, setFiles] = useState<IndexedFile[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState('indexed_at')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [filters, setFilters] = useState<FileFiltersState>(() => {
    try {
      const storedSearch = window.localStorage.getItem(FILE_SEARCH_TEXT_STORAGE_KEY)
      return storedSearch ? { search: storedSearch } : {}
    } catch {
      return {}
    }
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [semanticResults, setSemanticResults] = useState<Awaited<ReturnType<typeof searchFiles>>['results']>([])
  const [semanticLoading, setSemanticLoading] = useState(false)
  const [semanticError, setSemanticError] = useState<string | null>(null)
  const [searchMode, setSearchMode] = useState<FileSearchMode>(() => {
    try {
      const stored = window.localStorage.getItem(FILE_SEARCH_MODE_STORAGE_KEY)
      return stored === 'semantic' ? 'semantic' : 'standard'
    } catch {
      return 'standard'
    }
  })
  const [semanticResultLimit, setSemanticResultLimit] = useState<number>(() => {
    try {
      const stored = Number(window.localStorage.getItem(FILE_SEARCH_RESULT_LIMIT_STORAGE_KEY))
      return [10, 20, 50].includes(stored) ? stored : DEFAULT_SEMANTIC_RESULT_LIMIT
    } catch {
      return DEFAULT_SEMANTIC_RESULT_LIMIT
    }
  })
  const [reindexOperationsByFileId, setReindexOperationsByFileId] = useState<Record<number, string>>({})
  const { offline } = useBackendStatus()
  const urlSearchTerm = searchParams.get('search')?.trim() ?? ''
  const hasUrlSearchTerm = searchParams.has('search')
  const urlSearchMode = searchParams.get('mode')?.trim().toLowerCase() ?? ''
  const normalizedUrlSearchMode: FileSearchMode | null = (
    urlSearchMode === 'semantic' || urlSearchMode === 'standard'
  ) ? urlSearchMode : null
  const hasDeepLinkSemanticSearch = hasUrlSearchTerm && urlSearchMode === 'semantic' && urlSearchTerm.length > 0

  const debouncedSearch = useDebounce(filters.search, SEARCH_DEBOUNCE_MS)
  const searchText = debouncedSearch?.trim() || ''
  const semanticSearchActive = searchMode === 'semantic' && searchText.length > 0

  useEffect(() => {
    try {
      const value = filters.search?.trim() || ''
      if (value) {
        window.localStorage.setItem(FILE_SEARCH_TEXT_STORAGE_KEY, value)
      } else {
        window.localStorage.removeItem(FILE_SEARCH_TEXT_STORAGE_KEY)
      }
    } catch {
      // ignore storage failures
    }
  }, [filters.search])

  useEffect(() => {
    try {
      window.localStorage.setItem(FILE_SEARCH_MODE_STORAGE_KEY, searchMode)
    } catch {
      // ignore storage failures
    }
  }, [searchMode])

  useEffect(() => {
    try {
      window.localStorage.setItem(FILE_SEARCH_RESULT_LIMIT_STORAGE_KEY, String(semanticResultLimit))
    } catch {
      // ignore storage failures
    }
  }, [semanticResultLimit])

  useEffect(() => {
    if (hasUrlSearchTerm) {
      setFilters((prev) => {
        const nextSearch = urlSearchTerm || undefined
        if ((prev.search ?? undefined) === nextSearch) return prev
        return { ...prev, search: nextSearch }
      })
    }
    if (normalizedUrlSearchMode) {
      setSearchMode((prev) => (prev === normalizedUrlSearchMode ? prev : normalizedUrlSearchMode))
    }
  }, [hasUrlSearchTerm, normalizedUrlSearchMode, urlSearchTerm])

  const loadFiles = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = (await getFiles({
        search: searchMode === 'standard' ? (searchText || undefined) : undefined,
        extension: filters.extension,
        sort,
        order,
        offset,
        limit:     PAGE_SIZE,
      })) as { files?: IndexedFile[]; total?: number }
      setFiles(data.files || [])
      setTotal(data.total ?? 0)
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to load files')
      const disconnected = isBackendConnectionError(err)
      setError(msg)
      setFiles([])
      setTotal(0)
      if (!disconnected) {
        showToast('error', msg)
      }
    } finally {
      setLoading(false)
    }
  }, [searchMode, searchText, filters.extension, sort, order, offset])

  const loadSemanticResults = useCallback(async () => {
    setSemanticLoading(true)
    setSemanticError(null)
    try {
      const requestLimit = hasDeepLinkSemanticSearch ? 200 : semanticResultLimit
      const data = await searchFiles({
        query: searchText,
        limit: requestLimit,
        fileTypes: filters.extension,
      })
      setSemanticResults(data.results || [])
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to load semantic search results')
      const disconnected = isBackendConnectionError(err)
      setSemanticError(msg)
      setSemanticResults([])
      if (!disconnected) {
        showToast('error', msg)
      }
    } finally {
      setSemanticLoading(false)
    }
  }, [filters.extension, hasDeepLinkSemanticSearch, searchText, semanticResultLimit])

  const setReindexOperationForFile = useCallback((fileId: number, operationId: string) => {
    setReindexOperationsByFileId((prev) => ({ ...prev, [fileId]: operationId }))
  }, [])

  const clearReindexOperationForFile = useCallback((fileId: number) => {
    setReindexOperationsByFileId((prev) => {
      if (!(fileId in prev)) return prev
      const next = { ...prev }
      delete next[fileId]
      return next
    })
  }, [])

  useEffect(() => {
    const entries = Object.entries(reindexOperationsByFileId)
    if (entries.length === 0) {
      return
    }

    let timeoutId: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    const poll = async () => {
      try {
        const results = await Promise.all(
          entries.map(async ([fileId, operationId]) => {
            try {
              const operation = await getFileReindexOperation(operationId)
              return { fileId: Number(fileId), operation, statusError: null as string | null }
            } catch {
              return { fileId: Number(fileId), operation: null, statusError: 'status_unavailable' as string }
            }
          }),
        )

        const completed: Array<{ fileId: number; operation: FileReindexOperation }> = []
        for (const result of results) {
          if (result.statusError) {
            // If operation status cannot be fetched (for example evicted history),
            // clear local spinner state to avoid stuck UI.
            clearReindexOperationForFile(result.fileId)
            continue
          }
          if (result.operation && result.operation.status !== 'running') {
            completed.push({ fileId: result.fileId, operation: result.operation })
          }
        }

        if (completed.length > 0 && !cancelled) {
          let hadSuccess = false
          for (const item of completed) {
            clearReindexOperationForFile(item.fileId)
            if (item.operation.status === 'completed') {
              hadSuccess = true
              showToast('success', `Reindex complete: ${item.operation.filename}`)
            } else {
              showToast('error', item.operation.error || `Reindex failed: ${item.operation.filename}`)
            }
          }
          if (hadSuccess) {
            loadFiles()
          }
        }
      } catch {
        // Keep polling while operations are active; transient failures should not
        // drop operation UI state.
      } finally {
        if (!cancelled) {
          timeoutId = setTimeout(poll, 1500)
        }
      }
    }

    poll()

    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [clearReindexOperationForFile, loadFiles, reindexOperationsByFileId])

  useEffect(() => {
    let cancelled = false

    const hydrateRunningReindexOperations = async () => {
      try {
        const response = await listFileReindexOperations('running')
        if (cancelled) return
        const hydrated: Record<number, string> = {}
        for (const operation of response.operations || []) {
          if (operation.status === 'running' && Number.isFinite(operation.file_id)) {
            hydrated[operation.file_id] = operation.operation_id
          }
        }
        setReindexOperationsByFileId((prev) => ({ ...hydrated, ...prev }))
      } catch {
        // keep local operation state; sidebar still reflects global running status
      }
    }

    void hydrateRunningReindexOperations()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    void loadFiles()
    if (semanticSearchActive) {
      void loadSemanticResults()
    }
  }, [semanticSearchActive, loadFiles, loadSemanticResults])

  useEffect(() => {
    if (offline) {
      return
    }
  }, [offline])

  const handleSortChange = useCallback((newSort: string, newOrder: string) => {
    setSort(newSort)
    setOrder(newOrder as 'asc' | 'desc')
    setOffset(0)
  }, [])

  const handleFiltersChange = useCallback((newFilters: FileFiltersState) => {
    setFilters((prev) => ({ ...prev, ...newFilters }))
    setOffset(0)
  }, [])

  const handleSearchModeChange = useCallback((mode: FileSearchMode) => {
    setSearchMode(mode)
    setOffset(0)
  }, [])

  const handlePageChange = useCallback((newOffset: number) => {
    setOffset(newOffset)
  }, [])

  const handleChatAboutFile = useCallback((file: IndexedFile) => {
    if (!file?.id) return
    navigate('/chat', {
      state: {
        scopedFileId: file.id,
        scopedFileName: file.filename,
      },
    })
  }, [navigate])

  const handleTranslateFile = useCallback((file: IndexedFile) => {
    if (!file?.id) return
    navigate('/translate', {
      state: {
        scopedFileId: file.id,
        scopedFileName: file.filename,
        scopedFileSourceProvider: 'filesystem',
      },
    })
  }, [navigate])

  const handleOpenFile = useCallback(async (file: IndexedFile) => {
    if (!file?.path?.trim()) return
    try {
      await openFile(file.path)
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to open file')
      showToast('error', msg)
    }
  }, [])

  const handleReindex = useCallback(
    async (file: IndexedFile) => {
      if (!file?.id) return
      if (reindexOperationsByFileId[file.id]) return
      try {
        const result = (await reindexFile(file.id)) as { operation_id?: string }
        if (!result.operation_id) {
          throw new Error('Reindex operation did not return operation id')
        }
        setReindexOperationForFile(file.id, result.operation_id)
      } catch (err) {
        const msg = extractErrorMessage(err, 'Re-index failed')
        showToast('error', msg)
      }
    },
    [reindexOperationsByFileId, setReindexOperationForFile],
  )

  const handleRemove = useCallback(
    async (file: IndexedFile, e: React.MouseEvent) => {
      e?.stopPropagation()
      if (!file?.id) return
      const ok = await confirm({
        title:       'Remove From Index',
        message:     `Remove "${file.filename || 'this file'}" from the index? The file will remain on disk but will no longer be searchable.`,
        confirmLabel: 'Remove',
        cancelLabel:  'Cancel',
        variant:      'danger',
        icon:       'ri-delete-bin-line',
      })
      if (!ok) return
      try {
        await removeFile(file.id)
        showToast('success', 'File removed from index')
        loadFiles()
      } catch (err) {
        const msg = extractErrorMessage(err, 'Remove failed')
        showToast('error', msg)
      }
    },
    [confirm, loadFiles],
  )
  const hasSearch = (filters.search?.trim()?.length ?? 0) > 0
  const hasExtensionFilter = Array.isArray(filters.extension) && filters.extension.length > 0
  const showBaseEmptyState = !semanticSearchActive && !loading && files.length === 0 && total === 0 && !hasSearch && !hasExtensionFilter

  if (offline) {
    return (
      <div className="page page--files">
        <PageHeader
          title="Files"
          subtitle="Browse and search indexed documents"
          icon="ri-folder-line"
        />
        <div className="page__scroll">
          <ServiceUnavailableState />
        </div>
      </div>
    )
  }

  return (
    <div className="page page--files">
      <PageHeader
        title="Files"
        subtitle="Browse and search indexed documents"
        icon="ri-folder-line"
      />
      <div className="page__scroll">
        {error && <div className="page__error">{error}</div>}
        {showBaseEmptyState ? (
          <CenteredState
            icon="ri-file-copy-2-line"
            title="No files indexed yet."
            description="Go to Settings to add folders, then run a scan."
          />
        ) : (
          <>
            <FileFilters
              filters={filters}
              onChange={handleFiltersChange}
              searchMode={searchMode}
              onSearchModeChange={handleSearchModeChange}
              semanticResultLimit={semanticResultLimit}
              onSemanticResultLimitChange={setSemanticResultLimit}
            />
            <div className="files-page__table-wrapper">
              {semanticSearchActive ? (
                semanticError ? (
                  <div className="page__error">{semanticError}</div>
                ) : (
                  <FileSearchResults
                    results={semanticResults}
                    files={files}
                    loading={semanticLoading}
                    total={semanticResults.length}
                    onChatAboutFile={handleChatAboutFile}
                    onTranslate={handleTranslateFile}
                    onReindex={handleReindex}
                    onRemove={handleRemove}
                    onOpenFile={handleOpenFile}
                    reindexingFileIds={new Set(Object.keys(reindexOperationsByFileId).map(Number))}
                  />
                )
              ) : loading && files.length === 0 ? (
                <FileTableSkeleton />
              ) : (
                <FileTable
                  files={files}
                  total={total}
                  offset={offset}
                  limit={PAGE_SIZE}
                  sort={sort}
                  order={order}
                  onSortChange={handleSortChange}
                  onPageChange={handlePageChange}
                  onChatAboutFile={handleChatAboutFile}
                  onTranslate={handleTranslateFile}
                  onReindex={handleReindex}
                  onRemove={handleRemove}
                  onOpenFile={handleOpenFile}
                  reindexingFileIds={new Set(Object.keys(reindexOperationsByFileId).map(Number))}
                />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
