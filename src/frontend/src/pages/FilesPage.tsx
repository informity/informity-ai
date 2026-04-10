/**
 * Informity AI — Files page
 * File browser with table, filters (TASK-050), detail panel (TASK-051).
 */
import { useState, useEffect, useCallback } from 'react'
import { FileTable } from '../components/files/FileTable'
import { PageHeader } from '../components/PageHeader'
import { FileTableSkeleton } from '../components/files/FileTableSkeleton'
import { FileFilters } from '../components/files/FileFilters'
import { FileDetail } from '../components/files/FileDetail'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { CenteredState } from '../components/CenteredState'
import { getFiles, reindexFile, removeFile } from '../api'
import { showToast } from '../context/useToast'
import { useConfirm } from '../context/useConfirm'
import { useBackendStatus } from '../context/useBackendStatus'
import { useDebounce } from '../utils/useDebounce'
import { extractErrorMessage } from '../utils/errorMessages'
import { isBackendConnectionError } from '../utils/networkErrors'
import type { IndexedFile } from '../types/api'
import '../pages/PlaceholderPage.css'

const PAGE_SIZE = 25
const SEARCH_DEBOUNCE_MS = 300

interface FileFiltersState {
  search?: string
  extension?: string[]
}

export function FilesPage() {
  const confirm = useConfirm()
  const [files, setFiles] = useState<IndexedFile[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState('indexed_at')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [filters, setFilters] = useState<FileFiltersState>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFileId, setSelectedFileId] = useState<number | null>(null)
  const { offline } = useBackendStatus()

  const debouncedSearch = useDebounce(filters.search, SEARCH_DEBOUNCE_MS)

  const loadFiles = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = (await getFiles({
        search:    debouncedSearch?.trim() || undefined,
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
  }, [debouncedSearch, filters.extension, sort, order, offset])

  useEffect(() => {
    loadFiles()
  }, [loadFiles])

  useEffect(() => {
    if (offline) {
      setSelectedFileId(null)
    }
  }, [offline])

  const handleSortChange = useCallback((newSort: string, newOrder: string) => {
    setSort(newSort)
    setOrder(newOrder as 'asc' | 'desc')
    setOffset(0)
  }, [])

  const handleFiltersChange = useCallback((newFilters: FileFiltersState) => {
    setFilters(newFilters)
    setOffset(0)
  }, [])

  const handlePageChange = useCallback((newOffset: number) => {
    setOffset(newOffset)
  }, [])

  const handleSelectFile = useCallback((file: IndexedFile) => {
    setSelectedFileId(file?.id ?? null)
  }, [])

  const handleCloseDetail = useCallback(() => {
    setSelectedFileId(null)
  }, [])

  const handleFileRemoved = useCallback(() => {
    setSelectedFileId(null)
    loadFiles()
  }, [loadFiles])

  const handleReindex = useCallback(
    async (file: IndexedFile) => {
      if (!file?.id) return
      try {
        await reindexFile(file.id)
        showToast('success', 'File re-indexed')
        loadFiles()
      } catch (err) {
        const msg = extractErrorMessage(err, 'Re-index failed')
        showToast('error', msg)
      }
    },
    [loadFiles],
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
        if (selectedFileId === file.id) setSelectedFileId(null)
        loadFiles()
      } catch (err) {
        const msg = extractErrorMessage(err, 'Remove failed')
        showToast('error', msg)
      }
    },
    [confirm, loadFiles, selectedFileId],
  )
  const hasSearch = (filters.search?.trim()?.length ?? 0) > 0
  const hasExtensionFilter = Array.isArray(filters.extension) && filters.extension.length > 0
  const showBaseEmptyState = !loading && files.length === 0 && total === 0 && !hasSearch && !hasExtensionFilter

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
            <FileFilters filters={filters} onChange={handleFiltersChange} />
            <div className="files-page__table-wrapper">
              {loading && files.length === 0 ? (
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
                  onSelectFile={handleSelectFile}
                  onReindex={handleReindex}
                  onRemove={handleRemove}
                  selectedFileId={selectedFileId}
                />
              )}
            </div>
          </>
        )}
      </div>
      {selectedFileId && (
        <FileDetail
          fileId={selectedFileId}
          onClose={handleCloseDetail}
          onRemoved={handleFileRemoved}
        />
      )}
    </div>
  )
}
