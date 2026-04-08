/**
 * Informity AI — File detail panel
 * Sliding panel with file info, preview, actions.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { getFile, reindexFile, removeFile, ApiError } from '../../api'
import { useBackendStatus } from '../../context/useBackendStatus'
import { showToast } from '../../context/useToast'
import { useConfirm } from '../../context/useConfirm'
import { formatCategory, getFileIcon } from '../../utils/fileFormatting'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatDate } from '../../utils/formatDate'
import type { IndexedFile } from '../../types/api'
import './FileDetail.css'

const PREVIEW_CHARS = 2000

interface FileDetailProps {
  fileId: number | null
  onClose?: () => void
  onRemoved?: () => void
}

export function FileDetail({ fileId, onClose, onRemoved }: FileDetailProps) {
  const { offline } = useBackendStatus()
  const [file, setFile] = useState<IndexedFile | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [actionLoading, setActionLoading] = useState<'reindex' | 'remove' | null>(null)
  const confirm = useConfirm()
  const panelRef = useRef<HTMLDivElement>(null)

  const handleWheel = useCallback((e: WheelEvent) => {
    const panel = panelRef.current
    if (!panel) return
    const previewScroll = panel.querySelector('.file-detail__preview-scroll')
    if (previewScroll?.contains(e.target as Node)) {
      const el = previewScroll as HTMLElement
      const { scrollTop, scrollHeight, clientHeight } = el
      const atTop = scrollTop <= 0 && e.deltaY < 0
      const atBottom = scrollTop + clientHeight >= scrollHeight && e.deltaY > 0
      if (atTop || atBottom) e.preventDefault()
      return
    }
    e.preventDefault()
  }, [])

  useEffect(() => {
    if (!fileId) {
      setFile(null)
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    getFile(fileId)
      .then((data) => {
        setFile(data as IndexedFile)
        setExpanded(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to load file')
        setFile(null)
      })
      .finally(() => setLoading(false))
  }, [fileId])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose?.()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  useEffect(() => {
    const panel = panelRef.current
    if (!panel) return
    panel.addEventListener('wheel', handleWheel, { passive: false })
    return () => panel.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  const handleReindex = async () => {
    if (offline) return
    if (!fileId) return
    setActionLoading('reindex')
    try {
      await reindexFile(fileId)
      const data = (await getFile(fileId)) as IndexedFile
      setFile(data)
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Re-index failed'
      setError(msg)
      showToast('error', msg)
    } finally {
      setActionLoading(null)
    }
  }

  const handleRemove = async () => {
    if (offline) return
    if (!fileId || !file) return
    const ok = await confirm({
      title:       'Remove From Index',
      message:     `Remove "${file.filename || 'this file'}" from the index? The file will remain on disk but will no longer be searchable.`,
      confirmLabel: 'Remove',
      cancelLabel:  'Cancel',
      variant:     'danger',
      icon:       'ri-delete-bin-line',
    })
    if (!ok) return
    setActionLoading('remove')
    try {
      await removeFile(fileId)
      onRemoved?.()
      onClose?.()
      showToast('success', 'File removed from index')
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Remove failed'
      setError(msg)
      showToast('error', msg)
    } finally {
      setActionLoading(null)
    }
  }

  if (!fileId) return null

  const previewText = file?.extracted_text_preview ?? ''
  const showMore = previewText.length > PREVIEW_CHARS
  const displayText = expanded || !showMore ? previewText : previewText.slice(0, PREVIEW_CHARS)

  return (
    <div className="file-detail" role="dialog" aria-modal="true" aria-label="File Details">
      <div className="file-detail__backdrop" onClick={onClose} aria-hidden />
      <div className="file-detail__panel" ref={panelRef}>
        <div className="file-detail__header">
          <h2 className="file-detail__title">
            <span className="file-detail__title-icon">
              <i className={file ? getFileIcon(file.extension) : 'ri-file-line'} aria-hidden />
            </span>
            File Details
          </h2>
          <button type="button" className="file-detail__close" onClick={onClose} title="Close (Escape)">
            <i className="ri-close-line" aria-hidden style={{ fontSize: '1.25rem' }} />
          </button>
        </div>

        {loading ? (
          <div className="file-detail__body">
            <div className="file-detail__loading">Loading...</div>
          </div>
        ) : error ? (
          <div className="file-detail__body">
            <div className="file-detail__error">{error}</div>
          </div>
        ) : file ? (
          <div className="file-detail__body">
            <div className="file-detail__section file-detail__section--meta">
              <div className="file-detail__subsection-head ui-subsection-head">
                <h3 className="file-detail__section-title ui-subsection-title">
                  {file.filename}
                  <span className="file-detail__info file-detail__info--tooltip-below ui-tooltip-trigger" title="Full path">
                    <i className="ri-information-line" aria-hidden />
                    <span className="file-detail__tooltip file-detail__tooltip--path ui-tooltip ui-tooltip--path ui-tooltip--below">{file.path}</span>
                  </span>
                </h3>
              </div>
              <div className="file-detail__profile-grid">
                <div className="file-detail__profile-row">
                  <span className="file-detail__profile-row__label">Category</span>
                  <span className="file-detail__profile-row__value">{formatCategory(file.category)}</span>
                </div>
                <div className="file-detail__profile-row">
                  <span className="file-detail__profile-row__label">Size</span>
                  <span className="file-detail__profile-row__value">{formatFileSize(file.size_bytes)}</span>
                </div>
                <div className="file-detail__profile-row">
                  <span className="file-detail__profile-row__label">Modified</span>
                  <span className="file-detail__profile-row__value">{formatDate(file.modified_at)}</span>
                </div>
                <div className="file-detail__profile-row">
                  <span className="file-detail__profile-row__label">Indexed</span>
                  <span className="file-detail__profile-row__value">{formatDate(file.indexed_at)}</span>
                </div>
                <div className="file-detail__profile-row">
                  <span className="file-detail__profile-row__label">Chunks</span>
                  <span className="file-detail__profile-row__value">{file.chunk_count ?? 0}</span>
                </div>
              </div>
            </div>

            <div className="file-detail__section file-detail__section--preview">
              <div className="file-detail__subsection-head ui-subsection-head">
                <h3 className="file-detail__section-title ui-subsection-title">
                  Extracted Text Preview
                  <span className="file-detail__info ui-tooltip-trigger" title="Content hash">
                    <i className="ri-information-line" aria-hidden />
                    <span className="file-detail__tooltip file-detail__tooltip--path ui-tooltip ui-tooltip--path">File Hash: {file.content_hash}</span>
                  </span>
                </h3>
                <p className="file-detail__subsection-description ui-subsection-description">
                  Sample of text extracted and indexed from this file. Allows verification of extraction quality.
                </p>
              </div>
              <div className="file-detail__preview-scroll-wrap">
                <div className="file-detail__preview-scroll">
                  <div className="file-detail__preview">
                    {previewText ? (
                      <>
                        <pre className="file-detail__preview-text">{displayText}</pre>
                        {showMore && (
                          <button
                            type="button"
                            className="file-detail__show-more"
                            onClick={() => setExpanded((prev) => !prev)}
                          >
                            {expanded ? 'Show less' : 'Show more'}
                          </button>
                        )}
                      </>
                    ) : (
                      <p className="file-detail__preview-empty">No extracted text.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="file-detail__actions">
              <button
                type="button"
                className="file-detail__btn file-detail__btn--secondary"
                onClick={handleReindex}
                disabled={offline || actionLoading !== null}
                title="Re-extract and re-index this file (same as Rebuild Index for this file)"
              >
                <i className="ri-refresh-line" aria-hidden style={{ fontSize: '1rem' }} />
                <span>{actionLoading === 'reindex' ? 'Reindexing...' : 'Reindex'}</span>
              </button>
              <button
                type="button"
                className="file-detail__btn file-detail__btn--danger"
                onClick={handleRemove}
                disabled={offline || actionLoading !== null}
                title="Remove From Index"
              >
                <i className="ri-delete-bin-line" aria-hidden style={{ fontSize: '1rem' }} />
                <span>{actionLoading === 'remove' ? 'Removing…' : 'Remove From Index'}</span>
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
