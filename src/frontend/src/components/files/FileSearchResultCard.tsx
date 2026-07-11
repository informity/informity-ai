/**
 * Informity AI — Semantic file search result card
 * Compact card for semantic search hits with location context and open-file action.
 */
import { memo, useCallback, useState } from 'react'
import { openFile } from '../../api'
import { useBackendStatus } from '../../context/useBackendStatus'
import { showToast } from '../../context/useToast'
import { extractErrorMessage } from '../../utils/errorMessages'
import { getFileIcon } from '../../utils/fileFormatting'
import type { FileSearchResult } from '../../types/api'
import './FileSearchResultCard.css'

function getScoreTier(score: number): 'high' | 'medium' | 'low' {
  if (score <= 0.33) return 'high'
  if (score <= 0.66) return 'medium'
  return 'low'
}

function getRelevanceLabel(score: number): number {
  const raw = Math.round((1 - score) * 100)
  return Math.min(100, Math.max(0, raw))
}

function formatLocation(result: FileSearchResult): string | null {
  const parts: string[] = []
  if (typeof result.page_number === 'number' && Number.isFinite(result.page_number)) {
    parts.push(`Page ${result.page_number}`)
  }
  if (result.section_path?.trim()) {
    parts.push(result.section_path.trim())
  }
  if (result.block_type?.trim()) {
    parts.push(result.block_type.trim())
  }
  return parts.length > 0 ? parts.join(' · ') : null
}

interface FileSearchResultCardProps {
  result: FileSearchResult
}

function FileSearchResultCardComponent({ result }: FileSearchResultCardProps) {
  const { offline } = useBackendStatus()
  const [opening, setOpening] = useState(false)
  const canOpen = Boolean(result.path?.trim()) && !offline
  const iconClass = getFileIcon(result.filename.split('.').pop() ?? '')
  const scoreTier = getScoreTier(result.score)
  const locationLabel = formatLocation(result)

  const openResult = useCallback(async () => {
    if (!canOpen || opening || !result.path) return
    setOpening(true)
    try {
      await openFile(result.path)
    } catch (err) {
      const msg = extractErrorMessage(err, 'Failed to open file')
      showToast('error', msg)
    } finally {
      setOpening(false)
    }
  }, [canOpen, opening, result.path])

  const handleClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    event.stopPropagation()
    void openResult()
  }, [openResult])

  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    event.stopPropagation()
    void openResult()
  }, [openResult])

  return (
    <div
      className={`semantic-result-card${canOpen ? ' semantic-result-card--clickable' : ''}`}
      role={canOpen ? 'button' : undefined}
      tabIndex={canOpen ? 0 : undefined}
      onClick={canOpen ? handleClick : undefined}
      onKeyDown={canOpen ? handleKeyDown : undefined}
      aria-label={canOpen ? `Open ${result.filename}` : undefined}
    >
      <div className="semantic-result-card__header">
        <div className="semantic-result-card__icon">
          <i className={iconClass} aria-hidden style={{ fontSize: '1rem' }} />
        </div>
        <div className="semantic-result-card__meta">
          <span className="semantic-result-card__filename">{result.filename}</span>
          <span className="semantic-result-card__path" title={result.path}>
            {result.path}
          </span>
        </div>
        <div className="semantic-result-card__badges">
          <span className={`semantic-result-card__score semantic-result-card__score--${scoreTier}`}>
            {getRelevanceLabel(result.score)}
          </span>
          {locationLabel && (
            <span className="semantic-result-card__badge semantic-result-card__badge--muted">
              {locationLabel}
            </span>
          )}
        </div>
      </div>
      <div className="semantic-result-card__body">
        {result.preview?.trim() ? (
          <p className="semantic-result-card__preview">{result.preview}</p>
        ) : (
          <p className="semantic-result-card__preview semantic-result-card__preview--empty">No excerpt available.</p>
        )}
      </div>
      <div className="semantic-result-card__footer">
        <span className="semantic-result-card__badge">{result.category}</span>
        {canOpen && (
          <span className="semantic-result-card__open">
            <i className={opening ? 'ri-loader-4-line semantic-result-card__spinner' : 'ri-external-link-line'} aria-hidden />
            <span>{opening ? 'Opening…' : 'Open file'}</span>
          </span>
        )}
      </div>
    </div>
  )
}

export const FileSearchResultCard = memo(FileSearchResultCardComponent)
