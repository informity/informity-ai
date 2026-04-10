/**
 * Informity AI — Source citation card
 * Compact card for RAG sources (filename + relevance).
 * Click to open file in system default application.
 */
import { memo, useState, useCallback } from 'react'
import { openFile, ApiError } from '../../api'
import { useBackendStatus } from '../../context/useBackendStatus'
import { showToast } from '../../context/useToast'
import { getFileIcon } from '../../utils/fileFormatting'
import './SourceCard.css'

function computeEvidenceRank(rankIndex: number | undefined, rankTotal: number | undefined): number | null {
  if (typeof rankIndex !== 'number' || typeof rankTotal !== 'number') return null
  if (!Number.isFinite(rankIndex) || !Number.isFinite(rankTotal)) return null
  if (rankTotal <= 0 || rankIndex < 0 || rankIndex >= rankTotal) return null
  if (rankTotal === 1) return 100
  const spread = 80
  const rank = 100 - (rankIndex * spread / (rankTotal - 1))
  return Math.round(Math.min(100, Math.max(20, rank)))
}

function getEvidenceTier(rank: number | null): 'high' | 'medium' | 'low' | null {
  if (rank == null) return null
  if (rank >= 67) return 'high'
  if (rank >= 34) return 'medium'
  return 'low'
}

function getEvidenceTierLabel(tier: 'high' | 'medium' | 'low' | null): string {
  if (tier === 'high') return 'Strong'
  if (tier === 'medium') return 'Medium'
  return 'Weak'
}

interface SourceCardProps {
  filename?: string
  path?: string
  rankIndex?: number
  rankTotal?: number
}

function SourceCardComponent({
  filename,
  path,
  rankIndex,
  rankTotal,
}: SourceCardProps) {
  const { offline } = useBackendStatus()
  const [opening, setOpening] = useState(false)
  const ext = filename ? filename.split('.').pop() ?? '' : ''
  const iconClass = getFileIcon(ext)
  const evidenceRank = computeEvidenceRank(rankIndex, rankTotal)
  const evidenceTier = getEvidenceTier(evidenceRank)
  const canOpen = Boolean(path?.trim()) && !offline

  const openSource = useCallback(
    async () => {
      if (!canOpen || opening || !path) return
      setOpening(true)
      try {
        await openFile(path)
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to open file'
        showToast('error', msg)
      } finally {
        setOpening(false)
      }
    },
    [path, canOpen, opening],
  )
  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    void openSource()
  }, [openSource])
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    e.preventDefault()
    e.stopPropagation()
    void openSource()
  }, [openSource])

  return (
    <div
      className={`source-card ${canOpen ? 'source-card--clickable' : ''}`}
      role={canOpen ? 'button' : undefined}
      tabIndex={canOpen ? 0 : undefined}
      onClick={canOpen ? handleClick : undefined}
      onKeyDown={canOpen ? handleKeyDown : undefined}
      title={canOpen ? 'Open file' : undefined}
      aria-label={canOpen ? `Open ${filename?.trim() || 'file'}` : undefined}
    >
      <div className="source-card__header">
        <div className="source-card__icon">
          <i className={iconClass} aria-hidden style={{ fontSize: '1rem' }} />
        </div>
        <div className="source-card__meta">
          <span className="source-card__filename">{filename?.trim() || 'Unknown'}</span>
        </div>
        <div className="source-card__badges">
          {canOpen && (
            <span className="source-card__tooltip-wrap ui-tooltip-trigger" aria-hidden>
              <span className="source-card__open">
                <i className="ri-external-link-line" />
              </span>
              <span className="source-card__tooltip ui-tooltip ui-tooltip--compact ui-tooltip--nowrap ui-tooltip--right-anchor">Open file</span>
            </span>
          )}
          {evidenceRank !== null && (
            <span className="source-card__tooltip-wrap ui-tooltip-trigger">
              <span className={`source-card__score source-card__score--${evidenceTier}`}>
                {evidenceRank}
              </span>
              <span className="source-card__tooltip ui-tooltip ui-tooltip--compact ui-tooltip--right-anchor">
                {`Evidence score: ${evidenceRank}/100 (${getEvidenceTierLabel(evidenceTier)}). Relative to other sources in this answer, not an absolute measure.`}
              </span>
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export const SourceCard = memo(SourceCardComponent)
