/**
 * Informity AI — Dashboard view
 * Status-first design: content metrics and recent activity.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import type { WheelEvent } from 'react'
import { useLocation } from 'react-router-dom'
import {
  getIndexStatus,
  getFiles,
} from '../../api'
import { useBackendStatus } from '../../context/useBackendStatus'
import { DashboardSkeleton } from './DashboardSkeleton'
import { PageHeader } from '../PageHeader'
import { ServiceUnavailableState } from '../ServiceUnavailableState'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatRelativeTime } from '../../utils/formatRelativeTime'
import { proxyWheelToContainer } from '../../utils/wheelProxy'
import type { IndexedFile, IndexStatus } from '../../types/api'
import '../../styles/shared/buttons.css'
import './DashboardView.css'

interface StatCardProps {
  icon: string
  label: string
  value: string | number
  subtitle?: string
}

function StatCard({ icon, label, value, subtitle }: StatCardProps) {
  return (
    <div className="dashboard-card ui-card ui-card--accent">
      <div className="dashboard-card__icon ui-card__icon-accent">
        <i className={icon} aria-hidden style={{ fontSize: '1.25rem' }} />
      </div>
      <div className="dashboard-card__content">
        <span className="dashboard-card__value">{value}</span>
        <span className="dashboard-card__label">{label}</span>
        {subtitle && <span className="dashboard-card__subtitle">{subtitle}</span>}
      </div>
    </div>
  )
}

export function DashboardView() {
  const { offline } = useBackendStatus()
  const location = useLocation()
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [recentFiles, setRecentFiles] = useState<IndexedFile[]>([])
  const [loading, setLoading] = useState(true)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const loadIndexStatus = useCallback(async () => {
    try {
      const data = (await getIndexStatus()) as IndexStatus
      setIndexStatus(data)
    } catch {
      setIndexStatus(null)
    }
  }, [])

  const loadRecentFiles = useCallback(async () => {
    try {
      const data = (await getFiles({ sort: 'indexed_at', order: 'desc', limit: 5 })) as { files?: IndexedFile[] }
      setRecentFiles(data.files || [])
    } catch {
      setRecentFiles([])
    }
  }, [])

  useEffect(() => {
    if (location.pathname !== '/dashboard') return
    setLoading(true)
    Promise.all([loadIndexStatus(), loadRecentFiles()]).finally(() => setLoading(false))
  }, [location.pathname, loadIndexStatus, loadRecentFiles])

  useEffect(() => {
    const handleChatsUpdated = () => {
      loadIndexStatus()
    }
    window.addEventListener('chats-updated', handleChatsUpdated)
    return () => window.removeEventListener('chats-updated', handleChatsUpdated)
  }, [loadIndexStatus])

  const handlePageWheel = useCallback((e: WheelEvent<HTMLDivElement>) => {
    proxyWheelToContainer(e, scrollContainerRef.current)
  }, [])

  if (loading && !indexStatus) {
    return <DashboardSkeleton />
  }

  if (offline) {
    return (
      <div className="page page--dashboard" onWheel={handlePageWheel}>
        <PageHeader
          title="Dashboard"
          subtitle="Indexing overview and recent scan activity"
          icon="ri-layout-grid-line"
        />
        <div className="page__scroll" ref={scrollContainerRef}>
          <ServiceUnavailableState />
        </div>
      </div>
    )
  }
  return (
    <div className="page page--dashboard" onWheel={handlePageWheel}>
      <PageHeader
        title="Dashboard"
        subtitle="Indexing overview and recent scan activity"
        icon="ri-layout-grid-line"
      />

      <div className="page__scroll" ref={scrollContainerRef}>
          <div className="dashboard__content-metrics">
            <h2 className="dashboard__section-heading ui-section-heading">
              <i className="ri-pie-chart-line dashboard__section-icon ui-section-heading__icon" aria-hidden />
              Content
            </h2>
            <div className="dashboard__cards">
              <StatCard
                icon="ri-stack-line"
                label="Chunks"
                value={indexStatus?.total_chunks?.toLocaleString() ?? 0}
              />
              <StatCard
                icon="ri-ai-generate-3d-line"
                label="Embeddings"
                value={indexStatus?.total_embeddings?.toLocaleString() ?? 0}
              />
              <StatCard
                icon="ri-chat-ai-4-line"
                label="Chats"
                value={indexStatus?.chat_count?.toLocaleString() ?? 0}
              />
            </div>
          </div>

          <div className="dashboard__storage-section">
            <h2 className="dashboard__section-heading ui-section-heading">
              <i className="ri-save-line dashboard__section-icon ui-section-heading__icon" aria-hidden />
              Storage
            </h2>
            <div className="dashboard__cards">
              <StatCard
                icon="ri-file-copy-line"
                label="Indexed Content"
                value={formatFileSize(indexStatus?.indexed_content_size_bytes)}
              />
              <StatCard
                icon="ri-database-2-line"
                label="Database"
                value={formatFileSize(indexStatus?.db_size_bytes)}
              />
              <StatCard
                icon="ri-robot-2-line"
                label="Models"
                value={formatFileSize(indexStatus?.model_size_bytes)}
              />
            </div>
          </div>

        <div className="dashboard__recent">
          <h2 className="dashboard__section-heading ui-section-heading">
            <i className="ri-time-line dashboard__section-icon ui-section-heading__icon" aria-hidden />
            Recent Scan Activity
          </h2>
          {recentFiles.length > 0 ? (
            <div className="dashboard__recent-table">
              {recentFiles.map((f) => (
                <div key={f.id} className="dashboard__recent-row">
                  <i className="ri-file-text-line dashboard__recent-icon" aria-hidden />
                  <span className="dashboard__recent-filename" title={f.filename}>{f.filename}</span>
                  <span className="dashboard__recent-time">{formatRelativeTime(f.indexed_at)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="dashboard__recent-empty data-table__empty-state">
              <i className="ri-file-copy-2-line data-table__empty-icon" aria-hidden />
              <p>No recent activity.</p>
              <p className="data-table__empty-hint">Indexed files will appear here after indexing runs.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
