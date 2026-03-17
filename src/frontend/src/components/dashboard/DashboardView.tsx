/**
 * Informity AI — Dashboard view
 * Status-first design: hero status card, content metrics, recent activity, advanced actions.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import type { WheelEvent } from 'react'
import { useLocation } from 'react-router-dom'
import {
  cancelScan,
  getIndexStatus,
  getScanStatus,
  scanFiles,
  getFiles,
  rebuildIndex,
  getSettings,
} from '../../api'
import { ApiError } from '../../api'
import { showToast } from '../../context/useToast'
import { useConfirm } from '../../context/useConfirm'
import { useBackendStatus } from '../../context/useBackendStatus'
import { DashboardSkeleton } from './DashboardSkeleton'
import { PageHeader } from '../PageHeader'
import { ServiceUnavailableState } from '../ServiceUnavailableState'
import { formatFileSize } from '../../utils/formatFileSize'
import { formatDuration } from '../../utils/formatDuration'
import { formatRelativeTime } from '../../utils/formatRelativeTime'
import { proxyWheelToContainer } from '../../utils/wheelProxy'
import type { IndexedFile } from '../../types/api'
import '../../styles/shared/buttons.css'
import './DashboardView.css'

const POLL_INTERVAL_MS = 2000
const MENU_SCAN_NOW_PENDING_KEY = 'informity.menu.scan_now.pending'

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

interface IndexStatus {
  total_files?: number
  total_chunks?: number
  total_embeddings?: number
  chat_count?: number
  indexed_content_size_bytes?: number
  db_size_bytes?: number
  vectors_size_bytes?: number
  model_size_bytes?: number
}

interface ScanStatus {
  status?: string
  started_at?: string
  files_scanned?: number
  files_indexed?: number
  errors?: number
  timeout_errors?: number
  recent_errors?: Array<{
    path?: string
    filename?: string
    operation?: string
    error_code?: string | null
    error_message?: string
    is_timeout?: boolean
    created_at?: string | null
  }>
  elapsed_seconds?: number
}

interface SettingsData {
  watched_directories?: string[]
}

type ScanNoticeSeverity = 'warning' | 'error'

interface ScanNotice {
  key: string
  severity: ScanNoticeSeverity
  icon: string
  text: string
}

const MAX_SCAN_NOTICES = 3

function getRecentErrorKey(err: NonNullable<ScanStatus['recent_errors']>[number], idx: number): string {
  const source = err.path || err.filename || `item-${idx}`
  const code = err.error_code || 'error'
  const createdAt = err.created_at || 'unknown-time'
  return `${createdAt}|${source}|${code}`
}

function getScanNoticeFromRecentError(
  err: NonNullable<ScanStatus['recent_errors']>[number],
  idx: number,
): ScanNotice {
  const fileLabel = err.filename || err.path || 'Unknown file'
  const code = err.error_code ?? ''
  const key = getRecentErrorKey(err, idx)

  if (code === 'pdf_password_protected') {
    return {
      key,
      severity: 'warning',
      icon: 'ri-lock-line',
      text: `${fileLabel}: Password-protected PDF skipped.`,
    }
  }

  if (code === 'pdf_invalid_or_corrupt') {
    return {
      key,
      severity: 'warning',
      icon: 'ri-file-warning-line',
      text: `${fileLabel}: PDF is invalid or corrupted and could not be indexed.`,
    }
  }

  if (code === 'docling_extraction_error') {
    return {
      key,
      severity: 'error',
      icon: 'ri-file-damage-line',
      text: `${fileLabel}: Document extraction failed.`,
    }
  }

  if (code === 'scan_cancelled') {
    return {
      key,
      severity: 'warning',
      icon: 'ri-close-circle-line',
      text: `${fileLabel}: Scan was cancelled before completion.`,
    }
  }

  if (err.is_timeout || code.includes('timeout')) {
    return {
      key,
      severity: 'warning',
      icon: 'ri-time-line',
      text: `${fileLabel}: Processing timed out.`,
    }
  }

  const codeLabel = code ? ` (${code})` : ''
  return {
    key,
    severity: 'error',
    icon: 'ri-alert-line',
    text: `${fileLabel}: Scan failed${codeLabel}.`,
  }
}

export function DashboardView() {
  const confirm = useConfirm()
  const { offline } = useBackendStatus()
  const location = useLocation()
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [scanStatus, setScanStatus] = useState<ScanStatus | null>(null)
  const [recentFiles, setRecentFiles] = useState<IndexedFile[]>([])
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)
  const [dismissedNoticeKeys, setDismissedNoticeKeys] = useState<Set<string>>(new Set())
  const [suppressRecentErrors, setSuppressRecentErrors] = useState(false)
  const previousScanStatusRef = useRef<string | undefined>(undefined)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const loadIndexStatus = useCallback(async () => {
    try {
      const data = (await getIndexStatus()) as IndexStatus
      setIndexStatus(data)
    } catch {
      setIndexStatus(null)
    }
  }, [])

  const loadScanStatus = useCallback(async (): Promise<boolean> => {
    try {
      const data = (await getScanStatus()) as ScanStatus
      setScanStatus(data)
      if (data?.status !== 'running') {
        setCancelling(false)
      }
      return data?.status === 'running'
    } catch {
      setScanStatus(null)
      setCancelling(false)
      return false
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

  const loadSettings = useCallback(async () => {
    try {
      const data = (await getSettings()) as SettingsData
      setSettings(data)
    } catch {
      setSettings(null)
    }
  }, [])

  useEffect(() => {
    if (location.pathname !== '/dashboard') return
    setLoading(true)
    Promise.all([loadIndexStatus(), loadScanStatus(), loadRecentFiles(), loadSettings()]).finally(() =>
      setLoading(false),
    )
  }, [location.pathname, loadIndexStatus, loadScanStatus, loadRecentFiles, loadSettings])

  useEffect(() => {
    const handleChatsUpdated = () => {
      loadIndexStatus()
    }
    window.addEventListener('chats-updated', handleChatsUpdated)
    return () => window.removeEventListener('chats-updated', handleChatsUpdated)
  }, [loadIndexStatus])

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    let cancelled = false
    let pollInFlight = false

    const poll = async () => {
      if (cancelled || pollInFlight) return
      pollInFlight = true
      const isRunning = await loadScanStatus()
      if (isRunning) {
        await loadIndexStatus()
      } else {
        await loadIndexStatus()
        await loadRecentFiles()
      }
      pollInFlight = false
      if (!cancelled && scanStatus?.status === 'running') {
        timeoutId = setTimeout(poll, POLL_INTERVAL_MS)
      }
    }

    if (scanStatus?.status === 'running') {
      timeoutId = setTimeout(poll, POLL_INTERVAL_MS)
    }

    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [scanStatus?.status, loadScanStatus, loadIndexStatus, loadRecentFiles])

  useEffect(() => {
    const previousStatus = previousScanStatusRef.current
    const currentStatus = scanStatus?.status
    if (currentStatus === 'running' && previousStatus !== 'running') {
      setDismissedNoticeKeys(new Set())
      setSuppressRecentErrors(false)
    }
    if (previousStatus === 'running' && currentStatus !== 'running') {
      const completedWithoutErrors =
        (scanStatus?.errors ?? 0) === 0 &&
        (scanStatus?.timeout_errors ?? 0) === 0 &&
        (scanStatus?.recent_errors?.length ?? 0) === 0
      if (completedWithoutErrors) {
        setScanError(null)
        setDismissedNoticeKeys(new Set())
        setSuppressRecentErrors(true)
      }
    }
    previousScanStatusRef.current = currentStatus
  }, [scanStatus])

  const dismissNotice = useCallback((key: string) => {
    setDismissedNoticeKeys((previous) => {
      const next = new Set(previous)
      next.add(key)
      return next
    })
  }, [])

  const handleScanNow = async () => {
    if (offline) return
    setScanning(true)
    setScanError(null)
    setDismissedNoticeKeys(new Set())
    setSuppressRecentErrors(false)
    try {
      let dirs = settings?.watched_directories
      if (!dirs?.length) {
        const fresh = (await getSettings()) as SettingsData
        dirs = fresh?.watched_directories
        if (fresh) setSettings(fresh)
      }
      await scanFiles(dirs ?? undefined, false)
      const isRunning = await loadScanStatus()
      if (isRunning) {
        await loadIndexStatus()
      }
      showToast('success', 'Scan started')
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Scan failed'
      setScanError(msg)
      showToast('error', msg)
    } finally {
      setScanning(false)
    }
  }

  const isScanRunning = scanStatus?.status === 'running'
  const watchedDirCount = settings?.watched_directories?.length ?? 0

  useEffect(() => {
    try {
      const pending = sessionStorage.getItem(MENU_SCAN_NOW_PENDING_KEY) === '1'
      if (pending) {
        sessionStorage.removeItem(MENU_SCAN_NOW_PENDING_KEY)
        if (!isScanRunning && !scanning && !rebuilding && !cancelling && !offline) {
          void handleScanNow()
        }
      }
    } catch {
      // ignore storage errors
    }
  }, [handleScanNow, isScanRunning, scanning, rebuilding, cancelling, offline])

  useEffect(() => {
    const handleMenuScanNow = () => {
      if (isScanRunning || scanning || rebuilding || cancelling || offline) return
      void handleScanNow()
    }
    window.addEventListener('menu-scan-now', handleMenuScanNow)
    return () => window.removeEventListener('menu-scan-now', handleMenuScanNow)
  }, [handleScanNow, isScanRunning, scanning, rebuilding, cancelling, offline])

  const handleRescanAll = async () => {
    if (offline) return
    const ok = await confirm({
      title:       'Rescan All Files',
      message:     'Rescan all files in source directories (including unchanged)? This may take a while.',
      confirmLabel: 'Rescan',
      cancelLabel:  'Cancel',
      icon:       'ri-refresh-line',
    })
    if (!ok) return
    setScanning(true)
    setScanError(null)
    setDismissedNoticeKeys(new Set())
    setSuppressRecentErrors(false)
    try {
      let dirs = settings?.watched_directories
      if (!dirs?.length) {
        const fresh = (await getSettings()) as SettingsData
        dirs = fresh?.watched_directories
        if (fresh) setSettings(fresh)
      }
      await scanFiles(dirs ?? undefined, true)
      const isRunning = await loadScanStatus()
      if (isRunning) {
        await loadIndexStatus()
      }
      showToast('success', 'Rescan started')
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Rescan failed'
      setScanError(msg)
      showToast('error', msg)
    } finally {
      setScanning(false)
    }
  }

  const handleCancelScan = async () => {
    if (offline || !isScanRunning || cancelling) return
    setCancelling(true)
    setScanError(null)
    try {
      await cancelScan()
      await loadScanStatus()
      showToast('success', 'Cancelling scan…')
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Cancel failed'
      setScanError(msg)
      showToast('error', msg)
      setCancelling(false)
    }
  }

  const handleRebuild = async () => {
    if (offline) return
    const ok = await confirm({
      title:       'Rebuild Index',
      message:     'Rebuild the entire index? This will re-extract, re-chunk, and re-embed every file.',
      confirmLabel: 'Rebuild',
      cancelLabel:  'Cancel',
      icon:       'ri-stack-line',
    })
    if (!ok) return
    setRebuilding(true)
    setScanError(null)
    try {
      await rebuildIndex(false)
      const isRunning = await loadScanStatus()
      if (isRunning) {
        await loadIndexStatus()
      }
      showToast('success', 'Index rebuild started')
    } catch (err) {
      const is409 = err instanceof ApiError && err.status === 409
      if (
        is409 &&
        (await confirm({
          title:       'Scan or Rebuild Running',
          message:     'A scan or rebuild is already running. Cancel it and rebuild anyway?',
          confirmLabel: 'Rebuild',
          cancelLabel:  'Cancel',
          icon:       'ri-error-warning-line',
        }))
      ) {
        try {
          await rebuildIndex(true)
          const isRunning = await loadScanStatus()
          if (isRunning) {
            await loadIndexStatus()
          }
          showToast('success', 'Index rebuild started')
        } catch (forceErr) {
          const msg =
            forceErr instanceof ApiError ? forceErr.detail : forceErr instanceof Error ? forceErr.message : 'Rebuild failed'
          setScanError(msg)
          showToast('error', msg)
        }
      } else {
        const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Rebuild failed'
        setScanError(msg)
        showToast('error', msg)
      }
    } finally {
      setRebuilding(false)
    }
  }

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
          subtitle="Monitor indexing status and manage scans."
          icon="ri-layout-grid-line"
        />
        <div className="page__scroll" ref={scrollContainerRef}>
          <ServiceUnavailableState />
        </div>
      </div>
    )
  }
  const recentNotices = suppressRecentErrors
    ? []
    : (scanStatus?.recent_errors ?? []).slice(0, MAX_SCAN_NOTICES).map(getScanNoticeFromRecentError)
  const scanNotices: ScanNotice[] = []
  if (scanError) {
    scanNotices.push({
      key: 'scan-error',
      severity: 'error',
      icon: 'ri-alert-circle-line',
      text: scanError,
    })
  }
  scanNotices.push(...recentNotices)
  const visibleScanNotices = scanNotices.filter((notice) => !dismissedNoticeKeys.has(notice.key))

  return (
    <div className="page page--dashboard" onWheel={handlePageWheel}>
      <PageHeader
        title="Dashboard"
        subtitle="Monitor indexing status and manage scans."
        icon="ri-layout-grid-line"
      />

      <div className="page__scroll" ref={scrollContainerRef}>
        <div className="dashboard__hero-grid">
          <div className="dashboard__hero">
            <div className="dashboard__hero-main">
              <div className="dashboard__hero-icon">
                <i className="ri-file-copy-2-line" aria-hidden />
              </div>
              <div className="dashboard__hero-title">
                <span className="dashboard__hero-number">{indexStatus?.total_files?.toLocaleString() ?? 0}</span>
                <span className="dashboard__hero-number"> Files</span>
              </div>
            </div>
            <div className="dashboard__hero-meta">
              {scanStatus?.started_at ? (
                <span>Last scan: {formatRelativeTime(scanStatus.started_at)}</span>
              ) : (
                <span>No scans have been run yet</span>
              )}
              {watchedDirCount > 0 && (
                <>
                  {' · '}
                  <span>
                    {watchedDirCount} source {watchedDirCount === 1 ? 'directory' : 'directories'}
                  </span>
                </>
              )}
            </div>
            <div className="dashboard__hero-actions">
              <button
                type="button"
                className="settings-btn settings-btn--primary"
                onClick={handleScanNow}
                disabled={offline || scanning || rebuilding || isScanRunning || cancelling}
              >
                {scanning || isScanRunning ? (
                  <i className="ri-loader-4-line dashboard__btn-icon--spin" aria-hidden />
                ) : (
                  <i className="ri-scan-2-line" aria-hidden />
                )}
                <span>{isScanRunning ? 'Scanning…' : scanning ? 'Starting…' : 'Scan Now'}</span>
              </button>
              {isScanRunning && (
                <button
                  type="button"
                  className="settings-btn settings-btn--secondary"
                  onClick={handleCancelScan}
                  disabled={offline || cancelling}
                >
                  {cancelling ? (
                    <i className="ri-loader-4-line dashboard__btn-icon--spin" aria-hidden />
                  ) : (
                    <i className="ri-close-circle-line" aria-hidden />
                  )}
                  <span>{cancelling ? 'Cancelling…' : 'Cancel Scan'}</span>
                </button>
              )}
            </div>
            {isScanRunning && scanStatus && (
              <div className="dashboard__hero-progress">
                <div className="dashboard__progress-bar">
                  <div className="dashboard__progress-fill dashboard__progress-fill--indeterminate" />
                </div>
                <div className="dashboard__progress-text">
                  {scanStatus.files_scanned} files scanned · {scanStatus.files_indexed} indexed
                  {(scanStatus.errors ?? 0) > 0 ? ` · ${scanStatus.errors} errors` : ''} ·{' '}
                  {(scanStatus.timeout_errors ?? 0) > 0 ? `${scanStatus.timeout_errors} timeouts · ` : ''}
                  {formatDuration(scanStatus.elapsed_seconds)}
                </div>
              </div>
            )}
            {visibleScanNotices.length > 0 && (
              <div className="dashboard__scan-notices" role="status" aria-live="polite">
                {visibleScanNotices.map((notice) => (
                  <div
                    key={notice.key}
                    className={`dashboard__scan-notice dashboard__scan-notice--${notice.severity}`}
                  >
                    <i className={notice.icon} aria-hidden />
                    <span>{notice.text}</span>
                    <button
                      type="button"
                      className="dashboard__scan-notice-dismiss"
                      aria-label="Dismiss scan notice"
                      onClick={() => dismissNotice(notice.key)}
                    >
                      <i className="ri-close-line" aria-hidden />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="dashboard__content-metrics ui-section-divider">
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

          <div className="dashboard__storage-section ui-section-divider">
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
        </div>

        <div className="dashboard__recent ui-section-divider">
          <h2 className="dashboard__section-heading ui-section-heading">
            <i className="ri-time-line dashboard__section-icon ui-section-heading__icon" aria-hidden />
            Recent Activity
          </h2>
          {recentFiles.length > 0 ? (
            <div className="dashboard__recent-table">
              {recentFiles.map((f) => (
                <div key={f.id} className="dashboard__recent-row">
                  <i className="ri-file-text-line dashboard__recent-icon" aria-hidden />
                  <span className="dashboard__recent-filename">{f.filename}</span>
                  <span className="dashboard__recent-time">{formatRelativeTime(f.indexed_at)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="dashboard__recent-empty data-table__empty-state">
              <i className="ri-file-copy-2-line data-table__empty-icon" aria-hidden />
              <p>No recent activity.</p>
              <p className="data-table__empty-hint">Indexed files will appear here after a scan.</p>
            </div>
          )}
        </div>

        <div className="dashboard__advanced ui-section-divider">
          <h2 className="dashboard__section-heading ui-section-heading">
            <i className="ri-folder-settings-line dashboard__section-icon ui-section-heading__icon" aria-hidden />
            Advanced
          </h2>
          <div className="dashboard__advanced-content">
            <div className="dashboard__advanced-actions">
              <button
                type="button"
                className="settings-btn settings-btn--secondary"
                onClick={handleRescanAll}
                disabled={offline || scanning || rebuilding || isScanRunning}
              >
                <i className="ri-refresh-line" aria-hidden />
                <span>Rescan All Files</span>
              </button>
              <button
                type="button"
                className="settings-btn settings-btn--secondary"
                onClick={handleRebuild}
                disabled={offline || scanning || rebuilding || isScanRunning}
              >
                <i className="ri-stack-line" aria-hidden />
                <span>{rebuilding ? 'Rebuilding…' : 'Rebuild Index'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
