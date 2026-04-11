import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getScanStatus, listFileReindexOperations } from '../api'
import { useChatContext } from '../context/useChatContext'
import './Sidebar.css'

const SCAN_STATUS_POLL_MS = 3000

const NAV_ITEMS = [
  { path: '/chat', label: 'Chat', icon: 'ri-chat-ai-4-line' },
  { path: '/history', label: 'History', icon: 'ri-history-line' },
  { path: '/files', label: 'Files', icon: 'ri-folder-line' },
  { path: '/dashboard', label: 'Dashboard', icon: 'ri-layout-grid-line' },
  { path: '/settings', label: 'Settings', icon: 'ri-settings-3-line' },
]

interface SidebarProps {
  collapsed: boolean
  onToggleCollapsed: () => void
}

export function Sidebar({ collapsed, onToggleCollapsed }: SidebarProps) {
  const {
    isStreaming,
  } = useChatContext()
  const [isScanRunning, setIsScanRunning] = useState(false)
  const [isFileReindexRunning, setIsFileReindexRunning] = useState(false)

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    const pollStatuses = async () => {
      try {
        const [scanStatus, fileReindexStatus] = await Promise.all([
          getScanStatus() as Promise<{ status?: string }>,
          listFileReindexOperations('running'),
        ])
        if (!cancelled) {
          setIsScanRunning(scanStatus?.status === 'running')
          setIsFileReindexRunning((fileReindexStatus?.running_count ?? 0) > 0)
        }
      } catch {
        if (!cancelled) {
          setIsScanRunning(false)
          setIsFileReindexRunning(false)
        }
      } finally {
        if (!cancelled) {
          timeoutId = setTimeout(pollStatuses, SCAN_STATUS_POLL_MS)
        }
      }
    }

    pollStatuses()

    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [])

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      <div className="sidebar__header">
        <div
          className="sidebar__logo-shell"
          onClick={collapsed ? onToggleCollapsed : undefined}
          role={collapsed ? 'button' : undefined}
          tabIndex={collapsed ? 0 : undefined}
          onKeyDown={collapsed ? (e) => e.key === 'Enter' && onToggleCollapsed?.() : undefined}
          title={collapsed ? 'Expand sidebar (Cmd+B)' : undefined}
          aria-label={collapsed ? 'Expand sidebar' : undefined}
        >
          <img
            src="/logo.png"
            alt="Informity AI"
            className="sidebar__logo"
          />
        </div>
        {!collapsed && <span className="sidebar__title">Informity AI</span>}
        {!collapsed && (
          <button
            type="button"
            className="sidebar__toggle"
            onClick={onToggleCollapsed}
            title="Toggle sidebar (Cmd+B)"
            aria-label="Toggle sidebar"
          >
            <i className="ri-side-bar-line" aria-hidden />
          </button>
        )}
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ path, label, icon }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) =>
              `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
            }
          >
            <i className={`${icon} sidebar__icon`} aria-hidden />
            {!collapsed && (
              <span className="sidebar__label">
                <span>{label}</span>
                {(path === '/chat' && isStreaming)
                  || (path === '/dashboard' && isScanRunning)
                  || (path === '/files' && isFileReindexRunning) ? (
                  <span className="sidebar__status-slot">
                    <span
                      className="sidebar__status"
                      aria-live="polite"
                      aria-label={path === '/chat' ? 'Generating' : path === '/dashboard' ? 'Scanning' : 'Indexing'}
                    >
                      <i className="ri-loader-4-line sidebar__status-spinner" aria-hidden />
                    </span>
                  </span>
                ) : null}
              </span>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
