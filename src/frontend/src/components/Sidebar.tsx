import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { getScanStatus, listFileReindexOperations } from '../api'
import { useChatContext } from '../context/useChatContext'
import { useOptionalTranslateContext } from '../context/useTranslateContext'
import { SCAN_ACTION_STATE_EVENT, type ScanActionStateDetail } from '../utils/scanActionState'
import './Sidebar.css'

const SCAN_STATUS_POLL_MS = 3000

const NAV_ITEMS = [
  { path: '/chat',      label: 'Chat',      icon: 'ri-chat-ai-4-line',  devOnly: false },
  { path: '/translate', label: 'Translate', icon: 'ri-translate-2',     devOnly: false },
  { path: '/files',     label: 'Files',     icon: 'ri-folder-line',     devOnly: false },
  { path: '/dashboard', label: 'Dashboard', icon: 'ri-layout-grid-line',devOnly: false },
  { path: '/history',   label: 'History',   icon: 'ri-history-line',    devOnly: false },
  { path: '/settings',  label: 'Settings',  icon: 'ri-settings-3-line', devOnly: false },
]

const SETTINGS_SUBNAV = [
  { section: 'general',     label: 'General',      icon: 'ri-home-gear-line'    },
  { section: 'data',        label: 'Data Sources', icon: 'ri-folder-line'       },
  { section: 'indexing',    label: 'Indexing',     icon: 'ri-stack-line'        },
  { section: 'chat',        label: 'Chat',         icon: 'ri-chat-ai-4-line'    },
  { section: 'translate',   label: 'Translate',    icon: 'ri-translate-2'       },
  { section: 'models',      label: 'Models',       icon: 'ri-robot-2-line'      },
  { section: 'integrations',label: 'Integrations', icon: 'ri-function-add-line' },
  { section: 'diagnostics', label: 'Diagnostics',  icon: 'ri-pulse-line'        },
  { section: 'system',      label: 'System',       icon: 'ri-server-line'       },
]

interface SidebarProps {
  collapsed: boolean
  onToggleCollapsed: () => void
}

export function Sidebar({ collapsed, onToggleCollapsed }: SidebarProps) {
  const navigate = useNavigate()
  const { pathname, search } = useLocation()
  const activeSettingsSection = pathname === '/settings'
    ? (new URLSearchParams(search).get('section') ?? 'general')
    : null
  const { isStreaming } = useChatContext()
  const translateCtx = useOptionalTranslateContext()
  const isTranslating = translateCtx?.isTranslating ?? false
  const [isScanRunning, setIsScanRunning] = useState(false)
  const [isScanActionPending, setIsScanActionPending] = useState(false)
  const [isFileReindexRunning, setIsFileReindexRunning] = useState(false)

  useEffect(() => {
    const handleScanActionState = (event: Event) => {
      const detail = (event as CustomEvent<ScanActionStateDetail>).detail
      setIsScanActionPending(Boolean(detail?.running))
    }

    window.addEventListener(SCAN_ACTION_STATE_EVENT, handleScanActionState)
    return () => {
      window.removeEventListener(SCAN_ACTION_STATE_EVENT, handleScanActionState)
    }
  }, [])

  const handleNavClick = (path: string) => {
    if (collapsed && path === '/settings') {
      onToggleCollapsed()
    }
    navigate(path)
  }

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
        {NAV_ITEMS.filter(({ devOnly }) => !devOnly || import.meta.env.DEV).map(({ path, label, icon }) => {
          const showSpinner =
            (path === '/chat' && isStreaming)
            || (path === '/settings' && (isScanActionPending || isScanRunning))
            || (path === '/files' && isFileReindexRunning)
            || (path === '/translate' && isTranslating)
          const spinnerLabel =
            path === '/chat' ? 'Generating'
            : path === '/settings' ? 'Scanning'
            : path === '/files' ? 'Indexing'
            : 'Translating'
          const isSettings = path === '/settings'
          return (
            <div key={path}>
              <button
                type="button"
                className={`sidebar__link ${pathname === path ? 'sidebar__link--active' : ''}`}
                onClick={() => handleNavClick(path)}
                aria-label={label}
                aria-current={pathname === path ? 'page' : undefined}
              >
                <i className={`${icon} sidebar__icon`} aria-hidden />
                {!collapsed && (
                  <span className="sidebar__label">
                    <span>{label}</span>
                    {showSpinner ? (
                      <span className="sidebar__status-slot">
                        <span className="sidebar__status" aria-live="polite" aria-label={spinnerLabel}>
                          <i className="ri-loader-4-line sidebar__status-spinner" aria-hidden />
                        </span>
                      </span>
                    ) : null}
                  </span>
                )}
              </button>
              {isSettings && !collapsed && pathname === '/settings' && (
                <div className="sidebar__subnav">
                  {SETTINGS_SUBNAV.map(({ section, label: subLabel, icon: subIcon }) => (
                    <button
                      key={section}
                      type="button"
                      className={`sidebar__subnav-item${activeSettingsSection === section ? ' sidebar__subnav-item--active' : ''}`}
                      onClick={() => navigate(`/settings?section=${section}`)}
                    >
                      <i className={`${subIcon} sidebar__subnav-icon`} aria-hidden />
                      <span>{subLabel}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
