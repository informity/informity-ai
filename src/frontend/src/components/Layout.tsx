import { useState, useEffect, useCallback } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'
import { NetworkBanner } from './NetworkBanner'
import { PageFooter } from './PageFooter'
import { UpdateCheckModal } from './UpdateCheckModal'
import { useBackendStatus } from '../context/useBackendStatus'
import { listenDesktopMenuActions, openExternalUrl } from '../tauriRuntime'
import {
  MENU_NEW_CHAT_PENDING_KEY,
  MENU_SCAN_NOW_PENDING_KEY,
  SIDEBAR_COLLAPSED_KEY,
} from '../utils/storageKeys'
import {
  checkForUpdates,
  persistUpdateCheckResult,
  UPDATE_CHECK_EVENT,
  type UpdateCheckResult,
} from '../utils/updateCheck'
import '../pages/PlaceholderPage.css'
import './Layout.css'

export function Layout() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const { offline } = useBackendStatus()
  const [collapsed, setCollapsed] = useState(() => {
    const stored = localStorage.getItem(SIDEBAR_COLLAPSED_KEY)
    return stored === 'true'
  })
  const [updateModalOpen, setUpdateModalOpen] = useState(false)
  const [updateModalState, setUpdateModalState] = useState<'checking' | 'up_to_date' | 'update_available' | 'error'>('checking')
  const [updateCheckPending, setUpdateCheckPending] = useState(false)
  const [updateResult, setUpdateResult] = useState<UpdateCheckResult | null>(null)
  const [updateError, setUpdateError] = useState<string | null>(null)

  const queuePendingNewChat = useCallback(() => {
    try {
      sessionStorage.setItem(MENU_NEW_CHAT_PENDING_KEY, '1')
    } catch {
      // ignore storage errors; fallback behavior remains direct dispatch when already on /chat
    }
  }, [])

  const dispatchNewChat = useCallback(() => {
    window.dispatchEvent(new CustomEvent('new-chat'))
  }, [])

  const requestNewChat = useCallback(() => {
    if (offline) return
    if (pathname !== '/chat') {
      queuePendingNewChat()
      navigate('/chat')
      return
    }
    dispatchNewChat()
  }, [offline, pathname, queuePendingNewChat, navigate, dispatchNewChat])

  useEffect(() => {
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed))
  }, [collapsed])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === 'b') {
        e.preventDefault()
        setCollapsed((c) => !c)
        return
      }
      if (mod && e.key === 'n') {
        e.preventDefault()
        requestNewChat()
        return
      }
      if (mod && e.key === ',') {
        e.preventDefault()
        navigate('/settings')
        return
      }
      if (mod && e.key >= '1' && e.key <= '5') {
        e.preventDefault()
        const routes = ['/chat', '/history', '/files', '/dashboard', '/settings']
        navigate(routes[parseInt(e.key, 10) - 1])
        return
      }
      if (e.key === '/' && mod) {
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('open-keyboard-shortcuts'))
        return
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [navigate, requestNewChat])

  useEffect(() => {
    const isExternalHref = (href: string): boolean => {
      const normalized = href.trim().toLowerCase()
      return (
        normalized.startsWith('http://')
        || normalized.startsWith('https://')
        || normalized.startsWith('mailto:')
        || normalized.startsWith('tel:')
      )
    }

    const handleDocumentClick = (event: MouseEvent) => {
      if (event.defaultPrevented) return
      const target = event.target as HTMLElement | null
      const anchor = target?.closest('a[href]') as HTMLAnchorElement | null
      if (!anchor) return
      const href = anchor.getAttribute('href') || ''
      if (!isExternalHref(href)) return
      event.preventDefault()
      void openExternalUrl(href)
    }

    document.addEventListener('click', handleDocumentClick)
    return () => {
      document.removeEventListener('click', handleDocumentClick)
    }
  }, [])

  useEffect(() => {
    if (pathname !== '/chat') return
    let pending = false
    try {
      pending = sessionStorage.getItem(MENU_NEW_CHAT_PENDING_KEY) === '1'
      if (pending) sessionStorage.removeItem(MENU_NEW_CHAT_PENDING_KEY)
    } catch {
      pending = false
    }
    if (!pending) return
    dispatchNewChat()
  }, [dispatchNewChat, pathname])

  useEffect(() => {
    const runUpdateCheck = async () => {
      if (updateCheckPending) return
      setUpdateModalOpen(true)
      setUpdateModalState('checking')
      setUpdateError(null)
      setUpdateCheckPending(true)
      try {
        const result = await checkForUpdates()
        setUpdateResult(result)
        persistUpdateCheckResult(result)
        setUpdateModalState(result.updateAvailable ? 'update_available' : 'up_to_date')
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        setUpdateError(message)
        setUpdateModalState('error')
      } finally {
        setUpdateCheckPending(false)
      }
    }

    const onRequest = () => {
      void runUpdateCheck()
    }

    const isVisible = (element: HTMLElement) => {
      const style = window.getComputedStyle(element)
      return style.display !== 'none' && style.visibility !== 'hidden'
    }

    const focusPrimaryInput = () => {
      const selectors = [
        '.filter-search__input',
        '.chat-view__textarea',
        '.settings-input',
      ]
      for (const selector of selectors) {
        const el = document.querySelector<HTMLElement>(selector)
        if (!el || !isVisible(el)) continue
        el.focus()
        return
      }
    }

    let unlisten: (() => void) | null = null
    void listenDesktopMenuActions((action) => {
      switch (action) {
        case 'preferences':
          navigate('/settings')
          break
        case 'new-chat':
          requestNewChat()
          break
        case 'scan-now':
          try {
            sessionStorage.setItem(MENU_SCAN_NOW_PENDING_KEY, '1')
          } catch {
            // ignore storage errors; direct event still covers mounted dashboard.
          }
          navigate('/dashboard')
          window.dispatchEvent(new CustomEvent('menu-scan-now'))
          break
        case 'toggle-sidebar':
          setCollapsed((c) => !c)
          break
        case 'focus-search':
          focusPrimaryInput()
          break
        case 'check-updates':
          void runUpdateCheck()
          break
        default:
          break
      }
    })
      .then((fn) => {
        unlisten = fn
      })
      .catch(() => {
        // ignore desktop listener setup failures
      })

    window.addEventListener(UPDATE_CHECK_EVENT, onRequest as EventListener)

    return () => {
      window.removeEventListener(UPDATE_CHECK_EVENT, onRequest as EventListener)
      if (unlisten) {
        unlisten()
      }
    }
  }, [navigate, requestNewChat, updateCheckPending])

  return (
    <div className="layout">
      <div className="layout__body">
        <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)} />
        <KeyboardShortcutsModal />
        <main className="layout__main">
          <NetworkBanner />
          <div className="layout__main-inner">
            <Outlet />
            {pathname.startsWith('/settings') && <PageFooter />}
          </div>
        </main>
      </div>
      <UpdateCheckModal
        open={updateModalOpen}
        state={updateModalState}
        checking={updateCheckPending}
        currentVersion={updateResult?.currentVersion ?? null}
        latestVersion={updateResult?.latestVersion ?? null}
        releaseNotes={updateResult?.metadata?.release_notes ?? null}
        errorMessage={updateError}
        onClose={() => {
          if (updateCheckPending) return
          setUpdateModalOpen(false)
        }}
        onRetry={() => {
          if (updateCheckPending) return
          window.dispatchEvent(new CustomEvent(UPDATE_CHECK_EVENT))
        }}
        onDownload={() => {
          if (updateCheckPending) return
          const url = updateResult?.metadata?.download_url
          if (url) {
            void openExternalUrl(url)
          }
          setUpdateModalOpen(false)
        }}
      />
    </div>
  )
}
