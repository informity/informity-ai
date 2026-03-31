import { useState, useEffect } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'
import { NetworkBanner } from './NetworkBanner'
import { ModelOperationBanner } from './ModelOperationBanner'
import { PageFooter } from './PageFooter'
import { useBackendStatus } from '../context/useBackendStatus'
import { listenDesktopMenuActions } from '../tauriRuntime'
import '../pages/PlaceholderPage.css'
import './Layout.css'

const MENU_SCAN_NOW_PENDING_KEY = 'informity.menu.scan_now.pending'

export function Layout() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const { offline } = useBackendStatus()
  const [collapsed, setCollapsed] = useState(() => {
    const stored = localStorage.getItem('informity-sidebar-collapsed')
    return stored === 'true'
  })

  useEffect(() => {
    localStorage.setItem('informity-sidebar-collapsed', String(collapsed))
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
        if (offline) return
        if (pathname !== '/chat') navigate('/chat')
        else window.dispatchEvent(new CustomEvent('new-chat'))
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
  }, [navigate, pathname, offline])

  useEffect(() => {
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
          if (offline) break
          if (pathname !== '/chat') {
            navigate('/chat')
            setTimeout(() => window.dispatchEvent(new CustomEvent('new-chat')), 120)
          } else {
            window.dispatchEvent(new CustomEvent('new-chat'))
          }
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
        default:
          break
      }
    }).then((fn) => {
      unlisten = fn
    })

    return () => {
      if (unlisten) {
        unlisten()
      }
    }
  }, [navigate, offline, pathname])

  return (
    <div className="layout">
      <div className="layout__body">
        <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)} />
        <KeyboardShortcutsModal />
        <main className="layout__main">
          <NetworkBanner />
          <ModelOperationBanner />
          <div className="layout__main-inner">
            <Outlet />
            {pathname.startsWith('/settings') && <PageFooter />}
          </div>
        </main>
      </div>
    </div>
  )
}
