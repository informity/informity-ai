import { useState, useEffect } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'
import { NetworkBanner } from './NetworkBanner'
import { PageFooter } from './PageFooter'
import { useBackendStatus } from '../context/useBackendStatus'
import '../pages/PlaceholderPage.css'
import './Layout.css'

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
    </div>
  )
}
