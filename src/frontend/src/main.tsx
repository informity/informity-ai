import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'remixicon/fonts/remixicon.css'
import './index.css'
import App from './App'
import { BootOverlay } from './components/BootOverlay'
import { bootstrapDesktopBackend, StartupFailureError } from './tauriRuntime'
import { getStartupErrorMessage, type StartupFailureInfo } from './startupErrors'
import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_STORAGE_KEY } from './utils/uiTheme'

function applyTheme(theme: string | null | undefined) {
  const normalized = normalizeUiTheme(theme) ?? UI_THEME_DEFAULT
  document.documentElement.setAttribute('data-accent', normalized)
  try {
    localStorage.setItem(UI_THEME_STORAGE_KEY, normalized)
  } catch {
    // ignore
  }
}

async function initializeTheme() {
  try {
    const response = await fetch('/api/settings', { cache: 'no-store' })
    if (response.ok) {
      const payload = await response.json() as { ui_theme?: string }
      applyTheme(payload?.ui_theme)
      return
    }
  } catch {
    // fallback to local cache below
  }
  try {
    const saved = localStorage.getItem(UI_THEME_STORAGE_KEY)
    applyTheme(saved)
  } catch {
    applyTheme(UI_THEME_DEFAULT)
  }
}

function hideBootOverlay() {
  const overlay = document.getElementById('boot-overlay')
  if (!overlay) return
  overlay.classList.add('boot-overlay--hidden')
  window.setTimeout(() => overlay.remove(), 220)
}

async function renderApp() {
  let startupFailure: StartupFailureInfo | null = null
  const bootOverlayElement = document.getElementById('boot-overlay')
  const bootOverlayRoot = bootOverlayElement ? createRoot(bootOverlayElement) : null
  const bootStartedAt = Date.now()
  let bootMessage = 'Starting application…'
  let bootDescription: string | null = null
  const renderBootOverlay = () => {
    if (!bootOverlayRoot) return
    bootOverlayRoot.render(
      <StrictMode>
        <BootOverlay
          title={bootMessage}
          elapsedSeconds={Math.max(0, Math.floor((Date.now() - bootStartedAt) / 1000))}
          description={bootDescription}
        />
      </StrictMode>,
    )
  }

  renderBootOverlay()
  await initializeTheme()
  const elapsedTimerId = window.setInterval(renderBootOverlay, 1000)
  const longStartTimerId = window.setTimeout(() => {
    bootMessage = 'Still starting…'
    renderBootOverlay()
  }, 20000)

  try {
    await bootstrapDesktopBackend((payload) => {
      bootMessage = payload.message
      bootDescription = null
      renderBootOverlay()
    })
    window.clearTimeout(longStartTimerId)
    window.clearInterval(elapsedTimerId)
    bootMessage = 'Loading interface…'
    bootDescription = null
    renderBootOverlay()
  } catch (error) {
    window.clearTimeout(longStartTimerId)
    window.clearInterval(elapsedTimerId)
    if (error instanceof StartupFailureError) {
      startupFailure = { reason: error.reason, detail: error.detail }
    } else {
      startupFailure = { reason: null, detail: error instanceof Error ? error.message : String(error) }
    }
    bootMessage = 'Startup failed.'
    bootDescription = getStartupErrorMessage(startupFailure.reason)
    renderBootOverlay()
  }

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App startupFailure={startupFailure} />
    </StrictMode>,
  )

  if (!startupFailure) {
    requestAnimationFrame(() => {
      if (bootOverlayRoot) {
        bootOverlayRoot.unmount()
      }
      hideBootOverlay()
    })
  }
}

void renderApp()
