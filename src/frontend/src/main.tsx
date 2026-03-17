import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { bootstrapDesktopBackend } from './tauriRuntime'
import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_STORAGE_KEY } from './utils/uiTheme'

// Apply saved ui_theme (accent color) before first paint
try {
  const saved = localStorage.getItem(UI_THEME_STORAGE_KEY)
  const normalized = normalizeUiTheme(saved) ?? UI_THEME_DEFAULT
  document.documentElement.setAttribute('data-accent', normalized)
} catch {
  // ignore
}

function setBootStatus(message: string) {
  const status = document.getElementById('boot-status')
  if (status) status.textContent = message
}

function hideBootOverlay() {
  const overlay = document.getElementById('boot-overlay')
  if (!overlay) return
  overlay.classList.add('boot-overlay--hidden')
  window.setTimeout(() => overlay.remove(), 220)
}

async function renderApp() {
  let startupError: string | null = null
  setBootStatus('Starting Informity AI...')
  const longStartTimerId = window.setTimeout(() => {
    setBootStatus('Still starting, this may take a moment...')
  }, 20000)

  try {
    await bootstrapDesktopBackend((message) => {
      setBootStatus(message)
    })
    window.clearTimeout(longStartTimerId)
    setBootStatus('Loading interface...')
  } catch (error) {
    window.clearTimeout(longStartTimerId)
    startupError = error instanceof Error ? error.message : String(error)
    setBootStatus('Startup failed. Rendering diagnostics...')
  }

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App startupError={startupError} />
    </StrictMode>,
  )

  requestAnimationFrame(() => {
    hideBootOverlay()
  })
}

void renderApp()
