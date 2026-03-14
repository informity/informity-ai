import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_STORAGE_KEY } from './utils/uiTheme'

// Apply saved ui_theme (accent color) before first paint
try {
  const saved = localStorage.getItem(UI_THEME_STORAGE_KEY)
  const normalized = normalizeUiTheme(saved) ?? UI_THEME_DEFAULT
  document.documentElement.setAttribute('data-accent', normalized)
} catch {
  // ignore
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
