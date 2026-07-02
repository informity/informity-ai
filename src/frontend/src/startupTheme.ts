import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_STORAGE_KEY } from './utils/uiTheme'

function applyStartupTheme(theme: string | null | undefined) {
  const normalized = normalizeUiTheme(theme) ?? UI_THEME_DEFAULT
  document.documentElement.setAttribute('data-accent', normalized)
  try {
    localStorage.setItem(UI_THEME_STORAGE_KEY, normalized)
  } catch {
    // ignore
  }
}

try {
  const saved = localStorage.getItem(UI_THEME_STORAGE_KEY)
  applyStartupTheme(saved)
} catch {
  applyStartupTheme(UI_THEME_DEFAULT)
}
