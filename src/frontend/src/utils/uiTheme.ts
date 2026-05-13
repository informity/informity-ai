export const UI_THEME_STORAGE_KEY = 'informity-ui-theme'
export const UI_THEME_DEFAULT = 'onyx'

export const UI_THEME_VALUES = [
  'canvas',
  'ember',
  'sage',
  'onyx',
  'graphite',
] as const

export type UiThemeValue = (typeof UI_THEME_VALUES)[number]

export const UI_THEME_OPTIONS: Array<{ value: UiThemeValue; label: string }> = [
  { value: 'onyx', label: 'Onyx' },
  { value: 'graphite', label: 'Graphite' },
  { value: 'ember', label: 'Ember' },
  { value: 'sage', label: 'Sage' },
  { value: 'canvas', label: 'Canvas' },
]

export function normalizeUiTheme(theme: string | null | undefined): UiThemeValue | undefined {
  if (!theme) return undefined
  const normalized = theme.trim().toLowerCase()
  const aliasMap: Record<string, UiThemeValue> = {
    light: 'canvas',
    sand: 'canvas',
    'linen-dark': 'ember',
    linen: 'ember',
    mono: 'onyx',
    gray: 'graphite',
    overcast: 'graphite',
    purple: 'graphite',
    blue: 'graphite',
    green: 'graphite',
    orange: 'graphite',
  }
  const canonical = aliasMap[normalized] ?? normalized
  return UI_THEME_VALUES.includes(canonical as UiThemeValue)
    ? (canonical as UiThemeValue)
    : undefined
}
