export const UI_THEME_STORAGE_KEY = 'informity-ui-theme'
export const UI_THEME_DEFAULT = 'mono'

export const UI_THEME_VALUES = [
  'sand',
  'linen',
  'mono',
  'graphite',
] as const

export type UiThemeValue = (typeof UI_THEME_VALUES)[number]

export const UI_THEME_OPTIONS: Array<{ value: UiThemeValue; label: string }> = [
  { value: 'mono', label: 'Mono' },
  { value: 'graphite', label: 'Graphite' },
  { value: 'linen', label: 'Linen' },
  { value: 'sand', label: 'Sand' },
]

export function normalizeUiTheme(theme: string | null | undefined): UiThemeValue | undefined {
  if (!theme) return undefined
  const normalized = theme.trim().toLowerCase()
  const aliasMap: Record<string, UiThemeValue> = {
    light: 'sand',
    'linen-dark': 'linen',
    gray: 'graphite',
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
