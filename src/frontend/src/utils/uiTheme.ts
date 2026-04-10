export const UI_THEME_STORAGE_KEY = 'informity-ui-theme'
export const UI_THEME_DEFAULT = 'mono'

export const UI_THEME_VALUES = [
  'mono',
  'gray',
  'purple',
  'blue',
  'green',
  'orange',
] as const

export type UiThemeValue = (typeof UI_THEME_VALUES)[number]

export const UI_THEME_OPTIONS: Array<{ value: UiThemeValue; label: string }> = [
  { value: 'mono', label: 'Mono' },
  { value: 'gray', label: 'Gray' },
  { value: 'purple', label: 'Purple' },
  { value: 'blue', label: 'Blue' },
  { value: 'green', label: 'Green' },
  { value: 'orange', label: 'Orange' },
]

export function normalizeUiTheme(theme: string | null | undefined): UiThemeValue | undefined {
  if (!theme) return undefined
  const normalized = theme
  return UI_THEME_VALUES.includes(normalized as UiThemeValue)
    ? (normalized as UiThemeValue)
    : undefined
}
