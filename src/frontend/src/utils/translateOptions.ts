/** Shared language and tone option lists for the Translate feature.
 *  Used by TranslatePage, TranslateProvider, and SettingsView. */

export interface LanguageOption {
  label: string
  countryCode: string
}

export const TRANSLATE_LANGUAGE_OPTIONS: LanguageOption[] = [
  { label: 'French',     countryCode: 'fr' },
  { label: 'German',     countryCode: 'de' },
  { label: 'Italian',    countryCode: 'it' },
  { label: 'Portuguese', countryCode: 'pt' },
  { label: 'Spanish',    countryCode: 'es' },
]

export const TRANSLATE_LANGUAGE_LABELS = TRANSLATE_LANGUAGE_OPTIONS.map(l => l.label)

export const TRANSLATE_TONES = ['natural', 'formal', 'literal'] as const
export type TranslateTone = typeof TRANSLATE_TONES[number]

export const TRANSLATE_TONE_ICONS: Record<TranslateTone, string> = {
  natural: 'ri-leaf-line',
  formal:  'ri-building-line',
  literal: 'ri-box-1-line',
}
