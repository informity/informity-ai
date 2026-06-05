/** Shared language and tone option lists for the Translate feature.
 *  Used by TranslatePage, TranslateProvider, and SettingsView. */

export const TRANSLATE_DEFAULT_LANGUAGE = 'Spanish'
export const TRANSLATE_DEFAULT_TONE = 'natural'

export interface TranslateLanguageOption {
  label: string
  qwenCode: string
  countryCode: string
  nativeLabel?: string
  aliases?: string[]
}

export const TRANSLATE_LANGUAGE_OPTIONS: TranslateLanguageOption[] = [
  { label: 'Arabic (Standard)', qwenCode: 'ar', countryCode: 'sa', nativeLabel: 'العربية', aliases: ['Arabic', 'Standard Arabic', 'ar'] },
  { label: 'Bengali', qwenCode: 'bn', countryCode: 'bd', nativeLabel: 'বাংলা', aliases: ['Bangla', 'bn'] },
  { label: 'Chinese', qwenCode: 'zh', countryCode: 'cn', nativeLabel: '中文', aliases: ['Chinese (Simplified)', 'Simplified Chinese', 'zh-cn', 'zh-hans', 'zh'] },
  { label: 'Czech', qwenCode: 'cs', countryCode: 'cz', nativeLabel: 'Čeština', aliases: ['cs'] },
  { label: 'Dutch', qwenCode: 'nl', countryCode: 'nl', nativeLabel: 'Nederlands', aliases: ['nl'] },
  { label: 'English', qwenCode: 'en', countryCode: 'gb', nativeLabel: 'English', aliases: ['British English', 'American English', 'en'] },
  { label: 'French', qwenCode: 'fr', countryCode: 'fr', nativeLabel: 'Français', aliases: ['fr'] },
  { label: 'German', qwenCode: 'de', countryCode: 'de', nativeLabel: 'Deutsch', aliases: ['de'] },
  { label: 'Greek', qwenCode: 'el', countryCode: 'gr', nativeLabel: 'Ελληνικά', aliases: ['el'] },
  { label: 'Hebrew', qwenCode: 'he', countryCode: 'il', nativeLabel: 'עברית', aliases: ['he'] },
  { label: 'Hindi', qwenCode: 'hi', countryCode: 'in', nativeLabel: 'हिन्दी', aliases: ['hi'] },
  { label: 'Indonesian', qwenCode: 'id', countryCode: 'id', nativeLabel: 'Bahasa Indonesia', aliases: ['id'] },
  { label: 'Italian', qwenCode: 'it', countryCode: 'it', nativeLabel: 'Italiano', aliases: ['it'] },
  { label: 'Japanese', qwenCode: 'ja', countryCode: 'jp', nativeLabel: '日本語', aliases: ['ja'] },
  { label: 'Korean', qwenCode: 'ko', countryCode: 'kr', nativeLabel: '한국어', aliases: ['ko'] },
  { label: 'Persian', qwenCode: 'fa', countryCode: 'ir', nativeLabel: 'فارسی', aliases: ['Farsi', 'fa'] },
  { label: 'Polish', qwenCode: 'pl', countryCode: 'pl', nativeLabel: 'Polski', aliases: ['pl'] },
  { label: 'Portuguese', qwenCode: 'pt', countryCode: 'pt', nativeLabel: 'Português', aliases: ['European Portuguese', 'Brazilian Portuguese', 'pt'] },
  { label: 'Romanian', qwenCode: 'ro', countryCode: 'ro', nativeLabel: 'Română', aliases: ['ro'] },
  { label: 'Russian', qwenCode: 'ru', countryCode: 'ru', nativeLabel: 'Русский', aliases: ['ru'] },
  { label: 'Spanish', qwenCode: 'es', countryCode: 'es', nativeLabel: 'Español', aliases: ['es'] },
  { label: 'Swedish', qwenCode: 'sv', countryCode: 'se', nativeLabel: 'Svenska', aliases: ['sv'] },
  { label: 'Thai', qwenCode: 'th', countryCode: 'th', nativeLabel: 'ไทย', aliases: ['th'] },
  { label: 'Turkish', qwenCode: 'tr', countryCode: 'tr', nativeLabel: 'Türkçe', aliases: ['tr'] },
  { label: 'Ukrainian', qwenCode: 'uk', countryCode: 'ua', nativeLabel: 'Українська', aliases: ['uk'] },
  { label: 'Vietnamese', qwenCode: 'vi', countryCode: 'vn', nativeLabel: 'Tiếng Việt', aliases: ['vi'] },
]

export const TRANSLATE_LANGUAGE_LABELS = TRANSLATE_LANGUAGE_OPTIONS.map((language) => language.label)
const TRANSLATE_LANGUAGE_ORDER = new Map(TRANSLATE_LANGUAGE_LABELS.map((label, index) => [label, index] as const))

const TRANSLATE_LANGUAGE_LOOKUP = new Map<string, TranslateLanguageOption>(
  TRANSLATE_LANGUAGE_OPTIONS.flatMap((language) => {
    const keys = new Set<string>([
      language.label,
      language.label.toLowerCase(),
      language.qwenCode,
      language.qwenCode.toLowerCase(),
      language.countryCode,
      language.countryCode.toLowerCase(),
      ...(language.nativeLabel ? [language.nativeLabel, language.nativeLabel.toLowerCase()] : []),
      ...(language.aliases || []),
      ...(language.aliases || []).map((alias) => alias.toLowerCase()),
    ])
    return Array.from(keys).map((key) => [key, language] as const)
  }),
)

export function findTranslateLanguageOption(value: string | null | undefined): TranslateLanguageOption | undefined {
  const text = String(value || '').trim()
  if (!text) return undefined
  return TRANSLATE_LANGUAGE_LOOKUP.get(text) ?? TRANSLATE_LANGUAGE_LOOKUP.get(text.toLowerCase())
}

export function normalizeTranslateLanguage(value: string | null | undefined): string {
  const text = String(value || '').trim()
  if (!text) return TRANSLATE_DEFAULT_LANGUAGE
  return findTranslateLanguageOption(text)?.label ?? TRANSLATE_DEFAULT_LANGUAGE
}

export function normalizeTranslateLanguageList(values: unknown): string[] {
  if (!Array.isArray(values)) return []
  const seen = new Set<string>()
  const normalized: string[] = []
  for (const raw of values) {
    const option = findTranslateLanguageOption(String(raw || ''))
    if (!option) continue
    const language = option.label
    if (!language || seen.has(language)) continue
    seen.add(language)
    normalized.push(language)
  }
  return normalized.sort((left, right) => {
    const leftIndex = TRANSLATE_LANGUAGE_ORDER.get(left) ?? Number.MAX_SAFE_INTEGER
    const rightIndex = TRANSLATE_LANGUAGE_ORDER.get(right) ?? Number.MAX_SAFE_INTEGER
    return leftIndex - rightIndex
  })
}

export function searchTranslateLanguages(query: string, limit = TRANSLATE_LANGUAGE_OPTIONS.length): TranslateLanguageOption[] {
  const needle = String(query || '').trim().toLowerCase()
  const scored = TRANSLATE_LANGUAGE_OPTIONS
    .map((language) => {
      const haystack = [
        language.label,
        language.nativeLabel || '',
        language.qwenCode,
        language.countryCode,
        ...(language.aliases || []),
      ].join(' ').toLowerCase()
      const starts = Boolean(needle && (
        language.label.toLowerCase().startsWith(needle)
        || (language.nativeLabel || '').toLowerCase().startsWith(needle)
        || language.qwenCode.toLowerCase().startsWith(needle)
        || language.countryCode.toLowerCase().startsWith(needle)
        || (language.aliases || []).some((alias) => alias.toLowerCase().startsWith(needle))
      ))
      const includes = !starts && needle ? haystack.includes(needle) : false
      return { language, starts, includes }
    })
    .filter(({ starts, includes }) => !needle || starts || includes)
    .sort((a, b) => {
      if (a.starts !== b.starts) return a.starts ? -1 : 1
      return a.language.label.localeCompare(b.language.label)
    })
    .slice(0, limit)
    .map(({ language }) => language)
  return scored
}

export const TRANSLATE_TONES = ['natural', 'formal', 'literal'] as const
export type TranslateTone = typeof TRANSLATE_TONES[number]

export const TRANSLATE_TONE_ICONS: Record<TranslateTone, string> = {
  natural: 'ri-leaf-line',
  formal:  'ri-building-line',
  literal: 'ri-box-1-line',
}
