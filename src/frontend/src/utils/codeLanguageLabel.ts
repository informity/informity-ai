const LANGUAGE_LABEL_ALIASES: Record<string, string> = {
  ts: 'TypeScript',
  tsx: 'TypeScript',
  typescript: 'TypeScript',
  js: 'JavaScript',
  jsx: 'JavaScript',
  javascript: 'JavaScript',
  py: 'Python',
  sh: 'Shell',
  bash: 'Shell',
  zsh: 'Shell',
  yml: 'YAML',
  md: 'Markdown',
}

export function formatCodeLanguageLabel(language: string | null | undefined): string {
  const normalized = String(language || '').trim().toLowerCase()
  if (!normalized) return 'Code'
  if (normalized === 'code') return 'Code'
  const alias = LANGUAGE_LABEL_ALIASES[normalized]
  if (alias) return alias
  if (normalized.length <= 4) return normalized.toUpperCase()
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}
