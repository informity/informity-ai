const EXTENSION_ICONS: Record<string, string> = {
  pdf: 'ri-file-text-line',
  docx: 'ri-file-text-line',
  doc: 'ri-file-text-line',
  txt: 'ri-file-text-line',
  md: 'ri-file-text-line',
  rst: 'ri-file-text-line',
  log: 'ri-file-text-line',
  xlsx: 'ri-file-excel-2-line',
  xls: 'ri-file-excel-2-line',
  csv: 'ri-file-excel-2-line',
  pptx: 'ri-file-text-line',
  ppt: 'ri-file-text-line',
  epub: 'ri-file-text-line',
  html: 'ri-code-s-line',
  htm: 'ri-code-s-line',
}

const CATEGORY_LABELS: Record<string, string> = {
  document: 'Document',
  plaintext: 'Text',
  data: 'Data',
  web: 'Web',
  other: 'Other',
}

const EXTENSION_CATEGORY_LABEL_OVERRIDES: Record<string, string> = {
  epub: 'eBook',
  pptx: 'Presentation',
  xlsx: 'Spreadsheet',
}

export function getFileIcon(extension: string | undefined): string {
  const ext = (extension || '').toLowerCase().replace(/^\./, '')
  return EXTENSION_ICONS[ext] || 'ri-file-line'
}

export function formatCategory(category: string | undefined, extension?: string): string {
  if (!category) return '—'
  const ext = String(extension || '').toLowerCase().replace(/^\./, '')
  if (ext && EXTENSION_CATEGORY_LABEL_OVERRIDES[ext]) {
    return EXTENSION_CATEGORY_LABEL_OVERRIDES[ext]
  }
  const normalizedCategory = String(category).toLowerCase().trim()
  if (CATEGORY_LABELS[normalizedCategory]) {
    return CATEGORY_LABELS[normalizedCategory]
  }
  return String(category).charAt(0).toUpperCase() + String(category).slice(1)
}
