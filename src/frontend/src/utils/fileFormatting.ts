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
  html: 'ri-code-s-line',
  htm: 'ri-code-s-line',
}

export function getFileIcon(extension: string | undefined): string {
  const ext = (extension || '').toLowerCase().replace(/^\./, '')
  return EXTENSION_ICONS[ext] || 'ri-file-line'
}

export function formatCategory(category: string | undefined): string {
  if (!category) return '—'
  return String(category).charAt(0).toUpperCase() + String(category).slice(1)
}

