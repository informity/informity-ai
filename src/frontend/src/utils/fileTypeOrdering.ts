export interface FileTypeOption {
  id: string
  label: string
  extensions: string[]
}

export const FILE_TYPE_DISPLAY_ORDER = ['pdf', 'docx', 'spreadsheet', 'pptx', 'epub', 'web', 'text', 'data'] as const

export function sortFileTypeOptions(options: FileTypeOption[]): FileTypeOption[] {
  return [...options].sort((a, b) => {
    const ai = FILE_TYPE_DISPLAY_ORDER.indexOf(a.id as (typeof FILE_TYPE_DISPLAY_ORDER)[number])
    const bi = FILE_TYPE_DISPLAY_ORDER.indexOf(b.id as (typeof FILE_TYPE_DISPLAY_ORDER)[number])
    const aRank = ai === -1 ? Number.MAX_SAFE_INTEGER : ai
    const bRank = bi === -1 ? Number.MAX_SAFE_INTEGER : bi
    if (aRank !== bRank) return aRank - bRank
    return a.label.localeCompare(b.label)
  })
}
