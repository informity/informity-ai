import { createContext } from 'react'
import type { TranslateSection, TranslateJobStatus } from '../api'

export interface TranslateContextValue {
  // Current job state
  jobId: string | null
  jobStatus: TranslateJobStatus['status'] | null
  sections: TranslateSection[]
  sectionCount: number | null
  completedSections: number
  failedSections: number
  retryingSectionIndex: number | null  // section index currently being retried, null otherwise
  glossaryTermCount: number | null
  estimatedMinutes: number | null
  exceedsSoftLimit: boolean

  // Selected file
  fileId: number | null
  fileName: string | null
  pageCount: number | null
  isUpload: boolean  // true = translate.local upload, false = library file

  // Settings
  targetLanguage: string
  tone: string

  // Actions
  setFile: (file: { id: number; name: string; pageCount: number | null; isUpload: boolean } | null) => void
  setTargetLanguage: (lang: string) => void
  setTone: (tone: string) => void
  startTranslation: () => Promise<void>
  cancelTranslation: () => void
  clearResult: () => void

  // Derived
  isTranslating: boolean
  hasResult: boolean
  smallModelWarning: boolean  // true when active model is likely < 10B parameters
}

export const TranslateContext = createContext<TranslateContextValue | null>(null)
