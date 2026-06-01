import { useCallback, useRef, useState, type ReactNode } from 'react'
import {
  cancelTranslateJob,
  createTranslateJob,
  estimateTranslateJob,
  streamTranslateJob,
  type TranslateSection,
} from '../api'
import { showToast } from './useToast'
import { TranslateContext, type TranslateContextValue } from './translateContext'

interface FileInfo {
  id: number
  name: string
  pageCount: number | null
  isUpload: boolean
}

export function TranslateProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<TranslateContextValue['jobStatus']>(null)
  const [sections, setSections] = useState<TranslateSection[]>([])
  const [sectionCount, setSectionCount] = useState<number | null>(null)
  const [completedSections, setCompletedSections] = useState(0)
  const [failedSections, setFailedSections] = useState(0)
  const [glossaryTermCount, setGlossaryTermCount] = useState<number | null>(null)
  const [estimatedMinutes, setEstimatedMinutes] = useState<number | null>(null)
  const [exceedsSoftLimit, setExceedsSoftLimit] = useState(false)

  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null)
  const [targetLanguage, setTargetLanguage] = useState('Spanish')
  const [tone, setTone] = useState('natural')
  const [outputMode, setOutputMode] = useState<'markdown' | 'text'>('markdown')

  const abortRef = useRef<AbortController | null>(null)

  const setFile = useCallback(async (file: FileInfo | null) => {
    setFileInfo(file)
    setSections([])
    setSectionCount(null)
    setJobId(null)
    setJobStatus(null)
    setCompletedSections(0)
    setFailedSections(0)
    setGlossaryTermCount(null)
    setEstimatedMinutes(null)
    setExceedsSoftLimit(false)
    if (file) {
      try {
        const est = await estimateTranslateJob(file.id)
        setEstimatedMinutes(est.estimated_minutes)
        setExceedsSoftLimit(est.exceeds_soft_limit)
      } catch { /* non-critical — estimate is best-effort */ }
    }
  }, [])

  const startTranslation = useCallback(async () => {
    if (!fileInfo) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setSections([])
    setSectionCount(null)
    setCompletedSections(0)
    setFailedSections(0)
    setGlossaryTermCount(null)
    setJobStatus('queued')

    // Retry job creation with backoff — the LLM lock takes ~100-500ms to
    // release after a cancel, so a 409 immediately after Stop is expected.
    let job_id: string | undefined
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        const res = await createTranslateJob({
          file_id: fileInfo.id,
          target_language: targetLanguage,
          tone,
          output_mode: outputMode,
        })
        job_id = res.job_id
        break
      } catch (err: unknown) {
        const status = (err as { status?: number })?.status
        if (status === 409 && attempt < 4) {
          await new Promise(r => setTimeout(r, 400 * (attempt + 1)))
          continue
        }
        throw err
      }
    }
    if (!job_id) return

    try {
      setJobId(job_id)
      setJobStatus('running')

      await streamTranslateJob(job_id, {
        signal: controller.signal,
        onGlossaryDone: (count) => setGlossaryTermCount(count),
        onSectionsReady: (count) => setSectionCount(count),
        onSectionStarted: () => { /* progress tracked via onSectionDone */ },
        onSectionDone: (section) => {
          setSections((prev) => {
            const next = [...prev]
            const idx = next.findIndex((s) => s.section_index === section.section_index)
            if (idx >= 0) next[idx] = section
            else next.push(section)
            next.sort((a, b) => a.section_index - b.section_index)
            return next
          })
          setCompletedSections((n) => n + 1)
        },
        onSectionFailed: () => setFailedSections((n) => n + 1),
        onJobDone: () => setJobStatus('done'),
        onJobFailed: (error) => {
          setJobStatus('failed')
          showToast('error', `Translation failed: ${error}`)
        },
        onJobStalled: () => {
          setJobStatus('stalled')
          showToast('warning', 'Translation stalled. You may retry.')
        },
      })
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === 'AbortError') return
      const msg = (err as { message?: string })?.message || 'Translation failed'
      setJobStatus('failed')
      showToast('error', msg)
    }
  }, [fileInfo, targetLanguage, tone, outputMode])

  const cancelTranslation = useCallback(() => {
    abortRef.current?.abort()
    if (jobId) {
      showToast('info', 'Stopping translation…')
      cancelTranslateJob(jobId).catch(() => { /* best-effort */ })
    }
    setJobStatus(null)
  }, [jobId])

  const clearResult = useCallback(() => {
    abortRef.current?.abort()
    setJobId(null)
    setJobStatus(null)
    setSections([])
    setSectionCount(null)
    setCompletedSections(0)
    setFailedSections(0)
    setGlossaryTermCount(null)
  }, [])

  const isTranslating = jobStatus === 'queued' || jobStatus === 'running'
  const hasResult = sections.length > 0

  const value: TranslateContextValue = {
    jobId, jobStatus, sections, sectionCount, completedSections, failedSections,
    glossaryTermCount, estimatedMinutes, exceedsSoftLimit,
    fileId: fileInfo?.id ?? null,
    fileName: fileInfo?.name ?? null,
    pageCount: fileInfo?.pageCount ?? null,
    isUpload: fileInfo?.isUpload ?? false,
    targetLanguage, tone, outputMode,
    setFile, setTargetLanguage, setTone, setOutputMode,
    startTranslation, cancelTranslation, clearResult,
    isTranslating, hasResult,
  }

  return <TranslateContext.Provider value={value}>{children}</TranslateContext.Provider>
}
