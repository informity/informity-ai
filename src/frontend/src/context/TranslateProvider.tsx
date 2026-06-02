import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import {
  cancelTranslateJob,
  createTranslateJob,
  estimateTranslateJob,
  getSettings,
  getTranslateJob,
  streamTranslateJob,
  type TranslateSection,
} from '../api'

const ACTIVE_JOB_KEY = 'informity_active_translate_job'

interface PersistedJob {
  jobId: string
  fileId: number
  fileName: string
  pageCount: number | null
  isUpload: boolean
  targetLanguage: string
  tone: string
}

function saveActiveJob(job: PersistedJob): void {
  try { sessionStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(job)) } catch { /* ignore */ }
}

function loadActiveJob(): PersistedJob | null {
  try {
    const raw = sessionStorage.getItem(ACTIVE_JOB_KEY)
    return raw ? (JSON.parse(raw) as PersistedJob) : null
  } catch { return null }
}

function clearActiveJob(): void {
  try { sessionStorage.removeItem(ACTIVE_JOB_KEY) } catch { /* ignore */ }
}
import { showToast } from './useToast'
import { TranslateContext, type TranslateContextValue } from './translateContext'
import { TRANSLATE_LANGUAGE_LABELS, TRANSLATE_TONES } from '../utils/translateOptions'

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
  const [retryingSectionIndex, setRetryingSectionIndex] = useState<number | null>(null)
  const [glossaryTermCount, setGlossaryTermCount] = useState<number | null>(null)
  const [estimatedMinutes, setEstimatedMinutes] = useState<number | null>(null)
  const [exceedsSoftLimit, setExceedsSoftLimit] = useState(false)

  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null)
  const [targetLanguage, setTargetLanguage] = useState('Spanish')
  const [tone, setTone] = useState('natural')

  const abortRef = useRef<AbortController | null>(null)

  // Load defaults from settings once on mount — lives here so navigation
  // away and back doesn't reset manual language/tone changes mid-session.
  useEffect(() => {
    getSettings().then((s) => {
      const settings = s as Record<string, unknown>
      const lang = settings?.translate_default_language as string | undefined
      if (lang && TRANSLATE_LANGUAGE_LABELS.includes(lang)) setTargetLanguage(lang)
      const t = settings?.translate_default_tone as string | undefined
      if (t && (TRANSLATE_TONES as readonly string[]).includes(t)) setTone(t)
    }).catch(() => {})
  }, [])

  // Recover active translation after a page reload.
  // Checks sessionStorage for a persisted job, verifies it is still running
  // server-side, and reconnects the SSE stream so the UI catches up.
  useEffect(() => {
    const persisted = loadActiveJob()
    if (!persisted) return

    const controller = new AbortController()
    abortRef.current = controller

    getTranslateJob(persisted.jobId).then((job) => {
      // Only recover jobs that have meaningful state: running, queued, or completed.
      // A null/unknown status means the job ID is stale — clear and bail.
      const recoverableStatuses = ['queued', 'running', 'done', 'stalled', 'failed']
      if (!recoverableStatuses.includes(job.status)) {
        clearActiveJob()
        return
      }

      // Restore file and translation settings so the chip and controls appear
      setFileInfo({
        id: persisted.fileId,
        name: persisted.fileName,
        pageCount: persisted.pageCount,
        isUpload: persisted.isUpload,
      })
      setTargetLanguage(persisted.targetLanguage)
      setTone(persisted.tone)
      setJobId(persisted.jobId)
      setJobStatus(job.status as TranslateContextValue['jobStatus'])
      setSections([])
      setSectionCount(null)
      setCompletedSections(0)
      setFailedSections(0)

      // Connect to SSE — for completed jobs the backend immediately replays
      // glossary_done, sections_ready, and all section_done events, then
      // emits job_done and closes. For active jobs it continues streaming.
      return streamTranslateJob(persisted.jobId, {
        signal: controller.signal,
        onGlossaryDone: (count) => setGlossaryTermCount(count),
        onSectionsReady: (count) => setSectionCount(count),
        onSectionStarted: () => setRetryingSectionIndex(null),
        onSectionRetry: (idx) => setRetryingSectionIndex(idx),
        onSectionDone: (section) => {
          let isNew = false
          setSections((prev) => {
            const idx = prev.findIndex((s) => s.section_index === section.section_index)
            isNew = idx < 0
            const next = [...prev]
            if (idx >= 0) next[idx] = section
            else next.push(section)
            next.sort((a, b) => a.section_index - b.section_index)
            return next
          })
          if (isNew) setCompletedSections((n) => n + 1)
          setRetryingSectionIndex(null)
        },
        onSectionFailed: () => setFailedSections((n) => n + 1),
        onJobDone: (completed, failed) => {
          setCompletedSections(completed)
          setFailedSections(failed)
          setJobStatus('done')
        },
        onJobFailed: (error) => {
          setJobStatus('failed')
          showToast('error', `Translation failed: ${error}`)
        },
        onJobStalled: () => {
          setJobStatus('stalled')
          showToast('warning', 'Translation stalled. You may retry.')
        },
      })
    }).catch(() => {
      clearActiveJob()
    })

    return () => controller.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
    setRetryingSectionIndex(null)
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

    // Persist so a page reload can reconnect to this job
    saveActiveJob({
      jobId: job_id,
      fileId: fileInfo.id,
      fileName: fileInfo.name,
      pageCount: fileInfo.pageCount,
      isUpload: fileInfo.isUpload,
      targetLanguage,
      tone,
    })

    try {
      setJobId(job_id)
      setJobStatus('running')

      await streamTranslateJob(job_id, {
        signal: controller.signal,
        onGlossaryDone: (count) => setGlossaryTermCount(count),
        onSectionsReady: (count) => setSectionCount(count),
        onSectionStarted: () => setRetryingSectionIndex(null),
        onSectionRetry: (sectionIndex) => setRetryingSectionIndex(sectionIndex),
        onSectionDone: (section) => {
          let isNew = false
          setSections((prev) => {
            const idx = prev.findIndex((s) => s.section_index === section.section_index)
            isNew = idx < 0
            const next = [...prev]
            if (idx >= 0) next[idx] = section
            else next.push(section)
            next.sort((a, b) => a.section_index - b.section_index)
            return next
          })
          // Only count sections not already present — prevents double-counting on SSE reconnect.
          if (isNew) setCompletedSections((n) => n + 1)
          setRetryingSectionIndex(null)
        },
        onSectionFailed: () => setFailedSections((n) => n + 1),
        onJobDone: (completed, failed) => {
          setCompletedSections(completed)
          setFailedSections(failed)
          setJobStatus('done')
        },
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
  }, [fileInfo, targetLanguage, tone])

  const cancelTranslation = useCallback(() => {
    abortRef.current?.abort()
    if (jobId) {
      showToast('info', 'Stopping translation…')
      cancelTranslateJob(jobId).catch(() => { /* best-effort */ })
    }
    clearActiveJob()
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
    retryingSectionIndex,
    glossaryTermCount, estimatedMinutes, exceedsSoftLimit,
    fileId: fileInfo?.id ?? null,
    fileName: fileInfo?.name ?? null,
    pageCount: fileInfo?.pageCount ?? null,
    isUpload: fileInfo?.isUpload ?? false,
    targetLanguage, tone,
    setFile, setTargetLanguage, setTone,
    startTranslation, cancelTranslation, clearResult,
    isTranslating, hasResult,
  }

  return <TranslateContext.Provider value={value}>{children}</TranslateContext.Provider>
}
