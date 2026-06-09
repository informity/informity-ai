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
import { showToast } from './useToast'
import { TranslateContext, type TranslateContextValue } from './translateContext'
import {
  TRANSLATE_DEFAULT_LANGUAGE,
  TRANSLATE_DEFAULT_TONE,
  TRANSLATE_TONES,
  normalizeTranslateLanguage,
  normalizeTranslateLanguageList,
} from '../utils/translateOptions'

const ACTIVE_JOB_KEY = 'informity_active_translate_job'
const COMPLETED_RUNS_KEY = 'informity_completed_translate_runs'

interface PersistedJob {
  jobId: string
  fileId: number
  fileName: string
  pageCount: number | null
  isUpload: boolean
  targetLanguage: string
  tone: string
  startedAt: number
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

function clearCompletedRuns(): void {
  try { sessionStorage.removeItem(COMPLETED_RUNS_KEY) } catch { /* ignore */ }
}

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
  const [lastCompletedElapsedSeconds, setLastCompletedElapsedSeconds] = useState<number | null>(null)

  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null)
  const [targetLanguage, setTargetLanguage] = useState(TRANSLATE_DEFAULT_LANGUAGE)
  const [tone, setTone] = useState(TRANSLATE_DEFAULT_TONE)
  const [resultLanguage, setResultLanguage] = useState<string | null>(null)
  const [resultTone, setResultTone] = useState<string | null>(null)
  const [pinnedLanguages, setPinnedLanguages] = useState<string[]>([])

  const abortRef = useRef<AbortController | null>(null)
  const jobStartedAtRef = useRef<number>(0)
  const defaultsRef = useRef({
    language: TRANSLATE_DEFAULT_LANGUAGE,
    tone: TRANSLATE_DEFAULT_TONE,
    pinnedLanguages: [] as string[],
  })

  const applyTranslateDefaults = useCallback((settings: Record<string, unknown>) => {
    const lang = settings.translate_default_language as string | undefined
    const resolvedLanguage = normalizeTranslateLanguage(lang)
    const t = settings.translate_default_tone as string | undefined
    const resolvedTone = t && (TRANSLATE_TONES as readonly string[]).includes(t) ? t : TRANSLATE_DEFAULT_TONE
    const resolvedPinnedLanguages = normalizeTranslateLanguageList(
      [resolvedLanguage, ...(Array.isArray(settings.translate_pinned_languages) ? settings.translate_pinned_languages : [])],
      null,
    )
    defaultsRef.current = {
      language: resolvedLanguage,
      tone: resolvedTone,
      pinnedLanguages: resolvedPinnedLanguages,
    }
    setTargetLanguage(resolvedLanguage)
    setTone(resolvedTone)
    setPinnedLanguages(resolvedPinnedLanguages)
  }, [])

  // Load defaults from settings once on mount — lives here so navigation
  // away and back doesn't reset manual language/tone changes mid-session.
  useEffect(() => {
    getSettings()
      .then((s) => applyTranslateDefaults(s as Record<string, unknown>))
      .catch(() => {})
  }, [applyTranslateDefaults])

  useEffect(() => {
    if (pinnedLanguages.length > 0 && pinnedLanguages.includes(targetLanguage)) return
    if (targetLanguage === defaultsRef.current.language) return
    const fallbackLanguage = pinnedLanguages[0] || defaultsRef.current.language
    if (fallbackLanguage && fallbackLanguage !== targetLanguage) {
      setTargetLanguage(fallbackLanguage)
    }
  }, [pinnedLanguages, targetLanguage])

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
      const recoverableStatuses = ['queued', 'running', 'stalled']
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
      setResultLanguage(persisted.targetLanguage)
      setResultTone(persisted.tone)
      jobStartedAtRef.current = Number.isFinite(persisted.startedAt) && persisted.startedAt > 0
        ? persisted.startedAt
        : Date.now()
      setJobId(persisted.jobId)
      setJobStatus(job.status as TranslateContextValue['jobStatus'])
      setSections([])
      setSectionCount(null)
      setCompletedSections(0)
      setFailedSections(0)
      setLastCompletedElapsedSeconds(null)

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
        onJobDone: (completed, failed, elapsedSeconds) => {
          setCompletedSections(completed)
          setFailedSections(failed)
          const computedElapsed = (
            jobStartedAtRef.current > 0
              ? Math.max(0, Math.round((Date.now() - jobStartedAtRef.current) / 1000))
              : null
          )
          setLastCompletedElapsedSeconds(elapsedSeconds ?? computedElapsed)
          setJobStatus('done')
          jobStartedAtRef.current = 0
          clearActiveJob()
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

  const resetTranslationDefaults = useCallback(async () => {
    try {
      const settings = (await getSettings()) as Record<string, unknown>
      applyTranslateDefaults(settings)
    } catch {
      setTargetLanguage(defaultsRef.current.language)
      setTone(defaultsRef.current.tone)
      setPinnedLanguages(defaultsRef.current.pinnedLanguages)
    }
  }, [applyTranslateDefaults])

  const resetTranslationSession = useCallback(() => {
    abortRef.current?.abort()
    clearActiveJob()
    clearCompletedRuns()
    setFileInfo(null)
    setJobId(null)
    setJobStatus(null)
    setSections([])
    setSectionCount(null)
    setCompletedSections(0)
    setFailedSections(0)
    setRetryingSectionIndex(null)
    setGlossaryTermCount(null)
    setEstimatedMinutes(null)
    setExceedsSoftLimit(false)
    setLastCompletedElapsedSeconds(null)
    setResultLanguage(null)
    setResultTone(null)
    jobStartedAtRef.current = 0
  }, [])

  const startTranslation = useCallback(async () => {
    if (!fileInfo) return false
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    jobStartedAtRef.current = Date.now()
    let didComplete = false

    setSections([])
    setSectionCount(null)
    setCompletedSections(0)
    setFailedSections(0)
    setRetryingSectionIndex(null)
    setGlossaryTermCount(null)
    setJobStatus('queued')
    setLastCompletedElapsedSeconds(null)

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
      startedAt: jobStartedAtRef.current,
    })
    setResultLanguage(targetLanguage)
    setResultTone(tone)

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
        onJobDone: (completed, failed, elapsedSeconds) => {
          didComplete = true
          setCompletedSections(completed)
          setFailedSections(failed)
          setLastCompletedElapsedSeconds(elapsedSeconds ?? null)
          setJobStatus('done')
          clearActiveJob()
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
      if ((err as { name?: string })?.name === 'AbortError') return false
      const msg = (err as { message?: string })?.message || 'Translation failed'
      setJobStatus('failed')
      showToast('error', msg)
      return false
    }
    return didComplete
  }, [fileInfo, targetLanguage, tone])

  const cancelTranslation = useCallback(() => {
    abortRef.current?.abort()
    if (jobId) {
      showToast('info', 'Stopping translation…')
      cancelTranslateJob(jobId).catch(() => { /* best-effort */ })
    }
    clearActiveJob()
    setJobStatus(null)
    jobStartedAtRef.current = 0
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
    setLastCompletedElapsedSeconds(null)
    setResultLanguage(null)
    setResultTone(null)
    jobStartedAtRef.current = 0
  }, [])

  const isTranslating = jobStatus === 'queued' || jobStatus === 'running'
  const hasResult = sections.length > 0

  const value: TranslateContextValue = {
    jobId, jobStatus, sections, sectionCount, completedSections, failedSections,
    retryingSectionIndex,
    glossaryTermCount, estimatedMinutes, exceedsSoftLimit,
    lastCompletedElapsedSeconds,
    fileId: fileInfo?.id ?? null,
    fileName: fileInfo?.name ?? null,
    pageCount: fileInfo?.pageCount ?? null,
    isUpload: fileInfo?.isUpload ?? false,
    targetLanguage, tone,
    resultLanguage, resultTone,
    pinnedLanguages,
    setFile, setTargetLanguage, setTone,
    resetTranslationDefaults, resetTranslationSession,
    startTranslation, cancelTranslation, clearResult,
    isTranslating, hasResult,
  }

  return <TranslateContext.Provider value={value}>{children}</TranslateContext.Provider>
}
