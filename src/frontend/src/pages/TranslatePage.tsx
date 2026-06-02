import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { PageHeader } from '../components/PageHeader'
import { MarkdownContent } from '../components/MarkdownContent'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { useBackendStatus } from '../context/useBackendStatus'
import { useTranslateContext } from '../context/useTranslateContext'
import { useChatContext } from '../context/useChatContext'
import { uploadTranslateFile, deleteTranslateUpload } from '../api'
import { extractErrorMessage } from '../utils/errorMessages'
import { showToast } from '../context/useToast'
import { resizeComposerTextarea, applyComposerScopedPadding } from '../utils/composerSizing'
import { markdownToPlainText, downloadTextFile, downloadMarkdownFile } from '../utils/downloadHelpers'
import {
  TRANSLATE_LANGUAGE_OPTIONS,
  TRANSLATE_TONES,
  TRANSLATE_TONE_ICONS,
  type TranslateTone as Tone,
} from '../utils/translateOptions'
import type { TranslateSection } from '../api'
import './TranslatePage.css'

function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

// Keep local aliases so the rest of the file can use short names
const LANGUAGE_OPTIONS = TRANSLATE_LANGUAGE_OPTIONS
const TONES = TRANSLATE_TONES
const TONE_ICONS = TRANSLATE_TONE_ICONS

interface RunRecord {
  sections: TranslateSection[]
  language: string
  tone: Tone
  completedAt: number | null
  totalSections: number | null
  elapsedSeconds: number | null
  fileLabel: string
}

export function TranslatePage() {
  const { offline } = useBackendStatus()
  const { isStreaming, stopStreaming } = useChatContext()
  const location = useLocation()
  const {
    fileId, fileName, pageCount, isUpload, estimatedMinutes, exceedsSoftLimit,
    targetLanguage, tone, jobStatus, sections, sectionCount, completedSections,
    retryingSectionIndex,
    isTranslating, hasResult,
    setFile, setTargetLanguage, setTone, startTranslation, cancelTranslation, clearResult,
  } = useTranslateContext()

  // Runs accumulate above composer
  const [runs, setRuns] = useState<RunRecord[]>([])
  const activeRunRef = useRef<RunRecord | null>(null)
  const runStartRef = useRef<number>(0)
  const [exportMenuRun, setExportMenuRun] = useState<number | null>(null)

  // Composer state — matches ChatView
  const [animateToDocked, setAnimateToDocked] = useState(false)
  const wasDocked = useRef(false)
  const isCentered = !isTranslating && !hasResult && runs.length === 0

  // Menu state
  const [menuOpen, setMenuOpen] = useState<'language' | 'tone' | null>(null)
  const langMenuRef = useRef<HTMLDivElement>(null)
  const toneMenuRef = useRef<HTMLDivElement>(null)

  // Upload state
  const [uploadLoading, setUploadLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chipRef = useRef<HTMLSpanElement | null>(null)
  const pendingChipRef = useRef<HTMLSpanElement | null>(null)
  const inputWrapperRef = useRef<HTMLDivElement>(null)
  const resultsEndRef = useRef<HTMLDivElement>(null)
  const resultsContainerRef = useRef<HTMLDivElement>(null)

  const selectedLang = LANGUAGE_OPTIONS.find(l => l.label === targetLanguage) ?? LANGUAGE_OPTIONS.find(l => l.label === 'Spanish')!
  const canTranslate = !!fileId && !isTranslating && !isStreaming  // button morphs to Stop when streaming/translating

  // Load default language from settings

  // Pre-load file from Files page router state
  useEffect(() => {
    const state = location.state as { scopedFileId?: number; scopedFileName?: string } | null
    if (state?.scopedFileId && state.scopedFileName) {
      setFile({ id: state.scopedFileId, name: state.scopedFileName, pageCount: null, isUpload: false })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Centered → docked transition (mirrors ChatView logic)
  useEffect(() => {
    if (!isCentered && (isTranslating || hasResult || runs.length > 0)) {
      if (!wasDocked.current) {
        setAnimateToDocked(true)
        const t = setTimeout(() => setAnimateToDocked(false), 1200)
        wasDocked.current = true
        return () => clearTimeout(t)
      }
    }
    if (isCentered) wasDocked.current = false
  }, [isCentered, isTranslating, hasResult, runs.length])

  // Resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    resizeComposerTextarea(ta, !!fileId)
  }, [fileId])

  // useLayoutEffect fires synchronously after DOM commit, before paint —
  // ensures chip height is measured before the browser renders the frame.
  useLayoutEffect(() => {
    const activeChip = chipRef.current ?? pendingChipRef.current
    applyComposerScopedPadding(inputWrapperRef.current, activeChip)
    const ta = textareaRef.current
    if (ta) resizeComposerTextarea(ta, !!(fileId || uploadLoading))
  }, [fileId, uploadLoading])

  // Close menus on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (langMenuRef.current && !langMenuRef.current.contains(e.target as Node)) {
        setMenuOpen(prev => prev === 'language' ? null : prev)
      }
      if (toneMenuRef.current && !toneMenuRef.current.contains(e.target as Node)) {
        setMenuOpen(prev => prev === 'tone' ? null : prev)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // On mount: if the context already has sections (user navigated away and back),
  // reconstruct the run display. TranslateProvider keeps sections across navigation;
  // TranslatePage local state resets on unmount, so runs would be [] without this.
  // Must be declared before the [isTranslating] effect so it fires first — that way
  // activeRunRef is already set and the [isTranslating] effect sees it and skips.
  useEffect(() => {
    if (sections.length === 0 || runs.length > 0) return
    const run: RunRecord = {
      sections: [...sections],
      language: targetLanguage,
      tone: tone as Tone,
      completedAt: isTranslating ? null : Date.now(),
      totalSections: sectionCount,
      elapsedSeconds: null,
      fileLabel: fileName ?? 'Document',
    }
    if (isTranslating) {
      activeRunRef.current = run
      runStartRef.current = Date.now()
    }
    setRuns([run])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Create run entry IMMEDIATELY when translation starts — before sections arrive —
  // so the results area shows a progress indicator from the first moment.
  useEffect(() => {
    if (!isTranslating) return
    if (activeRunRef.current) return  // already created for this run
    const run: RunRecord = {
      sections: [], language: targetLanguage, tone: tone as Tone,
      completedAt: null, totalSections: sectionCount,
      elapsedSeconds: null, fileLabel: fileName ?? 'Document',
    }
    activeRunRef.current = run
    runStartRef.current = Date.now()
    setRuns(prev => [...prev, run])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTranslating])

  // Update active run as sections stream in.
  // Also handles the reload-recovery case: if sections arrive from SSE replay
  // but no run entry exists yet (activeRunRef is null and runs is empty),
  // create one now so the footer and content display correctly.
  useEffect(() => {
    if (sections.length === 0) return
    if (activeRunRef.current) {
      activeRunRef.current.sections = [...sections]
      activeRunRef.current.totalSections = sectionCount
      setRuns(prev => [...prev])
    } else if (runs.length === 0) {
      const run: RunRecord = {
        sections: [...sections],
        language: targetLanguage,
        tone: tone as Tone,
        completedAt: null,  // set by [jobStatus] effect when job_done arrives
        totalSections: sectionCount,
        elapsedSeconds: null,
        fileLabel: fileName ?? 'Document',
      }
      activeRunRef.current = run
      setRuns([run])
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sections, sectionCount])

  useEffect(() => {
    if (jobStatus === 'done' && activeRunRef.current && activeRunRef.current.completedAt === null) {
      activeRunRef.current.completedAt = Date.now()
      activeRunRef.current.elapsedSeconds = Math.round((Date.now() - runStartRef.current) / 1000)
      setRuns(prev => [...prev])
      activeRunRef.current = null
    }
  }, [jobStatus])

  // Scroll to bottom after run updates AND when job completes (footer appears).
  // useLayoutEffect fires after DOM commits so scrollHeight is accurate.
  useLayoutEffect(() => {
    const el = resultsContainerRef.current
    if (el && (isTranslating || jobStatus === 'done')) el.scrollTop = el.scrollHeight
  }, [runs, isTranslating, jobStatus])

  // Close export menu on outside click
  useEffect(() => {
    if (exportMenuRun === null) return
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest('.translate-page__export-trigger')) setExportMenuRun(null)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [exportMenuRun])

  const handleUpload = useCallback(async (file: File) => {
    setUploadLoading(true)
    try {
      const res = await uploadTranslateFile(file)
      setFile({ id: res.file_id, name: res.filename, pageCount: res.page_count ?? null, isUpload: true })
    } catch (err) {
      showToast('error', extractErrorMessage(err, 'Upload failed'))
    } finally { setUploadLoading(false) }
  }, [setFile])

  const handleDismiss = useCallback(async () => {
    if (fileId && isUpload) {
      try { await deleteTranslateUpload(fileId) } catch { /* best-effort */ }
    }
    setFile(null)
  }, [fileId, isUpload, setFile])

  const handleTranslate = useCallback(async () => {
    if (!fileId || isTranslating) return
    await startTranslation()
  }, [fileId, isTranslating, startTranslation])

  const handleNewTranslation = useCallback(async () => {
    activeRunRef.current = null
    if (isTranslating) cancelTranslation()
    if (fileId && isUpload) {
      try { await deleteTranslateUpload(fileId) } catch { /* best-effort */ }
    }
    setFile(null)
    clearResult()
    setRuns([])
    wasDocked.current = false
  }, [isTranslating, cancelTranslation, fileId, isUpload, setFile, clearResult])

  const handleCopyRun = useCallback((run: RunRecord) => {
    navigator.clipboard.writeText(run.sections.map(s => s.text).join('\n\n'))
      .then(() => showToast('success', 'Copied'))
  }, [])

  const handleSaveRun = useCallback((run: RunRecord, fmt: 'md' | 'txt') => {
    const markdown = run.sections.map(s => s.text).join('\n\n')
    const base = run.fileLabel.replace(/\.[^.]+$/, '')
    const lang = run.language.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    if (fmt === 'txt') {
      downloadTextFile(`${base}.${lang}.txt`, markdownToPlainText(markdown))
    } else {
      downloadMarkdownFile(`${base}.${lang}.md`, markdown)
    }
  }, [])

  const subtitle = 'Translate a document. Select from your indexed files or upload'

  if (offline) {
    return (
      <div className="page">
        <PageHeader title="Translate" subtitle={subtitle} icon="ri-translate-2" />
        <div className="page__scroll"><ServiceUnavailableState /></div>
      </div>
    )
  }

  const translatePlaceholder = isTranslating
    ? 'Translation in progress…'
    : isStreaming
      ? 'Chat is in progress. Please wait…'
      : fileId && hasResult
        ? 'Select a new file or upload to translate again.'
        : fileId
          ? 'Ready to translate. Press ⌘↵ or click Translate to start.'
          : 'Select or upload a document to translate…'

  return (
    <div className="translate-page">
      <PageHeader
        title="Translate"
        subtitle={subtitle}
        icon="ri-translate-2"
        action={
          <button
            type="button"
            className="translate-page__new-btn"
            onClick={() => void handleNewTranslation()}
            disabled={isTranslating}
            title="New Translation"
          >
            <i className="ri-translate-2" aria-hidden />
            New Translation
          </button>
        }
      />
      <div className={`translate-page__body${isCentered ? ' translate-page__body--centered' : ''}`}>

        {/* ── Result area (hidden when centered) ── */}
        <div ref={resultsContainerRef} className="translate-results">
          {runs.map((run, ri) => {
            const isActive = ri === runs.length - 1 && isTranslating
            const elapsedLabel = run.elapsedSeconds !== null
              ? `${Math.floor(run.elapsedSeconds / 60)}m ${run.elapsedSeconds % 60}s`
              : null
            const exportOpen = exportMenuRun === ri
            return (
              <div key={ri} className="translate-run">
                <div className="translate-run__sections">
                  {run.sections.map(s => (
                    <div key={s.section_index} className="translate-run__section">
                      <MarkdownContent>{s.text}</MarkdownContent>
                    </div>
                  ))}
                  {/* Streaming indicator — shows immediately when translation starts */}
                  {isActive && (
                    <div className="translate-run__section translate-run__section--streaming">
                      <span className="translate-run__cursor" aria-label="Translating…" />
                      <span className="translate-run__progress">
                        {retryingSectionIndex !== null
                          ? `Retrying section ${retryingSectionIndex + 1}…`
                          : sectionCount
                            ? `Section ${completedSections + 1} of ${sectionCount}`
                            : estimatedMinutes
                              ? `Preparing… Estimated time: ~${estimatedMinutes} min`
                              : 'Preparing…'}
                      </span>
                    </div>
                  )}
                </div>

                {/* Footer — feature/translate meta layout: metadata left, actions right */}
                {run.completedAt !== null && (
                  <div className="translate-page__meta">
                    <div className="translate-page__meta-left">
                      {elapsedLabel && (
                        <>
                          <div className="translate-page__meta-item">
                            <i className="ri-time-line translate-page__meta-icon" aria-hidden />
                            <span>{elapsedLabel}</span>
                          </div>
                          <span className="translate-page__meta-sep">|</span>
                        </>
                      )}
                      <div className="translate-page__meta-item">
                        <i className="ri-global-line translate-page__meta-icon" aria-hidden />
                        <span>{run.language}</span>
                      </div>
                      <span className="translate-page__meta-sep">|</span>
                      <div className="translate-page__meta-item">
                        <i className="ri-quill-pen-line translate-page__meta-icon" aria-hidden />
                        <span>{capitalize(run.tone)}</span>
                      </div>
                    </div>
                    <div className="translate-page__meta-right">
                      {/* Export dropdown — Markdown / Plain text */}
                      <div className="translate-page__export-trigger">
                        <button
                          type="button"
                          className="translate-page__meta-copy"
                          onClick={() => setExportMenuRun(exportOpen ? null : ri)}
                          title="Export"
                        >
                          <i className="ri-download-line" aria-hidden />
                        </button>
                        {exportOpen && (
                          <div className="translate-page__mode-menu translate-page__mode-menu--export" role="menu">
                            <button
                              type="button"
                              className="translate-page__mode-option"
                              onClick={() => { handleSaveRun(run, 'md'); setExportMenuRun(null) }}
                            >
                              <i className="ri-markdown-line" aria-hidden />
                              Markdown
                            </button>
                            <button
                              type="button"
                              className="translate-page__mode-option"
                              onClick={() => { handleSaveRun(run, 'txt'); setExportMenuRun(null) }}
                            >
                              <i className="ri-file-text-line" aria-hidden />
                              Plain text
                            </button>
                          </div>
                        )}
                      </div>
                      {/* Copy — always last */}
                      <button
                        type="button"
                        className="translate-page__meta-copy"
                        onClick={() => handleCopyRun(run)}
                        title="Copy translation"
                      >
                        <i className="ri-file-copy-line" aria-hidden />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
          <div ref={resultsEndRef} />
        </div>

        {/* ── Composer — identical structure to ChatView ── */}
        <div className={`translate-page__composer-wrap${animateToDocked ? ' translate-page__composer-wrap--docking' : ''}`}>

          {/* Warnings */}
          {exceedsSoftLimit && fileId && estimatedMinutes !== null && (
            <p className="translate-page__error">
              Large document (~{pageCount ?? '?'} pages) · ~{estimatedMinutes} min estimated
            </p>
          )}
          {/* isStreaming hint is now inline in the controls row — no warning box */}

          {/* Input wrapper — uses shared composer__ classes (globally loaded via index.css) */}
          <div
            ref={inputWrapperRef}
            className={`composer__input-wrapper${(fileId || uploadLoading) ? ' composer__input-wrapper--scoped' : ''}`}
          >
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              className="composer__file-input"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) { void handleUpload(f); e.target.value = '' }
              }}
            />

            {/* File chip — uses composer__scope-chip (globally loaded, same as Chat) */}
            {fileId && fileName && (
              <span ref={chipRef} className="composer__scope-chip" title={fileName}>
                <i className="ri-file-text-line" aria-hidden />
                <span>{fileName}{pageCount ? ` · ${pageCount}p` : ''}</span>
                <button
                  type="button"
                  className="composer__scope-clear"
                  aria-label="Remove file"
                  disabled={isTranslating}
                  onClick={() => void handleDismiss()}
                >
                  <i className="ri-close-line" aria-hidden />
                </button>
              </span>
            )}

            {/* Upload pending chip — spinner only, no text, matches Chat's pending upload chip */}
            {uploadLoading && !fileId && (
              <span ref={pendingChipRef} className="composer__scope-chip translate-page__pending-chip" aria-label="Uploading…">
                <i className="ri-loader-4-line translate-page__spinner" aria-hidden />
              </span>
            )}

            {/* Textarea — file drop zone and visual composer input */}
            <textarea
              ref={textareaRef}
              className={`composer__textarea${(fileId || uploadLoading) ? ' composer__textarea--scoped' : ''}`}
              rows={1}
              placeholder={translatePlaceholder}
              readOnly={!!fileId}
              disabled={isTranslating}
              onDragOver={(e) => { if (!fileId) e.preventDefault() }}
              onDrop={(e) => {
                if (!fileId) { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) void handleUpload(f) }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canTranslate) {
                  e.preventDefault(); void handleTranslate()
                }
              }}
            />

            {/* Controls row — uses shared composer__ classes */}
            <div className="composer__controls-row">

              {/* LEFT: + upload, tone picker */}
              <div className="composer__controls-left">

                {/* + upload button */}
                <button
                  type="button"
                  className="translate-page__icon-btn"
                  onClick={() => { if (!fileId) fileInputRef.current?.click() }}
                  disabled={!!fileId || uploadLoading || isTranslating}
                  aria-label="Upload file"
                >
                  <i className="ri-add-line" aria-hidden />
                </button>

                {/* Tone picker */}
                <div ref={toneMenuRef} className="translate-page__options-trigger">
                  <button
                    type="button"
                    className={`translate-page__icon-btn${menuOpen === 'tone' ? ' translate-page__icon-btn--active' : ''}`}
                    aria-haspopup="menu"
                    aria-expanded={menuOpen === 'tone'}
                    aria-label="Tone"
                    disabled={isTranslating}
                    onClick={() => setMenuOpen(prev => prev === 'tone' ? null : 'tone')}
                  >
                    <i className="ri-quill-pen-line" aria-hidden />
                  </button>
                  {menuOpen === 'tone' && (
                    <div className="translate-page__mode-menu translate-page__mode-menu--left" role="menu">
                      {TONES.map(t => (
                        <button
                          key={t}
                          type="button"
                          className={`translate-page__mode-option${tone === t ? ' translate-page__mode-option--active' : ''}`}
                          onClick={() => { setTone(t); setMenuOpen(null) }}
                        >
                          <i className={TONE_ICONS[t]} aria-hidden />
                          {capitalize(t)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* RIGHT: language selector, translate/stop button */}
              <div className="composer__controls-right">

                {/* Language selector — mirrors chat-view__mode-selector */}
                <div ref={langMenuRef} className="translate-page__mode-selector">
                  <button
                    type="button"
                    className="translate-page__mode-button"
                    aria-haspopup="menu"
                    aria-expanded={menuOpen === 'language'}
                    disabled={isTranslating}
                    onClick={() => setMenuOpen(prev => prev === 'language' ? null : 'language')}
                  >
                    <span className={`fi fi-${selectedLang.countryCode} translate-page__flag`} aria-hidden />
                    <span className="translate-page__mode-label">{targetLanguage}</span>
                    <i className="ri-arrow-down-s-line" aria-hidden />
                  </button>
                  {menuOpen === 'language' && (
                    <div className="translate-page__mode-menu" role="menu">
                      {LANGUAGE_OPTIONS.map(l => (
                        <button
                          key={l.label}
                          type="button"
                          className={`translate-page__mode-option${targetLanguage === l.label ? ' translate-page__mode-option--active' : ''}`}
                          onClick={() => { setTargetLanguage(l.label); setMenuOpen(null) }}
                        >
                          <span className={`fi fi-${l.countryCode} translate-page__flag`} aria-hidden />
                          <span>{l.label}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Send / Stop button */}
                {isTranslating ? (
                  <button
                    type="button"
                    className="translate-page__send"
                    onClick={cancelTranslation}
                    title="Stop Translation"
                    aria-label="Stop Translation"
                  >
                    <i className="ri-stop-large-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                  </button>
                ) : isStreaming ? (
                  <button
                    type="button"
                    className="translate-page__send"
                    onClick={() => void stopStreaming()}
                    title="Stop Chat to Translate"
                    aria-label="Stop Chat"
                    style={{ gap: '0.375rem', padding: '0.375rem 0.75rem' }}
                  >
                    <i className="ri-stop-large-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                    <span style={{ fontSize: 'var(--font-size-sm)' }}>Stop Chat</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="translate-page__send"
                    disabled={!canTranslate}
                    onClick={() => void handleTranslate()}
                    aria-label="Translate"
                  >
                    <i className="ri-translate-2" aria-hidden style={{ fontSize: '1.125rem' }} />
                  </button>
                )}
              </div>
            </div>
          </div>

          <p className="translate-page__disclaimer">
            Informity AI translator can make mistakes. Please double-check the original source.
          </p>
        </div>
      </div>
    </div>
  )
}
