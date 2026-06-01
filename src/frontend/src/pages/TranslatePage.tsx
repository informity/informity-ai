import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { PageHeader } from '../components/PageHeader'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { useBackendStatus } from '../context/useBackendStatus'
import { useTranslateContext } from '../context/useTranslateContext'
import { useChatContext } from '../context/useChatContext'
import { uploadTranslateFile, deleteTranslateUpload, getSettings } from '../api'
import { extractErrorMessage } from '../utils/errorMessages'
import { showToast } from '../context/useToast'
import { resizeComposerTextarea, applyComposerScopedPadding } from '../utils/composerSizing'
import type { TranslateSection } from '../api'
import './TranslatePage.css'

interface LanguageOption { label: string; countryCode: string }

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { label: 'French',     countryCode: 'fr' },
  { label: 'German',     countryCode: 'de' },
  { label: 'Italian',    countryCode: 'it' },
  { label: 'Portuguese', countryCode: 'pt' },
  { label: 'Spanish',    countryCode: 'es' },
]

const TONES = ['natural', 'formal', 'literal'] as const
type Tone = typeof TONES[number]

interface RunRecord {
  sections: TranslateSection[]
  language: string
  tone: Tone
  steering: string
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
    isTranslating, hasResult,
    setFile, setTargetLanguage, setTone, startTranslation, cancelTranslation, clearResult,
  } = useTranslateContext()

  // Runs accumulate above composer
  const [runs, setRuns] = useState<RunRecord[]>([])
  const activeRunRef = useRef<RunRecord | null>(null)
  const runStartRef = useRef<number>(0)

  // Composer state — matches ChatView
  const [animateToDocked, setAnimateToDocked] = useState(false)
  const wasDocked = useRef(false)
  const isCentered = !isTranslating && !hasResult && runs.length === 0

  // Steering prompt
  const [steering, setSteering] = useState('')

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

  const selectedLang = LANGUAGE_OPTIONS.find(l => l.label === targetLanguage) ?? LANGUAGE_OPTIONS.find(l => l.label === 'Spanish')!
  const canTranslate = !!fileId && !isTranslating && !isStreaming

  // Load default language from settings
  useEffect(() => {
    getSettings().then((s) => {
      const lang = (s as Record<string, unknown>)?.translate_default_language as string | undefined
      if (lang && LANGUAGE_OPTIONS.some(l => l.label === lang)) setTargetLanguage(lang)
    }).catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
  }, [steering, fileId])

  useEffect(() => {
    // Use whichever chip is visible — file chip or uploading pending chip
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

  // Accumulate sections into active run
  useEffect(() => {
    if (!isTranslating && !hasResult) return
    if (sections.length === 0) return
    if (!activeRunRef.current && isTranslating) {
      const run: RunRecord = {
        sections: [], language: targetLanguage, tone: tone as Tone,
        steering, completedAt: null, totalSections: sectionCount,
        elapsedSeconds: null, fileLabel: fileName ?? 'Document',
      }
      activeRunRef.current = run
      runStartRef.current = Date.now()
      setRuns(prev => [...prev, run])
    }
    if (activeRunRef.current) {
      activeRunRef.current.sections = [...sections]
      activeRunRef.current.totalSections = sectionCount
      setRuns(prev => [...prev])
    }
  }, [sections, sectionCount, isTranslating, hasResult, targetLanguage, tone, steering, fileName])

  useEffect(() => {
    if (jobStatus === 'done' && activeRunRef.current && activeRunRef.current.completedAt === null) {
      activeRunRef.current.completedAt = Date.now()
      activeRunRef.current.elapsedSeconds = Math.round((Date.now() - runStartRef.current) / 1000)
      setRuns(prev => [...prev])
      activeRunRef.current = null
    }
  }, [jobStatus])

  useEffect(() => {
    if (isTranslating) resultsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sections.length, isTranslating])

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
    if (isTranslating) cancelTranslation()
    if (fileId && isUpload) {
      try { await deleteTranslateUpload(fileId) } catch { /* best-effort */ }
    }
    setFile(null)
    clearResult()
    setRuns([])
    setSteering('')
    setTargetLanguage(LANGUAGE_OPTIONS.find(l => l.label === 'Spanish')?.label ?? LANGUAGE_OPTIONS[0].label)
    setTone('natural')
    wasDocked.current = false
  }, [isTranslating, cancelTranslation, fileId, isUpload, setFile, clearResult, setTargetLanguage, setTone])

  const handleCopyRun = useCallback((run: RunRecord) => {
    navigator.clipboard.writeText(run.sections.map(s => s.text).join('\n\n'))
      .then(() => showToast('success', 'Copied'))
  }, [])

  const handleSaveRun = useCallback((run: RunRecord, fmt: 'md' | 'txt') => {
    let content = run.sections.map(s => s.text).join('\n\n')
    if (fmt === 'txt') {
      content = content
        .replace(/^#{1,6}\s+/gm, '').replace(/\*\*(.+?)\*\*/gs, '$1')
        .replace(/[*_]{1,2}(.+?)[*_]{1,2}/gs, '$1')
    }
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const base = run.fileLabel.replace(/\.[^.]+$/, '')
    const lang = run.language.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    a.href = url; a.download = `${base}.${lang}.${fmt}`
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url)
  }, [])

  const subtitle = 'Translate a document. Select from your indexed files or upload.'

  if (offline) {
    return (
      <div className="page page--translate">
        <PageHeader title="Translate" subtitle={subtitle} icon="ri-translate-2" />
        <div className="page__scroll"><ServiceUnavailableState /></div>
      </div>
    )
  }

  const translatePlaceholder = fileId
    ? 'Steer the translation (optional) — e.g. focus on methodology, skip references…'
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
            title="New translation"
          >
            <i className="ri-translate-2" aria-hidden />
            New Translation
          </button>
        }
      />
      <div className={`translate-page__body${isCentered ? ' translate-page__body--centered' : ''}`}>

        {/* ── Result area (hidden when centered) ── */}
        <div className="translate-results">
          {runs.map((run, ri) => (
            <div key={ri} className="translate-run">
              <div className="translate-run__header">
                <span className="translate-run__label">
                  {run.fileLabel} → {run.language}
                  {run.tone !== 'natural' && ` · ${run.tone}`}
                  {run.steering && ` · "${run.steering.slice(0, 50)}${run.steering.length > 50 ? '…' : ''}"`}
                </span>
              </div>
              <div className="translate-run__sections">
                {run.sections.map(s => (
                  <div key={s.section_index} className="translate-run__section">
                    <ReactMarkdown>{s.text}</ReactMarkdown>
                  </div>
                ))}
                {ri === runs.length - 1 && isTranslating && (
                  <div className="translate-run__section translate-run__section--streaming">
                    <span className="translate-run__cursor" aria-label="Translating…" />
                    <span className="translate-run__progress">
                      {sectionCount ? `Section ${completedSections + 1} of ${sectionCount}` : 'Translating…'}
                    </span>
                  </div>
                )}
              </div>
              {run.completedAt !== null && (
                <div className="translate-run__footer">
                  <div className="translate-run__footer-actions">
                    <button type="button" className="translate-run__action" onClick={() => handleCopyRun(run)}>
                      <i className="ri-clipboard-line" aria-hidden /> Copy
                    </button>
                    <button type="button" className="translate-run__action" onClick={() => handleSaveRun(run, 'md')}>
                      <i className="ri-download-line" aria-hidden /> .md
                    </button>
                    <button type="button" className="translate-run__action" onClick={() => handleSaveRun(run, 'txt')}>
                      <i className="ri-download-line" aria-hidden /> .txt
                    </button>
                  </div>
                  <span className="translate-run__footer-meta">
                    {run.language} · {run.tone}
                    {run.elapsedSeconds !== null && ` · ${Math.floor(run.elapsedSeconds / 60)}m ${run.elapsedSeconds % 60}s`}
                  </span>
                </div>
              )}
            </div>
          ))}
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
          {isStreaming && (
            <p className="translate-page__error">
              LLM in use by chat.{' '}
              <button type="button" className="translate-page__stop-chat" onClick={() => void stopStreaming()}>
                Stop Chat
              </button>
            </p>
          )}

          {/* Input wrapper — uses shared composer__ classes (globally loaded via index.css) */}
          <div
            ref={inputWrapperRef}
            className={`composer__input-wrapper${fileId ? ' composer__input-wrapper--scoped' : ''}`}
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

            {/* Upload pending chip — spinner only, no text */}
            {uploadLoading && !fileId && (
              <span ref={pendingChipRef} className="composer__scope-chip" aria-label="Uploading…">
                <i className="ri-loader-4-line translate-page__spinner" aria-hidden />
              </span>
            )}

            {/* Textarea — uses shared composer__textarea */}
            <textarea
              ref={textareaRef}
              className={`composer__textarea${fileId ? ' composer__textarea--scoped' : ''}`}
              rows={1}
              placeholder={translatePlaceholder}
              value={steering}
              onChange={(e) => setSteering(e.target.value)}
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
                  title="Upload file"
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
                    disabled={isTranslating}
                    title={`Tone: ${tone}`}
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
                          {t.charAt(0).toUpperCase() + t.slice(1)}
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
                    title="Stop translation"
                    aria-label="Stop translation"
                  >
                    <i className="ri-stop-large-line" aria-hidden style={{ fontSize: '1.125rem' }} />
                  </button>
                ) : (
                  <button
                    type="button"
                    className="translate-page__send"
                    disabled={!canTranslate}
                    onClick={() => void handleTranslate()}
                    title={canTranslate ? 'Translate (⌘↵)' : isStreaming ? 'LLM busy' : 'Select a file first'}
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
