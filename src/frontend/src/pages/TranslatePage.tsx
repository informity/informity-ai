import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { PageHeader } from '../components/PageHeader'
import { CenteredState } from '../components/CenteredState'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { useBackendStatus } from '../context/useBackendStatus'
import { useTranslateContext } from '../context/useTranslateContext'
import { useChatContext } from '../context/useChatContext'
import { getFiles, uploadTranslateFile, deleteTranslateUpload, getSettings } from '../api'
import { extractErrorMessage } from '../utils/errorMessages'
import { showToast } from '../context/useToast'
import {
  resizeComposerTextarea,
  applyComposerScopedPadding,
} from '../utils/composerSizing'
import type { IndexedFile } from '../types/api'
import type { TranslateSection } from '../api'
import './TranslatePage.css'

// Language list with flag-icons country codes
const LANGUAGES: { value: string; label: string; flag: string }[] = [
  { value: 'Arabic',                  label: 'Arabic',                  flag: 'sa' },
  { value: 'Chinese (Simplified)',    label: 'Chinese (Simplified)',    flag: 'cn' },
  { value: 'Chinese (Traditional)',   label: 'Chinese (Traditional)',   flag: 'tw' },
  { value: 'Czech',                   label: 'Czech',                   flag: 'cz' },
  { value: 'Danish',                  label: 'Danish',                  flag: 'dk' },
  { value: 'Dutch',                   label: 'Dutch',                   flag: 'nl' },
  { value: 'Finnish',                 label: 'Finnish',                 flag: 'fi' },
  { value: 'French',                  label: 'French',                  flag: 'fr' },
  { value: 'German',                  label: 'German',                  flag: 'de' },
  { value: 'Hindi',                   label: 'Hindi',                   flag: 'in' },
  { value: 'Italian',                 label: 'Italian',                 flag: 'it' },
  { value: 'Japanese',                label: 'Japanese',                flag: 'jp' },
  { value: 'Korean',                  label: 'Korean',                  flag: 'kr' },
  { value: 'Norwegian',               label: 'Norwegian',               flag: 'no' },
  { value: 'Polish',                  label: 'Polish',                  flag: 'pl' },
  { value: 'Portuguese',              label: 'Portuguese',              flag: 'pt' },
  { value: 'Romanian',                label: 'Romanian',                flag: 'ro' },
  { value: 'Russian',                 label: 'Russian',                 flag: 'ru' },
  { value: 'Spanish',                 label: 'Spanish',                 flag: 'es' },
  { value: 'Swedish',                 label: 'Swedish',                 flag: 'se' },
  { value: 'Thai',                    label: 'Thai',                    flag: 'th' },
  { value: 'Turkish',                 label: 'Turkish',                 flag: 'tr' },
  { value: 'Ukrainian',               label: 'Ukrainian',               flag: 'ua' },
  { value: 'Vietnamese',              label: 'Vietnamese',              flag: 'vn' },
]

interface RunRecord {
  sections: TranslateSection[]
  language: string
  tone: string
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
    setFile, setTargetLanguage, setTone, startTranslation, cancelTranslation,
  } = useTranslateContext()

  // Translation runs accumulate above composer
  const [runs, setRuns] = useState<RunRecord[]>([])

  // Composer state — mirrors ChatView
  const [animateToDocked, setAnimateToDocked] = useState(false)
  const wasDocked = useRef(false)
  const isCentered = !isTranslating && !hasResult && runs.length === 0
  const activeRunRef = useRef<RunRecord | null>(null)
  const runStartRef = useRef<number>(0)

  // Steering prompt
  const [steering, setSteering] = useState('')

  // File search combobox
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<IndexedFile[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [uploadLoading, setUploadLoading] = useState(false)
  const comboRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chipRowRef = useRef<HTMLDivElement>(null)
  const inputWrapperRef = useRef<HTMLDivElement>(null)
  const resultsEndRef = useRef<HTMLDivElement>(null)

  // Load default language from settings once
  useEffect(() => {
    getSettings().then((s) => {
      const lang = (s as Record<string, unknown>)?.translate_default_language as string | undefined
      if (lang && LANGUAGES.some(l => l.value === lang)) {
        setTargetLanguage(lang)
      }
    }).catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Pre-load file from router state (Files → Translate navigation)
  useEffect(() => {
    const state = location.state as { scopedFileId?: number; scopedFileName?: string } | null
    if (state?.scopedFileId && state.scopedFileName) {
      setFile({
        id: state.scopedFileId,
        name: state.scopedFileName,
        pageCount: null,
        isUpload: false,
      })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Centered → docked transition
  useEffect(() => {
    if (wasDocked.current && isCentered) {
      // reset — new session
    } else if (!isCentered && (isTranslating || hasResult || runs.length > 0)) {
      if (!wasDocked.current) {
        setAnimateToDocked(true)
        const t = setTimeout(() => setAnimateToDocked(false), 1200)
        wasDocked.current = true
        return () => clearTimeout(t)
      }
    }
  }, [isCentered, isTranslating, hasResult, runs.length])

  // Resize textarea on steering input
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    resizeComposerTextarea(ta, !!fileId)
  }, [steering, fileId])

  // Update scoped padding when chip row changes
  useEffect(() => {
    applyComposerScopedPadding(inputWrapperRef.current, chipRowRef.current)
    const ta = textareaRef.current
    if (ta) resizeComposerTextarea(ta, !!fileId)
  }, [fileId])

  // Accumulate sections into active run as they arrive
  useEffect(() => {
    if (!isTranslating && !hasResult) return
    if (sections.length === 0) return

    if (!activeRunRef.current && isTranslating) {
      const run: RunRecord = {
        sections: [],
        language: targetLanguage,
        tone,
        steering,
        completedAt: null,
        totalSections: sectionCount,
        elapsedSeconds: null,
        fileLabel: fileName ?? 'Document',
      }
      activeRunRef.current = run
      runStartRef.current = Date.now()
      setRuns(prev => [...prev, run])
    }

    if (activeRunRef.current) {
      activeRunRef.current.sections = [...sections]
      activeRunRef.current.totalSections = sectionCount
      setRuns(prev => [...prev]) // trigger re-render
    }
  }, [sections, sectionCount, isTranslating, hasResult, targetLanguage, tone, steering, fileName])

  // Mark run complete
  useEffect(() => {
    if (jobStatus === 'done' && activeRunRef.current && activeRunRef.current.completedAt === null) {
      activeRunRef.current.completedAt = Date.now()
      activeRunRef.current.elapsedSeconds = Math.round((Date.now() - runStartRef.current) / 1000)
      setRuns(prev => [...prev])
      activeRunRef.current = null
    }
  }, [jobStatus])

  // Scroll to bottom as sections arrive
  useEffect(() => {
    if (isTranslating) resultsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sections.length, isTranslating])

  // File search suggestions
  useEffect(() => {
    if (!query.trim()) { setSuggestions([]); return }
    const t = setTimeout(async () => {
      try {
        const data = await getFiles({ search: query.trim(), limit: 12 }) as { files?: IndexedFile[] }
        setSuggestions(data.files || [])
      } catch { setSuggestions([]) }
    }, 200)
    return () => clearTimeout(t)
  }, [query])

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (comboRef.current && !comboRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSelectFile = useCallback((f: IndexedFile) => {
    setFile({ id: f.id, name: f.filename || f.path || `File #${f.id}`, pageCount: f.page_count ?? null, isUpload: false })
    setQuery(''); setSuggestions([]); setShowSuggestions(false)
  }, [setFile])

  const handleUpload = useCallback(async (file: File) => {
    setUploadLoading(true)
    try {
      const res = await uploadTranslateFile(file)
      setFile({ id: res.file_id, name: res.filename, pageCount: res.page_count ?? null, isUpload: true })
      showToast('success', `Uploaded: ${res.filename}`)
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

  const handleCopyRun = useCallback((run: RunRecord) => {
    const text = run.sections.map(s => s.text).join('\n\n')
    navigator.clipboard.writeText(text).then(() => showToast('success', 'Copied'))
  }, [])

  const handleSaveRun = useCallback((run: RunRecord, fmt: 'md' | 'txt') => {
    let content = run.sections.map(s => s.text).join('\n\n')
    if (fmt === 'txt') {
      content = content
        .replace(/^#{1,6}\s+/gm, '')
        .replace(/\*\*(.+?)\*\*/gs, '$1')
        .replace(/[*_]{1,2}(.+?)[*_]{1,2}/gs, '$1')
    }
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const base = run.fileLabel.replace(/\.[^.]+$/, '')
    const lang = run.language.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    a.href = url; a.download = `${base}.${lang}.${fmt}`
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url)
    showToast('success', `Saved ${fmt === 'md' ? 'Markdown' : 'text'}`)
  }, [])

  const selectedLang = LANGUAGES.find(l => l.value === targetLanguage)

  const canTranslate = !!fileId && !isTranslating && !isStreaming

  if (offline) {
    return (
      <div className="page page--translate">
        <PageHeader title="Translate" subtitle="Translate indexed documents locally" icon="ri-translate-2" />
        <div className="page__scroll"><ServiceUnavailableState /></div>
      </div>
    )
  }

  return (
    <div className="translate-page">
      <div className={`translate-page__body${isCentered ? ' translate-page__body--centered' : ''}`}>

        {/* ── Result area ── */}
        <div className="translate-results">
          {runs.length === 0 && !isTranslating && (
            <div className="translate-results__empty">
              <CenteredState
                icon="ri-translate-2"
                title="Select a document to translate"
                description="Search your library or drop a file in the box below, then choose a language and press Translate."
              />
            </div>
          )}
          {runs.map((run, ri) => (
            <div key={ri} className="translate-run">
              {/* Run header */}
              <div className="translate-run__header">
                <span className="translate-run__label">
                  {run.fileLabel} → {run.language}
                  {run.tone !== 'natural' && ` · ${run.tone}`}
                  {run.steering && ` · "${run.steering.slice(0, 40)}${run.steering.length > 40 ? '…' : ''}"`}
                </span>
              </div>
              {/* Sections */}
              <div className="translate-run__sections">
                {run.sections.map((s) => (
                  <div key={s.section_index} className="translate-run__section">
                    {s.section_title && s.section_title !== '(untitled)' && (
                      <p className="translate-run__section-title">{s.section_title}</p>
                    )}
                    <ReactMarkdown>{s.text}</ReactMarkdown>
                  </div>
                ))}
                {/* Streaming cursor for active run */}
                {ri === runs.length - 1 && isTranslating && (
                  <div className="translate-run__section translate-run__section--streaming">
                    <span className="translate-run__cursor" aria-label="Translating…" />
                    <span className="translate-run__progress">
                      {sectionCount
                        ? `Section ${completedSections + 1} of ${sectionCount}`
                        : 'Translating…'}
                    </span>
                  </div>
                )}
              </div>
              {/* Run footer — shown when complete */}
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

        {/* ── Composer ── */}
        <div className={`composer-wrap translate-composer${animateToDocked ? ' composer-wrap--docking' : ''}`}>

          {/* File chip row */}
          {fileId && (
            <div ref={chipRowRef} className="composer__chips translate-composer__chips">
              <span className="composer__chip translate-composer__chip">
                <span className="composer__chip-label">
                  <i className="ri-file-text-line" aria-hidden />
                  <span title={fileName ?? ''}>{fileName}</span>
                  {pageCount != null && <em>{pageCount}p</em>}
                </span>
                <button
                  type="button"
                  className="composer__chip-remove"
                  onClick={handleDismiss}
                  disabled={isTranslating}
                  title="Remove file"
                >
                  <i className="ri-close-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                </button>
              </span>
            </div>
          )}

          {/* Warnings */}
          {exceedsSoftLimit && fileId && estimatedMinutes !== null && (
            <p className="composer__warning">
              Large document (~{pageCount ?? '?'} pages) · ~{estimatedMinutes} min estimated — you can still proceed.
            </p>
          )}
          {isStreaming && (
            <p className="composer__warning">
              LLM in use by chat.{' '}
              <button type="button" className="translate-composer__stop-chat" onClick={() => void stopStreaming()}>
                Stop Chat
              </button>
            </p>
          )}

          {/* Input wrapper */}
          <div
            ref={inputWrapperRef}
            className="composer__input-wrapper translate-composer__input-wrapper"
          >
            {/* File search combobox — shown when no file selected */}
            {!fileId && (
              <div ref={comboRef} className="translate-composer__search">
                <i className="ri-search-line translate-composer__search-icon" aria-hidden />
                <input
                  className="translate-composer__search-input"
                  placeholder="Search library files… or drop a file below"
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setShowSuggestions(true) }}
                  onFocus={() => query && setShowSuggestions(true)}
                  aria-label="Search library files"
                />
                {showSuggestions && suggestions.length > 0 && (
                  <ul className="translate-composer__suggestions" role="listbox">
                    {suggestions.map(f => (
                      <li key={f.id} className="translate-composer__suggestion" role="option"
                        onMouseDown={(e) => { e.preventDefault(); handleSelectFile(f) }}>
                        <i className="ri-file-text-line" aria-hidden />
                        <span className="translate-composer__suggestion-name">{f.filename || f.path}</span>
                        {f.page_count && <em className="translate-composer__suggestion-meta">{f.page_count}p</em>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Steering textarea */}
            <textarea
              ref={textareaRef}
              className={`composer__textarea translate-composer__textarea${fileId ? ' composer__textarea--scoped' : ''}`}
              placeholder={fileId
                ? 'Steer translation (optional) — e.g. focus on methodology, skip references…'
                : 'Or drop a file here to upload…'}
              value={steering}
              onChange={(e) => setSteering(e.target.value)}
              disabled={!fileId || isTranslating}
              rows={1}
              onDragOver={(e) => { if (!fileId) e.preventDefault() }}
              onDrop={(e) => {
                if (!fileId) {
                  e.preventDefault()
                  const f = e.dataTransfer.files[0]
                  if (f) void handleUpload(f)
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canTranslate) {
                  e.preventDefault(); void handleTranslate()
                }
              }}
            />

            {/* Controls row */}
            <div className="composer__controls-row translate-composer__controls-row">
              <div className="composer__controls-left translate-composer__controls-left">
                {/* Language selector */}
                <div className="translate-composer__lang-wrap">
                  {selectedLang && (
                    <span className={`fi fi-${selectedLang.flag} translate-composer__flag`} aria-hidden />
                  )}
                  <select
                    className="translate-composer__select"
                    value={targetLanguage}
                    onChange={(e) => setTargetLanguage(e.target.value)}
                    disabled={isTranslating}
                    title="Target language"
                  >
                    {LANGUAGES.map(l => (
                      <option key={l.value} value={l.value}>{l.label}</option>
                    ))}
                  </select>
                </div>

                {/* Tone selector */}
                <select
                  className="translate-composer__select"
                  value={tone}
                  onChange={(e) => setTone(e.target.value)}
                  disabled={isTranslating}
                  title="Translation tone"
                >
                  <option value="natural">Natural</option>
                  <option value="formal">Formal</option>
                  <option value="literal">Literal</option>
                </select>

                {/* Upload button when no file */}
                {!fileId && (
                  <>
                    <button
                      type="button"
                      className="translate-composer__upload-btn"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploadLoading}
                      title="Upload a file"
                    >
                      <i className={uploadLoading ? 'ri-loader-4-line translate-composer__spinner' : 'ri-upload-2-line'} aria-hidden />
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      className="translate-composer__file-input"
                      onChange={(e) => {
                        const f = e.target.files?.[0]
                        if (f) { void handleUpload(f); e.target.value = '' }
                      }}
                    />
                  </>
                )}
              </div>

              <div className="composer__controls-right">
                {isTranslating ? (
                  <button
                    type="button"
                    className="composer__send translate-composer__cancel"
                    onClick={cancelTranslation}
                    title="Cancel translation"
                  >
                    <i className="ri-stop-line" aria-hidden style={{ fontSize: '1rem' }} />
                  </button>
                ) : (
                  <button
                    type="button"
                    className={`composer__send composer__send--active translate-composer__send${!canTranslate ? ' translate-composer__send--disabled' : ''}`}
                    onClick={() => void handleTranslate()}
                    disabled={!canTranslate}
                    title={canTranslate ? 'Translate (⌘Enter)' : isStreaming ? 'LLM busy' : 'Select a file first'}
                  >
                    <i className="ri-translate-2" aria-hidden style={{ fontSize: '1rem' }} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
