import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { PageHeader } from '../components/PageHeader'
import { CenteredState } from '../components/CenteredState'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { useBackendStatus } from '../context/useBackendStatus'
import { useTranslateContext } from '../context/useTranslateContext'
import { useChatContext } from '../context/useChatContext'
import { getFiles, uploadTranslateFile, deleteTranslateUpload } from '../api'
import { extractErrorMessage } from '../utils/errorMessages'
import { showToast } from '../context/useToast'
import type { IndexedFile } from '../types/api'
import './TranslatePage.css'

const TARGET_LANGUAGES = [
  'Arabic', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Czech', 'Danish',
  'Dutch', 'English', 'French', 'Finnish', 'German', 'Hindi', 'Italian',
  'Japanese', 'Korean', 'Norwegian', 'Polish', 'Portuguese', 'Romanian',
  'Russian', 'Spanish', 'Swedish', 'Thai', 'Turkish', 'Ukrainian', 'Vietnamese',
]

export function TranslatePage() {
  const { offline } = useBackendStatus()
  const { isStreaming, stopStreaming } = useChatContext()
  const {
    sections, sectionCount, completedSections, failedSections, jobStatus,
    estimatedMinutes, exceedsSoftLimit, isTranslating, hasResult,
    targetLanguage, tone, outputMode, fileId, fileName, pageCount,
    setFile, setTargetLanguage, setTone, setOutputMode,
    startTranslation, cancelTranslation, clearResult,
  } = useTranslateContext()

  // Combobox state
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<IndexedFile[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [uploadLoading, setUploadLoading] = useState(false)
  const comboRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)
  const [dragOver, setDragOver] = useState(false)

  // Fetch suggestions as user types
  useEffect(() => {
    if (!query.trim()) { setSuggestions([]); return }
    const timer = setTimeout(async () => {
      try {
        const data = await getFiles({ search: query.trim(), limit: 15 }) as { files?: IndexedFile[] }
        setSuggestions(data.files || [])
      } catch { setSuggestions([]) }
    }, 200)
    return () => clearTimeout(timer)
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

  const selectLibraryFile = useCallback((f: IndexedFile) => {
    setFile({ id: f.id, name: f.filename || f.path || `File #${f.id}`, pageCount: f.page_count ?? null, isUpload: false })
    setQuery('')
    setSuggestions([])
    setShowSuggestions(false)
    clearResult()
  }, [setFile, clearResult])

  const handleUploadFile = useCallback(async (file: File) => {
    setUploadLoading(true)
    try {
      const res = await uploadTranslateFile(file)
      setFile({ id: res.file_id, name: res.filename, pageCount: res.page_count ?? null, isUpload: true })
      clearResult()
      showToast('success', `Uploaded: ${res.filename}`)
    } catch (err) {
      showToast('error', extractErrorMessage(err, 'Upload failed'))
    } finally {
      setUploadLoading(false)
    }
  }, [setFile, clearResult])

  const handleDismissFileClean = useCallback(async () => {
    if (fileId) {
      // Best-effort: delete translate uploads; library files return 403 which we ignore.
      try { await deleteTranslateUpload(fileId) } catch { /* ignored */ }
    }
    setFile(null)
    clearResult()
  }, [fileId, setFile, clearResult])

  const handleCopy = useCallback(() => {
    const text = sections.map((s) => {
      const title = s.section_title ? `## ${s.section_title}\n\n` : ''
      return `${title}${s.text}`
    }).join('\n\n---\n\n')
    navigator.clipboard.writeText(text).then(() => showToast('success', 'Copied to clipboard'))
  }, [sections])

  const handleSave = useCallback((fmt: 'md' | 'txt') => {
    const lines: string[] = []
    for (const s of sections) {
      if (s.section_title) lines.push(`## ${s.section_title}\n`)
      lines.push(s.text)
      lines.push('')
    }
    let content = lines.join('\n')
    if (fmt === 'txt') {
      content = content
        .replace(/^#{1,6}\s+/gm, '')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/[*_]{1,2}(.+?)[*_]{1,2}/g, '$1')
    }
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const base = (fileName || 'translation').replace(/\.[^.]+$/, '')
    const lang = targetLanguage.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    a.href = url
    a.download = `${base}.${lang}.${fmt}`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }, [sections, fileName, targetLanguage])

  const progressLabel = (() => {
    if (jobStatus === 'queued') return 'Preparing…'
    if (isTranslating && sectionCount) return `Section ${completedSections + 1} of ${sectionCount}`
    if (isTranslating) return 'Translating…'
    if (jobStatus === 'done') return `Complete · ${completedSections} section${completedSections !== 1 ? 's' : ''}${failedSections > 0 ? ` (${failedSections} failed)` : ''}`
    if (jobStatus === 'failed') return 'Translation failed'
    if (jobStatus === 'stalled') return 'Translation stalled'
    return null
  })()

  if (offline) {
    return (
      <div className="page page--translate">
        <PageHeader title="Translate" subtitle="Translate indexed documents locally" icon="ri-translate-2" />
        <div className="page__scroll"><ServiceUnavailableState /></div>
      </div>
    )
  }

  return (
    <div className="page page--translate">
      <PageHeader title="Translate" subtitle="Translate indexed documents locally using the Qwen model" icon="ri-translate-2" />
      <div className="page__scroll">
        <div className="translate-page">

          {/* ── Zone 1: File picker ── */}
          <div className="translate-picker">
            {fileId ? (
              <div className="translate-chip">
                <i className="ri-file-text-line translate-chip__icon" aria-hidden />
                <span className="translate-chip__name" title={fileName ?? ''}>{fileName}</span>
                {pageCount && <span className="translate-chip__meta">{pageCount}p</span>}
                <button type="button" className="translate-chip__dismiss" onClick={handleDismissFileClean} title="Clear file">
                  <i className="ri-close-line" aria-hidden />
                </button>
              </div>
            ) : (
              <div className="translate-picker__inputs">
                <div className="translate-combobox" ref={comboRef}>
                  <div className="translate-combobox__input-wrap">
                    <i className="ri-search-line translate-combobox__icon" aria-hidden />
                    <input
                      className="translate-combobox__input"
                      placeholder="Search library files…"
                      value={query}
                      onChange={(e) => { setQuery(e.target.value); setShowSuggestions(true) }}
                      onFocus={() => query && setShowSuggestions(true)}
                      aria-label="Search library files"
                    />
                  </div>
                  {showSuggestions && suggestions.length > 0 && (
                    <ul className="translate-combobox__list" role="listbox">
                      {suggestions.map((f) => (
                        <li key={f.id} className="translate-combobox__option" role="option"
                          onMouseDown={(e) => { e.preventDefault(); selectLibraryFile(f) }}>
                          <i className="ri-file-text-line translate-combobox__option-icon" aria-hidden />
                          <span className="translate-combobox__option-name">{f.filename || f.path}</span>
                          {f.page_count && <span className="translate-combobox__option-meta">{f.page_count}p</span>}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div
                  ref={dropZoneRef}
                  className={`translate-dropzone${dragOver ? ' translate-dropzone--over' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault(); setDragOver(false)
                    const f = e.dataTransfer.files[0]
                    if (f) handleUploadFile(f)
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <i className="ri-upload-cloud-2-line translate-dropzone__icon" aria-hidden />
                  <span>{uploadLoading ? 'Uploading…' : 'Drop a file or click to upload'}</span>
                  <input ref={fileInputRef} type="file" className="translate-dropzone__input"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) { handleUploadFile(f); e.target.value = '' } }} />
                </div>
              </div>
            )}

            {/* Estimate warning */}
            {fileId && estimatedMinutes !== null && (
              <p className={`translate-estimate${exceedsSoftLimit ? ' translate-estimate--warn' : ''}`}>
                {exceedsSoftLimit
                  ? `Large document (${pageCount ?? '?'} pages) · ~${estimatedMinutes} min estimated`
                  : `~${estimatedMinutes} min estimated`}
              </p>
            )}
          </div>

          {/* ── Zone 2: Controls ── */}
          <div className="translate-controls">
            <select className="translate-controls__select" value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)} disabled={isTranslating}>
              {TARGET_LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
            <select className="translate-controls__select" value={tone}
              onChange={(e) => setTone(e.target.value)} disabled={isTranslating}>
              <option value="natural">Natural</option>
              <option value="formal">Formal</option>
              <option value="literal">Literal</option>
            </select>
            <select className="translate-controls__select" value={outputMode}
              onChange={(e) => setOutputMode(e.target.value as 'markdown' | 'text')} disabled={isTranslating}>
              <option value="markdown">Markdown</option>
              <option value="text">Plain text</option>
            </select>
            {isTranslating ? (
              <button type="button" className="settings-btn settings-btn--secondary translate-controls__btn"
                onClick={cancelTranslation}>Cancel</button>
            ) : isStreaming ? (
              <button type="button" className="settings-btn settings-btn--secondary translate-controls__btn"
                onClick={() => void stopStreaming()}>Stop Chat</button>
            ) : (
              <button type="button" className="settings-btn settings-btn--primary translate-controls__btn"
                disabled={!fileId || isStreaming}
                onClick={() => startTranslation()}>
                Translate
              </button>
            )}
          </div>

          {isStreaming && !isTranslating && (
            <p className="translate-busy">LLM is in use by chat. Stop chat to translate.</p>
          )}

          {/* ── Zone 3: Result ── */}
          {hasResult || isTranslating ? (
            <div className="translate-result">
              {/* Sticky toolbar */}
              <div className="translate-result__toolbar">
                <span className="translate-result__progress">{progressLabel}</span>
                <div className="translate-result__actions">
                  {hasResult && (
                    <>
                      <button type="button" className="translate-result__action" onClick={handleCopy}>Copy</button>
                      {!isTranslating && (
                        <>
                          <button type="button" className="translate-result__action" onClick={() => handleSave('md')}>Save .md</button>
                          <button type="button" className="translate-result__action" onClick={() => handleSave('txt')}>Save .txt</button>
                        </>
                      )}
                    </>
                  )}
                  {isTranslating && (
                    <button type="button" className="translate-result__action translate-result__action--cancel"
                      onClick={cancelTranslation}>Cancel</button>
                  )}
                </div>
              </div>

              {/* Sections */}
              <div className="translate-result__body">
                {sections.map((s) => (
                  <div key={s.section_index} className="translate-section">
                    {outputMode === 'markdown'
                      ? <ReactMarkdown>{s.text}</ReactMarkdown>
                      : <p className="translate-section__plain">{s.text}</p>}
                  </div>
                ))}
                {isTranslating && (
                  <div className="translate-section translate-section--loading">
                    <span className="translate-section__cursor" aria-label="Translating…" />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="translate-result translate-result--empty">
              <CenteredState
                icon="ri-translate-2"
                title="Select a document to translate"
                description="Search your library or drop a file above, then choose a language and press Translate."
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
