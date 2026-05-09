/**
 * Informity AI — Settings view
 * Full settings form with sections: Privacy, Appearance, Data Sources, Indexing,
 * Chat, Diagnostics, Models, System. Save, Discard, Reset Settings, Danger Zone.
 */
import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import {
  cancelModelDownload,
  downloadModel,
  getModelOperationEvents,
  getModelProfile,
  getModelsCatalog,
  type ModelOperationEventResponse,
  type ModelsCatalogResponse,
} from '../../api'
import { isChatMode, type ChatMode } from '../../types/api'
import { SETTINGS_ACTIVE_TAB_STORAGE_KEY } from '../../utils/storageKeys'
import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_OPTIONS, UI_THEME_STORAGE_KEY } from '../../utils/uiTheme'
import { formatModelSizeGb } from '../../utils/formatModelSizeGb'
import { getFriendlyModelDownloadError } from '../../utils/modelDownloadErrors'
import { isDesktopRuntime, nativePickDirectoryDialog } from '../../tauriRuntime'
import '../../styles/shared/buttons.css'
import './SettingsView.css'
const DIAGNOSTICS_PROFILE_OPTIONS = [
  { value: 'standard', label: 'Standard' },
  { value: 'troubleshooting', label: 'Troubleshooting' },
  { value: 'custom', label: 'Custom (Advanced)' },
]
const TRACE_REDACTION_OPTIONS = [
  { value: 'minimal', label: 'Minimal (Recommended)' },
  { value: 'strict', label: 'Strict' },
  { value: 'off', label: 'Off (Least Private)' },
]
const LOG_LEVEL_OPTIONS = [
  { value: 'debug', label: 'Debug' },
  { value: 'info', label: 'Info (Recommended)' },
  { value: 'warning', label: 'Warning' },
  { value: 'error', label: 'Error' },
]
const CHAT_MODE_OPTIONS = [
  { value: 'researcher', label: 'Researcher (Recommended)' },
  { value: 'assistant', label: 'Assistant' },
]

type SettingsTab =
  | 'general'
  | 'chat'
  | 'data'
  | 'indexing'
  | 'diagnostics'
  | 'models'
  | 'system'

const SETTINGS_TABS: Array<{ id: SettingsTab; label: string; icon: string }> = [
  { id: 'general', label: 'General', icon: 'ri-home-gear-line' },
  { id: 'chat', label: 'Chat', icon: 'ri-chat-ai-4-line' },
  { id: 'models', label: 'Models', icon: 'ri-robot-2-line' },
  { id: 'data', label: 'Data Sources', icon: 'ri-folder-line' },
  { id: 'indexing', label: 'Indexing', icon: 'ri-stack-line' },
  { id: 'diagnostics', label: 'Diagnostics', icon: 'ri-pulse-line' },
  { id: 'system', label: 'System', icon: 'ri-server-line' },
]
const SETTINGS_TAB_IDS = new Set<SettingsTab>(SETTINGS_TABS.map((tab) => tab.id))
const FILE_TYPE_DISPLAY_ORDER = ['pdf', 'docx', 'spreadsheet', 'pptx', 'epub', 'web', 'text', 'data'] as const

const INDEXING_SPEED_LABELS = ['', 'Responsive', 'Gentle', 'Balanced', 'Fast', 'Fastest']
const INDEXING_SPEED_TO_THREADS = [2, 4, 6, 8, 0]
const CHAT_CPU_RESPONSIVENESS_LABELS = ['', 'Most Responsive', 'Balanced', 'Fastest']
const CHAT_CPU_RESPONSIVENESS_TO_THREADS = [2, 4, 6]
const MASKED_TAVILY_KEY = '••••••••••••••••••••••••••••••••••••••••••••••••••'
const MASKED_LINKUP_KEY = '••••••••••••••••••••••••••••••••••••••••••••••••••'
const MODEL_FILENAME_ALIASES: Record<string, string> = {
  'Qwen3.6-35B-A3B-Q4_K_M.gguf': 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
}

function threadsToSpeed(threads: number): number {
  if (threads === 0) return 5
  return Math.min(4, Math.max(1, Math.round(threads / 2)))
}

function llmThreadsToResponsiveness(threads: number): number {
  if (threads <= 2) return 1
  if (threads <= 4) return 2
  return 3
}

function parseInteger(raw: string, fallback: number): number {
  const parsed = Number.parseInt(raw, 10)
  return Number.isNaN(parsed) ? fallback : parsed
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function canonicalizeModelFilename(filename: string | null | undefined): string {
  const value = String(filename || '').trim()
  if (!value) return ''
  return MODEL_FILENAME_ALIASES[value] || value
}

function isMaskedTavilyKey(value: string): boolean {
  return value === MASKED_TAVILY_KEY
}

function isMaskedLinkupKey(value: string): boolean {
  return value === MASKED_LINKUP_KEY
}

function renderLabelWithMutedParens(text: string) {
  const parts = text.split(/(\([^)]*\))/g)
  return (
    <>
      {parts.map((part, index) => (
        /^\([^)]*\)$/.test(part)
          ? <span key={`paren-${index}`} className="settings-label-paren-muted">{part}</span>
          : <span key={`text-${index}`}>{part}</span>
      ))}
    </>
  )
}

interface ModelProfile {
  name?: string
  family?: string
  reasoning_mode?: string
  max_tokens?: number
  context_length?: number
  temperature?: number
  rag_top_k?: number
  coverage_top_k?: number
  rag_max_score?: number
  rag_context_ratio?: number
  top_p?: number
  prompt_format?: string
}

interface FileTypeOption {
  id: string
  label: string
  extensions: string[]
}

interface DiagnosticsProfilePreset {
  log_level: string
  chat_trace_logging: boolean
  chat_trace_redaction_mode: string
  chat_trace_user_retention_days: number
  chat_trace_evaluation_retention_days: number
}

interface SettingsData {
  watched_directories?: string[]
  ignore_patterns?: string[]
  exclude_macos_system?: boolean
  exclude_developer_data?: boolean
  supported_extensions?: string[]
  follow_symlinks?: boolean
  chunk_size_tokens?: number
  chunk_overlap_tokens?: number
  embedding_batch_size?: number
  embedding_max_threads?: number
  llm_cpu_threads?: number
  enable_ocr_for_images?: boolean
  max_indexable_file_size_mb?: number
  scan_file_timeout_seconds?: number
  full_privacy?: boolean
  tavily_api_key_set?: boolean
  linkup_api_key_set?: boolean
  web_search_max_results?: number
  web_search_timeout_seconds?: number
  adaptive_rag_tuning?: boolean
  chat_history_messages?: number
  default_chat_mode?: ChatMode
  enable_chat_roles?: boolean
  entity_extract_acronym?: boolean
  entity_extract_person_name?: boolean
  entity_extract_organization?: boolean
  entity_extract_location?: boolean
  entity_extract_numeric_id?: boolean
  log_level?: string
  diagnostics_profile?: string
  diagnostics_profile_presets?: Record<string, DiagnosticsProfilePreset>
  chat_trace_logging?: boolean
  chat_trace_redaction_mode?: string
  chat_trace_user_retention_days?: number
  chat_trace_evaluation_retention_days?: number
  enable_raw_output_control?: boolean
  ui_theme?: string
  enable_menu_bar_icon?: boolean
  llm_model_filename?: string
  available_models?: string[]
  embedding_model?: string
  rag_reranker_model?: string
  model_profile?: ModelProfile
  file_type_options?: FileTypeOption[]
}

interface FormState {
  watched_directories: string[]
  ignore_patterns: string[]
  exclude_macos_system: boolean
  exclude_developer_data: boolean
  supported_extensions: string[]
  follow_symlinks: boolean
  chunk_size_tokens: number
  chunk_overlap_tokens: number
  embedding_batch_size: number
  embedding_max_threads: number
  llm_cpu_threads: number
  enable_ocr_for_images: boolean
  max_indexable_file_size_mb: number
  scan_file_timeout_seconds: number
  full_privacy: boolean
  tavily_api_key: string
  clear_tavily_api_key: boolean
  linkup_api_key: string
  clear_linkup_api_key: boolean
  web_search_max_results: number
  web_search_timeout_seconds: number
  adaptive_rag_tuning: boolean
  chat_history_messages: number
  default_chat_mode: ChatMode
  enable_chat_roles: boolean
  entity_extract_acronym: boolean
  entity_extract_person_name: boolean
  entity_extract_organization: boolean
  entity_extract_location: boolean
  entity_extract_numeric_id: boolean
  log_level: string
  diagnostics_profile: string
  chat_trace_logging: boolean
  chat_trace_redaction_mode: string
  chat_trace_user_retention_days: number
  chat_trace_evaluation_retention_days: number
  enable_raw_output_control: boolean
  ui_theme: string
  enable_menu_bar_icon: boolean
  llm_model_filename: string
}

interface SettingsViewProps {
  settings: SettingsData | null
  fileTypeOptions?: FileTypeOption[]
  onSave: (form: FormState) => void
  onDiscard?: () => void
  onResetSettings: () => void
  onResetIndex: () => void
  saving: boolean
}

function buildFormState(settings: SettingsData): FormState {
  const normalizedTheme = normalizeUiTheme(settings.ui_theme)
  return {
    watched_directories: [...(settings.watched_directories || [])],
    ignore_patterns: [...(settings.ignore_patterns || [])],
    exclude_macos_system: settings.exclude_macos_system ?? true,
    exclude_developer_data: settings.exclude_developer_data ?? true,
    supported_extensions: [...(settings.supported_extensions || [])],
    follow_symlinks: settings.follow_symlinks ?? false,
    chunk_size_tokens: settings.chunk_size_tokens ?? 512,
    chunk_overlap_tokens: settings.chunk_overlap_tokens ?? 60,
    embedding_batch_size: settings.embedding_batch_size ?? 32,
    embedding_max_threads: settings.embedding_max_threads ?? 6,
    llm_cpu_threads: settings.llm_cpu_threads ?? 4,
    enable_ocr_for_images: settings.enable_ocr_for_images ?? true,
    max_indexable_file_size_mb: settings.max_indexable_file_size_mb ?? 100,
    scan_file_timeout_seconds: settings.scan_file_timeout_seconds ?? 600,
    full_privacy: settings.full_privacy ?? true,
    tavily_api_key: settings.tavily_api_key_set ? MASKED_TAVILY_KEY : '',
    clear_tavily_api_key: false,
    linkup_api_key: settings.linkup_api_key_set ? MASKED_LINKUP_KEY : '',
    clear_linkup_api_key: false,
    web_search_max_results: settings.web_search_max_results ?? 5,
    web_search_timeout_seconds: settings.web_search_timeout_seconds ?? 8,
    adaptive_rag_tuning: settings.adaptive_rag_tuning ?? true,
    chat_history_messages: settings.chat_history_messages ?? 5,
    default_chat_mode: isChatMode(settings.default_chat_mode) ? settings.default_chat_mode : 'researcher',
    enable_chat_roles: settings.enable_chat_roles ?? false,
    entity_extract_acronym: settings.entity_extract_acronym ?? true,
    entity_extract_person_name: settings.entity_extract_person_name ?? false,
    entity_extract_organization: settings.entity_extract_organization ?? false,
    entity_extract_location: settings.entity_extract_location ?? false,
    entity_extract_numeric_id: settings.entity_extract_numeric_id ?? false,
    log_level: settings.log_level ?? 'info',
    diagnostics_profile: settings.diagnostics_profile ?? 'standard',
    chat_trace_logging: settings.chat_trace_logging ?? false,
    chat_trace_redaction_mode: settings.chat_trace_redaction_mode ?? 'minimal',
    chat_trace_user_retention_days: settings.chat_trace_user_retention_days ?? 30,
    chat_trace_evaluation_retention_days: settings.chat_trace_evaluation_retention_days ?? 30,
    enable_raw_output_control: settings.enable_raw_output_control ?? false,
    ui_theme: normalizedTheme ?? UI_THEME_DEFAULT,
    enable_menu_bar_icon: settings.enable_menu_bar_icon ?? false,
    llm_model_filename: canonicalizeModelFilename(settings.llm_model_filename),
  }
}

function ProfileRow({ label, value }: { label: string; value: string | number | undefined }) {
  return (
    <div className="settings-profile-row">
      <span className="settings-profile-row__label">{renderLabelWithMutedParens(label)}</span>
      <span className="settings-profile-row__value">{value ?? '--'}</span>
    </div>
  )
}

function getInitialActiveTab(): SettingsTab {
  try {
    const saved = localStorage.getItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY)
    if (saved && SETTINGS_TAB_IDS.has(saved as SettingsTab)) {
      return saved as SettingsTab
    }
  } catch {
    // Ignore localStorage errors and use default tab.
  }
  return 'general'
}

export function SettingsView({
  settings,
  fileTypeOptions,
  onSave,
  onDiscard,
  onResetSettings,
  onResetIndex,
  saving,
}: SettingsViewProps) {
  const [form, setForm] = useState<FormState>(() => buildFormState(settings || {}))
  const [activeTab, setActiveTab] = useState<SettingsTab>(getInitialActiveTab)
  const [previewProfile, setPreviewProfile] = useState<ModelProfile | null>(null)
  const [modelProfileNames, setModelProfileNames] = useState<Map<string, string>>(new Map())
  const [dirInput, setDirInput] = useState('')
  const [ignoreInput, setIgnoreInput] = useState('')
  const [modelsCatalog, setModelsCatalog] = useState<ModelsCatalogResponse | null>(null)
  const [modelDownloadPending, setModelDownloadPending] = useState(false)
  const [modelDownloadError, setModelDownloadError] = useState<string | null>(null)
  const [modelEvent, setModelEvent] = useState<ModelOperationEventResponse | null>(null)
  const modelEventStateRef = useRef<ModelOperationEventResponse['state'] | null>(null)
  const persistedModel = canonicalizeModelFilename(settings?.llm_model_filename ?? '')
  const effectiveProfile = previewProfile ?? settings?.model_profile

  useEffect(() => {
    if (settings) {
      setForm(buildFormState(settings))
      setPreviewProfile(null)
    }
  }, [settings])

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, activeTab)
    } catch {
      // Ignore localStorage errors.
    }
  }, [activeTab])

  useEffect(() => {
    const selected = canonicalizeModelFilename(form.llm_model_filename)
    if (!settings || !selected || selected === persistedModel) {
      setPreviewProfile(null)
      return
    }

    let cancelled = false
    getModelProfile(selected)
      .then((data) => {
        if (cancelled) return
        setPreviewProfile(data as ModelProfile)
      })
      .catch(() => {
        if (!cancelled) setPreviewProfile(null)
      })

    return () => {
      cancelled = true
    }
  }, [form.llm_model_filename, persistedModel, settings])

  useEffect(() => {
    const models = settings?.available_models ?? []
    if (models.length === 0) return
    let cancelled = false
    Promise.all(
      models.map((filename) =>
        getModelProfile(filename)
          .then((data) => ({ filename, name: (data as ModelProfile).name ?? filename }))
          .catch(() => ({ filename, name: filename })),
      ),
    ).then((results) => {
      if (cancelled) return
      const getBSize = (name: string) => { const m = name.match(/(\d+)B/i); return m ? parseInt(m[1], 10) : Infinity }
      results.sort((a, b) => getBSize(a.name) - getBSize(b.name))
      setModelProfileNames(new Map(results.map((r) => [r.filename, r.name])))
    }).catch(() => {})
    return () => { cancelled = true }
  }, [settings?.available_models])

  useEffect(() => {
    let cancelled = false
    getModelsCatalog()
      .then((catalog) => {
        if (cancelled) return
        setModelsCatalog(catalog)
      })
      .catch(() => {
        // Ignore catalog fetch failures here; model dropdown falls back to available_models.
      })
    return () => {
      cancelled = true
    }
  }, [settings?.llm_model_filename, settings?.available_models])

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const event = await getModelOperationEvents()
        if (cancelled) return

        const prevState = modelEventStateRef.current
        modelEventStateRef.current = event.state
        setModelEvent(event)

        if (prevState === 'in_progress' && (event.state === 'completed' || event.state === 'cancelled' || event.state === 'failed')) {
          await refreshModelsCatalog()
        }

        if (event.state === 'failed') {
          setModelDownloadError(getFriendlyModelDownloadError(event.error || ''))
        } else if (event.state === 'completed' || event.state === 'cancelled' || event.state === 'idle') {
          setModelDownloadError(null)
        }
      } catch {
        // Keep last known event if polling fails.
      }
    }

    void poll()
    const id = window.setInterval(() => { void poll() }, 1500)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const refreshModelsCatalog = async (): Promise<void> => {
    try {
      const catalog = await getModelsCatalog()
      setModelsCatalog(catalog)
    } catch {
      // Keep existing catalog state on fetch errors.
    }
  }

  if (!settings) return null
  const diagnosticsPresetValues = settings.diagnostics_profile_presets ?? {}

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }))

  const applyDiagnosticsProfile = (profile: string) => {
    setForm((prev) => {
      const preset = diagnosticsPresetValues[profile]
      if (!preset) {
        return { ...prev, diagnostics_profile: profile }
      }
      return {
        ...prev,
        diagnostics_profile: profile,
        log_level: preset.log_level,
        chat_trace_logging: preset.chat_trace_logging,
        chat_trace_redaction_mode: preset.chat_trace_redaction_mode,
        chat_trace_user_retention_days: preset.chat_trace_user_retention_days,
        chat_trace_evaluation_retention_days: preset.chat_trace_evaluation_retention_days,
      }
    })
  }

  const updateDiagnosticsControl = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((prev) => ({ ...prev, [key]: value, diagnostics_profile: 'custom' }))

  const addDir = () => {
    const path = dirInput.trim()
    if (!path) return
    if (form.watched_directories.includes(path)) return
    update('watched_directories', [...form.watched_directories, path])
    setDirInput('')
  }

  const removeDir = (path: string) => {
    update('watched_directories', form.watched_directories.filter((p) => p !== path))
  }

  const browseDir = async () => {
    const selected = await nativePickDirectoryDialog('Choose Source Directory')
    if (!selected) return
    setDirInput(selected)
  }

  const addIgnore = () => {
    const pattern = ignoreInput.trim()
    if (!pattern) return
    if (form.ignore_patterns.includes(pattern)) return
    update('ignore_patterns', [...form.ignore_patterns, pattern])
    setIgnoreInput('')
  }

  const removeIgnore = (pattern: string) => {
    update('ignore_patterns', form.ignore_patterns.filter((p) => p !== pattern))
  }

  const canAddDir = dirInput.trim().length > 0
  const canAddIgnore = ignoreInput.trim().length > 0

  const speedVal = threadsToSpeed(form.embedding_max_threads ?? 6)
  const handleSpeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseInt(e.target.value, 10)
    const threads = INDEXING_SPEED_TO_THREADS[Math.max(0, Math.min(v - 1, 4))]
    update('embedding_max_threads', threads)
  }
  const chatCpuVal = llmThreadsToResponsiveness(form.llm_cpu_threads ?? 4)
  const handleChatCpuChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseInt(e.target.value, 10)
    const threads = CHAT_CPU_RESPONSIVENESS_TO_THREADS[Math.max(0, Math.min(v - 1, 2))]
    update('llm_cpu_threads', threads)
  }

  const handleSave = () => onSave(form)
  const handleDiscard = () => {
    setForm(buildFormState(settings))
    onDiscard?.()
  }

  const selectedModelFilename = canonicalizeModelFilename(form.llm_model_filename || settings.llm_model_filename || '')
  const catalogModels = modelsCatalog?.models || []
  const selectedCatalogEntry = catalogModels.find((model) => model.model_filename === selectedModelFilename)
  const knownModelFilenames = (() => {
    const ordered: string[] = []
    const seen = new Set<string>()
    const add = (value: string | undefined | null) => {
      const normalized = canonicalizeModelFilename(value)
      if (!normalized || seen.has(normalized)) return
      seen.add(normalized)
      ordered.push(normalized)
    }

    for (const model of catalogModels) add(model.model_filename)
    for (const model of settings.available_models || []) add(model)
    add(selectedModelFilename)
    return ordered
  })()
  const installedModelSet = new Set(
    catalogModels.length > 0
      ? catalogModels.filter((model) => model.installed).map((model) => canonicalizeModelFilename(model.model_filename))
      : (settings.available_models || []).map((model) => canonicalizeModelFilename(model)),
  )
  const selectedModelInstalled = selectedModelFilename ? installedModelSet.has(selectedModelFilename) : false
  const modelEventMatchesSelected = modelEvent?.model_filename === selectedModelFilename
  const modelDownloadInProgress = modelEventMatchesSelected && modelEvent?.state === 'in_progress'
  const canSaveSettings = !saving && selectedModelInstalled && !modelDownloadInProgress && !modelDownloadPending
  const orderedFileTypeOptions = [...(fileTypeOptions || [])].sort((a, b) => {
    const ai = FILE_TYPE_DISPLAY_ORDER.indexOf(a.id as (typeof FILE_TYPE_DISPLAY_ORDER)[number])
    const bi = FILE_TYPE_DISPLAY_ORDER.indexOf(b.id as (typeof FILE_TYPE_DISPLAY_ORDER)[number])
    const aRank = ai === -1 ? Number.MAX_SAFE_INTEGER : ai
    const bRank = bi === -1 ? Number.MAX_SAFE_INTEGER : bi
    if (aRank !== bRank) return aRank - bRank
    return a.label.localeCompare(b.label)
  })

  const formatBytes = (value: number): string => {
    if (!Number.isFinite(value) || value <= 0) return '0 KB'
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let unit = 0
    let next = value
    while (next >= 1024 && unit < units.length - 1) {
      next /= 1024
      unit += 1
    }
    const precision = next >= 100 || unit === 0 ? 0 : 1
    return `${next.toFixed(precision)} ${units[unit]}`
  }
  const modelProgressSummary = (() => {
    if (!modelEventMatchesSelected || !modelEvent) return null
    if (modelEvent.state !== 'in_progress') return null
    const pct = Math.max(0, Math.min(100, modelEvent.overall_pct || 0))
    const done = Math.max(0, modelEvent.bytes_done || 0)
    const total = Math.max(0, modelEvent.bytes_total || 0)
    const transfer = total > 0 ? `${formatBytes(done)} / ${formatBytes(total)}` : `${formatBytes(done)}`
    return `${pct}% • ${transfer}`
  })()

  const handleDownloadSelectedModel = async (): Promise<void> => {
    if (!selectedModelFilename || selectedModelInstalled || modelDownloadPending) return
    setModelDownloadPending(true)
    setModelDownloadError(null)
    setModelEvent((prev) => ({
      state: 'in_progress',
      stage: 'queued',
      model_filename: selectedModelFilename,
      overall_pct: prev?.overall_pct ?? 0,
      bytes_done: prev?.bytes_done ?? 0,
      bytes_total: prev?.bytes_total ?? 0,
      speed_bps: prev?.speed_bps ?? 0,
      eta_sec: prev?.eta_sec ?? null,
      paused: false,
      error: null,
    }))
    try {
      await downloadModel(selectedModelFilename)
      const event = await getModelOperationEvents()
      setModelEvent(event)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setModelDownloadError(getFriendlyModelDownloadError(message))
    } finally {
      setModelDownloadPending(false)
    }
  }

  const handleCancelModelDownload = async (): Promise<void> => {
    if (!modelDownloadInProgress || modelDownloadPending) return
    setModelDownloadPending(true)
    setModelDownloadError(null)
    setModelEvent((prev) => ({
      state: 'cancelled',
      stage: 'cancelled',
      model_filename: prev?.model_filename ?? selectedModelFilename,
      overall_pct: 0,
      bytes_done: 0,
      bytes_total: 0,
      speed_bps: 0,
      eta_sec: null,
      paused: false,
      error: null,
    }))
    try {
      await cancelModelDownload()
      await refreshModelsCatalog()
      const event = await getModelOperationEvents()
      setModelEvent(event)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setModelDownloadError(getFriendlyModelDownloadError(message))
    } finally {
      setModelDownloadPending(false)
    }
  }

  const profile = effectiveProfile
  const diagnosticsPreset = diagnosticsPresetValues[form.diagnostics_profile || '']
  const diagnosticsProfileRows = diagnosticsPreset
    ? [
        { label: 'Profile', value: form.diagnostics_profile },
        { label: 'Log level', value: diagnosticsPreset.log_level },
        { label: 'Per-chat trace logging', value: diagnosticsPreset.chat_trace_logging ? 'on' : 'off' },
        { label: 'Trace redaction', value: diagnosticsPreset.chat_trace_redaction_mode },
        {
          label: 'Trace retention (user / eval)',
          value: `${diagnosticsPreset.chat_trace_user_retention_days} / ${diagnosticsPreset.chat_trace_evaluation_retention_days} days`,
        },
      ]
    : [
        { label: 'Profile', value: 'custom' },
        { label: 'Log level', value: form.log_level ?? '--' },
        { label: 'Per-chat trace logging', value: form.chat_trace_logging ? 'on' : 'off' },
        { label: 'Trace redaction', value: form.chat_trace_redaction_mode ?? '--' },
        {
          label: 'Trace retention (user / eval)',
          value: `${form.chat_trace_user_retention_days ?? '--'} / ${form.chat_trace_evaluation_retention_days ?? '--'} days`,
        },
      ]

  const sectionClass = (isVisible: boolean) => (
    `settings-section${isVisible ? '' : ' settings-section--hidden'}`
  )

  return (
    <div className="settings-view">
      <div className="settings-tabs" role="tablist" aria-label="Settings Sections">
        {SETTINGS_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`settings-tab${activeTab === tab.id ? ' settings-tab--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <i className={tab.icon} aria-hidden="true" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="settings-content">
        <section className={sectionClass(activeTab === 'general')}>
          <div className="settings-section-header ui-title ui-title--section">
            <i className="ri-home-gear-line section-icon" aria-hidden="true" />
            General
          </div>
          <p className="settings-section-description ui-description">
            Core application preferences including privacy and appearance.
          </p>

          <div className="settings-subsection">
            <div className="settings-subsection-head ui-subsection-head">
              <div className="settings-subsection-title ui-subsection-title">
                Full Privacy Mode
              </div>
              <p className="settings-subsection-description ui-subsection-description">
                All processing stays on this computer with no network access. Requires restart.
              </p>
            </div>
            <label className="settings-checkbox-row">
              <input
                type="checkbox"
                checked={form.full_privacy ?? true}
                onChange={(e) => update('full_privacy', e.target.checked)}
              />
              <div><span className="settings-checkbox-row-label">Enable</span></div>
            </label>
          </div>
          <div className="settings-subsection">
            <div className="settings-subsection-head ui-subsection-head">
              <div className="settings-subsection-title ui-subsection-title">
                <i className="ri-contrast-2-line subsection-icon ui-subsection-icon" aria-hidden="true" />
                Theme
              </div>
              <p className="settings-subsection-description ui-subsection-description">Choose the accent color. Preview instantly; save to persist.</p>
            </div>
            <select
              className="settings-select"
              value={form.ui_theme ?? UI_THEME_DEFAULT}
              onChange={(e) => {
                const value = e.target.value
                update('ui_theme', value)
                document.documentElement.setAttribute('data-accent', value)
                try {
                  localStorage.setItem(UI_THEME_STORAGE_KEY, value)
                } catch {
                  /* ignore */
                }
              }}
            >
              {UI_THEME_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="settings-subsection">
            <div className="settings-subsection-head ui-subsection-head">
              <div className="settings-subsection-title ui-subsection-title">
                <i className="ri-layout-top-2-line subsection-icon ui-subsection-icon" aria-hidden="true" />
                Menu Bar Icon
              </div>
              <p className="settings-subsection-description ui-subsection-description">
                Show the Informity AI icon in the macOS menu bar while the app is running.
              </p>
            </div>
            <label className="settings-checkbox-row">
              <input
                type="checkbox"
                checked={form.enable_menu_bar_icon ?? false}
                onChange={(e) => update('enable_menu_bar_icon', e.target.checked)}
              />
              <div><span className="settings-checkbox-row-label">Enable menu bar icon</span></div>
            </label>
          </div>
        </section>

        <section className={sectionClass(activeTab === 'chat')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-chat-ai-4-line section-icon" aria-hidden="true" />
          Chat
        </div>
        <p className="settings-section-description ui-description">Conversation context and default chat settings.</p>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-chat-settings-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Default Chat Mode
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Sets the default mode for new chats. You can still switch modes in the chat composer.
            </p>
          </div>
          <select
            className="settings-select"
            value={form.default_chat_mode}
            onChange={(e) => update('default_chat_mode', isChatMode(e.target.value) ? e.target.value : 'researcher')}
          >
            {CHAT_MODE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <label className="settings-checkbox-row" style={{ marginTop: '0.75rem' }}>
            <input
              type="checkbox"
              checked={form.enable_chat_roles ?? false}
              onChange={(e) => update('enable_chat_roles', e.target.checked)}
            />
            <div>
              <span className="settings-checkbox-row-label">
                Enable roles
                <span className="settings-checkbox-row-info ui-tooltip-trigger">
                  <i className="ri-information-line" aria-hidden="true" />
                  <span className="settings-tooltip ui-tooltip">
                    Add role-based guidance to responses (best with scoped documents)
                  </span>
                </span>
              </span>
            </div>
          </label>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-message-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Conversation Memory
            </div>
            <p className="settings-subsection-description ui-subsection-description">How many recent messages are kept for context in new replies. Higher values improve continuity but may slow responses.</p>
          </div>
          <div className="settings-slider-row">
            <span className="settings-slider-min">0</span>
            <span className="settings-slider-label">
              Messages: <span className="settings-slider-current">{form.chat_history_messages ?? 5}</span>
            </span>
            <span className="settings-slider-max">10</span>
          </div>
          <input
            type="range"
            className="settings-slider"
            min={0}
            max={10}
            step={1}
            value={form.chat_history_messages ?? 5}
            onChange={(e) => update('chat_history_messages', clamp(parseInteger(e.target.value, 5), 0, 10))}
          />
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-archive-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Activity Logs
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Save a local log of each chat session to help diagnose issues if something goes wrong.
            </p>
          </div>
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.chat_trace_logging ?? false}
              onChange={(e) => update('chat_trace_logging', e.target.checked)}
            />
            <div>
              <span className="settings-checkbox-row-label">
                Save chat activity logs
                <span className="settings-checkbox-row-info ui-tooltip-trigger">
                  <i className="ri-information-line" aria-hidden="true" />
                  <span className="settings-tooltip ui-tooltip">
                    Saves chat diagnostics locally on your device to help with troubleshooting. Disabled by default.
                  </span>
                </span>
              </span>
            </div>
          </label>
        </div>

        </section>

        <section className={sectionClass(activeTab === 'chat')}>
        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-search-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Web Search Provider
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Assistant web search is powered by third-party services <a className="settings-link" href="https://app.tavily.com/" target="_blank" rel="noreferrer">Tavily</a> and <a className="settings-link" href="https://www.linkup.so/" target="_blank" rel="noreferrer">Linkup</a>.
            </p>
          </div>
          <div className="settings-control-group">
            <label className="settings-control-label" htmlFor="settings-tavily-api-key">
              Tavily API key
              <span className="settings-checkbox-row-info ui-tooltip-trigger">
                <i className="ri-information-line" aria-hidden="true" />
                <span className="settings-tooltip ui-tooltip ui-tooltip--up-right">
                  1000 free queries per month. Pricing subject to change.
                </span>
              </span>
            </label>
            <div className="settings-input-wrap">
              <input
                id="settings-tavily-api-key"
                type="password"
                className="settings-input settings-input--with-clear"
                placeholder="tvly-..."
                value={form.tavily_api_key}
                onChange={(e) => {
                  update('tavily_api_key', e.target.value)
                  if (form.clear_tavily_api_key) update('clear_tavily_api_key', false)
                }}
                onFocus={() => {
                  if (isMaskedTavilyKey(form.tavily_api_key)) {
                    update('tavily_api_key', '')
                  }
                }}
                onBlur={() => {
                  if (
                    !String(form.tavily_api_key || '').trim()
                    && settings.tavily_api_key_set
                    && !form.clear_tavily_api_key
                  ) {
                    update('tavily_api_key', MASKED_TAVILY_KEY)
                  }
                }}
                autoComplete="off"
              />
              {(settings.tavily_api_key_set || String(form.tavily_api_key || '').trim().length > 0) && (
                <button
                  type="button"
                  className="settings-input-clear"
                  aria-label="Clear Tavily API key"
                  title="Clear API key"
                  onClick={() => {
                    update('tavily_api_key', '')
                    update('clear_tavily_api_key', true)
                  }}
                >
                  <i className="ri-close-line" aria-hidden />
                </button>
              )}
            </div>
          </div>
          <div className="settings-control-group">
            <label className="settings-control-label" htmlFor="settings-linkup-api-key">
              Linkup API key
              <span className="settings-checkbox-row-info ui-tooltip-trigger">
                <i className="ri-information-line" aria-hidden="true" />
                <span className="settings-tooltip ui-tooltip ui-tooltip--up-right">
                  1000 standard or 100 deep free queries per month. Pricing subject to change.
                </span>
              </span>
            </label>
            <div className="settings-input-wrap">
              <input
                id="settings-linkup-api-key"
                type="password"
                className="settings-input settings-input--with-clear"
                placeholder="sk-..."
                value={form.linkup_api_key}
                onChange={(e) => {
                  update('linkup_api_key', e.target.value)
                  if (form.clear_linkup_api_key) update('clear_linkup_api_key', false)
                }}
                onFocus={() => {
                  if (isMaskedLinkupKey(form.linkup_api_key)) {
                    update('linkup_api_key', '')
                  }
                }}
                onBlur={() => {
                  if (
                    !String(form.linkup_api_key || '').trim()
                    && settings.linkup_api_key_set
                    && !form.clear_linkup_api_key
                  ) {
                    update('linkup_api_key', MASKED_LINKUP_KEY)
                  }
                }}
                autoComplete="off"
              />
              {(settings.linkup_api_key_set || String(form.linkup_api_key || '').trim().length > 0) && (
                <button
                  type="button"
                  className="settings-input-clear"
                  aria-label="Clear Linkup API key"
                  title="Clear API key"
                  onClick={() => {
                    update('linkup_api_key', '')
                    update('clear_linkup_api_key', true)
                  }}
                >
                  <i className="ri-close-line" aria-hidden />
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-menu-search-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Web Search Performance
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Configure how many web search results to fetch and how long each request can run.
            </p>
          </div>
          <div className="settings-control-group">
            <label className="settings-control-label" htmlFor="settings-web-search-max-results">Results per search</label>
            <input
              id="settings-web-search-max-results"
              type="number"
              className="settings-input settings-input--number"
              min={1}
              max={10}
              step={1}
              value={form.web_search_max_results}
              onChange={(e) => update('web_search_max_results', clamp(parseInteger(e.target.value, 5), 1, 10))}
            />
          </div>
          <div className="settings-control-group">
            <label className="settings-control-label" htmlFor="settings-web-search-timeout">
              {renderLabelWithMutedParens('Search timeout (seconds)')}
            </label>
            <input
              id="settings-web-search-timeout"
              type="number"
              className="settings-input settings-input--number"
              min={1}
              max={30}
              step={1}
              value={form.web_search_timeout_seconds}
              onChange={(e) => update('web_search_timeout_seconds', clamp(parseInteger(e.target.value, 8), 1, 30))}
            />
          </div>
        </div>
        </section>

        <section className={sectionClass(activeTab === 'data')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-folder-line section-icon" aria-hidden="true" />
          Data Sources
        </div>
        <p className="settings-section-description ui-description">
          Choose which folders and file types the application scans and makes searchable.
        </p>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-folders-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Source Directories
            </div>
            <p className="settings-subsection-description ui-subsection-description">The application will scan these folders and index their contents.</p>
          </div>
          <div className="settings-add-row">
            <input
              type="text"
              className="settings-input"
              placeholder="e.g. /Users/you/Documents"
              value={dirInput}
              onChange={(e) => setDirInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && canAddDir && (e.preventDefault(), addDir())}
            />
            {isDesktopRuntime() && (
              <button type="button" className="settings-btn settings-btn--add" onClick={browseDir}>
                <i className="ri-folder-open-line" aria-hidden />
                <span>Browse</span>
              </button>
            )}
            <button type="button" className="settings-btn settings-btn--add" onClick={addDir} disabled={!canAddDir}>
              + Add
            </button>
          </div>
          {(form.watched_directories?.length ?? 0) > 0 && (
            <div className="settings-list-scroll">
              <ul className="settings-list">
                {(form.watched_directories || []).map((path) => (
                  <li key={path} className="settings-list__item">
                    <span className="settings-list__text">{path}</span>
                    <button
                      type="button"
                      className="settings-list__remove"
                      onClick={() => removeDir(path)}
                      title="Remove"
                      aria-label={`Remove ${path}`}
                    >
                      <i className="ri-close-line" aria-hidden style={{ fontSize: '0.875rem' }} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <label className="settings-checkbox-row settings-checkbox-row--spaced">
            <input
              type="checkbox"
              checked={form.follow_symlinks ?? false}
              onChange={(e) => update('follow_symlinks', e.target.checked)}
            />
            <div><span className="settings-checkbox-row-label">Follow symbolic links during scanning</span></div>
          </label>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-file-copy-2-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              File Categories to Index
            </div>
            <p className="settings-subsection-description ui-subsection-description">Only checked file types will be scanned and indexed.</p>
          </div>
          <div className="settings-file-types">
            {orderedFileTypeOptions.map((opt) => {
              const exts = opt.extensions || []
              const current = form.supported_extensions || []
              const allChecked = exts.length > 0 && exts.every((e) => current.includes(e))
              return (
                <label key={opt.id} className="settings-file-type">
                  <input
                    type="checkbox"
                    checked={allChecked}
                    onChange={() => {
                      const next = allChecked
                        ? current.filter((e) => !exts.includes(e))
                        : [...new Set([...current, ...exts])]
                      update('supported_extensions', next)
                    }}
                  />
                  <span>{opt.label}</span>
                </label>
              )
            })}
          </div>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-filter-off-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Exclude Patterns
            </div>
            <p className="settings-subsection-description ui-subsection-description">Glob patterns for files and folders to skip, in addition to the macOS and developer exclusions below.</p>
          </div>
          <div className="settings-add-row">
            <input
              type="text"
              className="settings-input"
              placeholder="e.g. *.log, backups"
              value={ignoreInput}
              onChange={(e) => setIgnoreInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && canAddIgnore && (e.preventDefault(), addIgnore())}
            />
            <button type="button" className="settings-btn settings-btn--add" onClick={addIgnore} disabled={!canAddIgnore}>
              + Add
            </button>
          </div>
          {(form.ignore_patterns?.length ?? 0) > 0 && (
            <div className="settings-list-scroll settings-list-scroll--pills">
              <ul className="settings-list settings-list--pills">
                {(form.ignore_patterns || []).map((p) => (
                  <li key={p} className="settings-list__item settings-list__item--pill">
                    <span className="settings-list__text">{p}</span>
                    <button
                      type="button"
                      className="settings-list__remove"
                      onClick={() => removeIgnore(p)}
                      aria-label={`Remove ${p}`}
                    >
                      <i className="ri-close-line" aria-hidden style={{ fontSize: '0.75rem' }} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.exclude_macos_system ?? true}
              onChange={(e) => update('exclude_macos_system', e.target.checked)}
            />
            <div>
              <span className="settings-checkbox-row-label">
                Exclude common macOS system and application data
                <span className="settings-checkbox-row-info ui-tooltip-trigger">
                  <i className="ri-information-line" aria-hidden="true" />
                  <span className="settings-tooltip ui-tooltip">
                    Excludes common macOS system files and directories such as .DS_Store, .Trash, Library/Caches, Library/Logs, and other system-generated content.
                  </span>
                </span>
              </span>
            </div>
          </label>
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.exclude_developer_data ?? true}
              onChange={(e) => update('exclude_developer_data', e.target.checked)}
            />
            <div>
              <span className="settings-checkbox-row-label">
                Exclude common developer data
                <span className="settings-checkbox-row-info ui-tooltip-trigger">
                  <i className="ri-information-line" aria-hidden="true" />
                  <span className="settings-tooltip ui-tooltip">
                    Excludes common developer directories and files such as .git, node_modules, __pycache__, .venv, dist, build, and other development artifacts.
                  </span>
                </span>
              </span>
            </div>
          </label>
        </div>
        </section>

        <section className={sectionClass(activeTab === 'indexing')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-stack-line section-icon" aria-hidden="true" />
          Indexing
        </div>
        <p className="settings-section-description ui-description">
          Controls how the application reads and prepares your files for search and chat.
        </p>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-arrow-right-down-box-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Document Extraction
            </div>
            <p className="settings-subsection-description ui-subsection-description">Options for extracting text from documents, including image-only PDFs.</p>
          </div>
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.enable_ocr_for_images ?? true}
              onChange={(e) => update('enable_ocr_for_images', e.target.checked)}
            />
            <div><span className="settings-checkbox-row-label">Enable OCR for scanned documents</span></div>
          </label>
          <div className="settings-subsection-field">
            <label htmlFor="max-indexable-file-size" className="settings-subsection-field-label">
              Max Indexable File Size <span className="settings-subsection-field-unit">(MB)</span>
              <span className="settings-checkbox-row-info ui-tooltip-trigger">
                <i className="ri-information-line" aria-hidden="true" />
                <span className="settings-tooltip ui-tooltip">Files exceeding this limit are skipped during indexing and data extraction. Maximum allowed: 500 MB.</span>
              </span>
            </label>
            <input
              id="max-indexable-file-size"
              type="number"
              className="settings-input settings-input--number"
              min={1}
              max={500}
              step={10}
              value={form.max_indexable_file_size_mb ?? 100}
              onChange={(e) => update('max_indexable_file_size_mb', clamp(parseInteger(e.target.value, 100), 1, 500))}
            />
          </div>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-fingerprint-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Entity Extraction
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Control which entity types are identified and indexed alongside your documents.
            </p>
          </div>
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.entity_extract_acronym ?? true}
              onChange={(e) => update('entity_extract_acronym', e.target.checked)}
            />
            <div><span className="settings-checkbox-row-label">Enable extraction of acronyms</span></div>
          </label>
          <label className="settings-checkbox-row">
            <input
              type="checkbox"
              checked={form.entity_extract_person_name ?? false}
              onChange={(e) => update('entity_extract_person_name', e.target.checked)}
            />
            <div><span className="settings-checkbox-row-label">Enable extraction of person names</span></div>
          </label>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-speed-up-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Performance
            </div>
            <p className="settings-subsection-description ui-subsection-description">Trade off indexing speed against keeping your Mac responsive. Requires restart.</p>
          </div>
          <div className="settings-slider-row">
            <span className="settings-slider-min">Responsive</span>
            <span className="settings-slider-label">
              <span className="settings-slider-current">{INDEXING_SPEED_LABELS[speedVal] || 'Balanced'}</span>
            </span>
            <span className="settings-slider-max">Fastest</span>
          </div>
          <input
            type="range"
            className="settings-slider"
            min={1}
            max={5}
            step={1}
            value={speedVal}
            onChange={handleSpeedChange}
          />
          <div className="settings-subsection-field">
            <label htmlFor="scan-file-timeout" className="settings-subsection-field-label">
              File Processing Timeout <span className="settings-subsection-field-unit">(seconds)</span>
              <span className="settings-checkbox-row-info ui-tooltip-trigger">
                <i className="ri-information-line" aria-hidden="true" />
                <span className="settings-tooltip ui-tooltip">Per-file time limit for extraction. Increase if you have large or complex documents. Maximum allowed: 600 seconds.</span>
              </span>
            </label>
            <input
              id="scan-file-timeout"
              type="number"
              className="settings-input settings-input--number"
              min={1}
              max={600}
              value={form.scan_file_timeout_seconds ?? 600}
              onChange={(e) => update('scan_file_timeout_seconds', clamp(parseInteger(e.target.value, 600), 1, 600))}
            />
          </div>
        </div>

        </section>

        <section className={sectionClass(activeTab === 'models')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-robot-2-line section-icon" aria-hidden="true" />
          Models
        </div>
        <p className="settings-section-description ui-description">Select the AI model to use and view its capabilities.</p>

        {profile && (
          <>
            <div className="settings-subsection settings-subsection--profile">
              <div className="settings-subsection-head ui-subsection-head">
                <div className="settings-subsection-title ui-subsection-title">
                  <i className="ri-chat-ai-4-line subsection-icon ui-subsection-icon" aria-hidden="true" />
                  Main Model
                </div>
                <p className="settings-subsection-description ui-subsection-description">The AI model used for query classifications and chat responses. Requires restart.</p>
              </div>
            </div>
            <div className="settings-control-group">
              <div className="settings-add-row settings-add-row--model">
                <select
                  id="settings-llm-model"
                  aria-label="Main model"
                  className="settings-select"
                  value={selectedModelFilename}
                  onChange={(e) => {
                    update('llm_model_filename', canonicalizeModelFilename(e.target.value))
                    setModelDownloadError(null)
                  }}
                >
                  {knownModelFilenames.map((modelName) => {
                    const catalogEntry = catalogModels.find((model) => model.model_filename === modelName)
                    const installed = installedModelSet.has(modelName)
                    const baseLabel = catalogEntry?.display_name || modelProfileNames.get(modelName) || modelName
                    const suffix = installed ? '' : ' (Not installed)'
                    return (
                      <option key={modelName} value={modelName}>
                        {`${baseLabel}${suffix}`}
                      </option>
                    )
                  })}
                </select>
                {!selectedModelInstalled && (
                  <>
                    <button
                      type="button"
                      className={`settings-btn settings-btn--add${modelDownloadInProgress ? ' settings-btn--add-cancel' : ''}`}
                      onClick={() => {
                        if (modelDownloadInProgress) {
                          void handleCancelModelDownload()
                          return
                        }
                        void handleDownloadSelectedModel()
                      }}
                      disabled={modelDownloadPending || !selectedModelFilename}
                    >
                      {modelDownloadPending ? 'Working...' : (modelDownloadInProgress ? 'Cancel' : '+ Add')}
                    </button>
                    {modelProgressSummary && (
                      <span className="settings-model-progress-inline">{modelProgressSummary}</span>
                    )}
                  </>
                )}
              </div>
              {modelDownloadError && (
                <p className="settings-field-hint">{modelDownloadError}</p>
              )}
            </div>
            <div className="settings-profile-grid">
              <ProfileRow label="Profile" value={profile.name} />
              <ProfileRow label="Model" value={selectedModelFilename} />
              <ProfileRow label="Family" value={profile.family} />
              <ProfileRow label="Reasoning" value={profile.reasoning_mode} />
              <ProfileRow label="Max tokens" value={profile.max_tokens ?? '--'} />
              <ProfileRow label="Context length" value={profile.context_length ?? '--'} />
              <ProfileRow label="Temperature" value={profile.temperature ?? '--'} />
              <ProfileRow label="Retrieval (top-k)" value={profile.rag_top_k ?? '--'} />
              <ProfileRow label="Document matching threshold" value={profile.rag_max_score ?? '--'} />
              <ProfileRow label="Context ratio" value={profile.rag_context_ratio ?? '--'} />
              <ProfileRow label="Model size" value={formatModelSizeGb(selectedCatalogEntry?.model_size_bytes)} />
            </div>
          </>
        )}
        </section>

        <section className={sectionClass(activeTab === 'diagnostics')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-pulse-line section-icon" aria-hidden="true" />
          Diagnostics & Observability
        </div>
        <p className="settings-section-description ui-description">Logging and diagnostics settings. Use these if something seems off or you need to share details with support.</p>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-list-settings-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Diagnostics Profile
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Quick presets for logging and diagnostics behavior. Troubleshooting enables richer diagnostics with bounded retention.
            </p>
          </div>
          <select
            className="settings-select"
            value={form.diagnostics_profile ?? 'standard'}
            onChange={(e) => applyDiagnosticsProfile(e.target.value)}
          >
            {DIAGNOSTICS_PROFILE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <div className="settings-profile-grid">
            {diagnosticsProfileRows.map((row) => (
              <ProfileRow key={row.label} label={row.label} value={row.value} />
            ))}
          </div>
        </div>

        {form.diagnostics_profile === 'custom' && (
          <div className="settings-subsection">
            <div className="settings-subsection-head ui-subsection-head">
              <div className="settings-subsection-title ui-subsection-title">
                <i className="ri-tools-line subsection-icon ui-subsection-icon" aria-hidden="true" />
                Advanced Diagnostics
              </div>
              <p className="settings-subsection-description ui-subsection-description">
                Fine-grained controls for the Custom diagnostics profile.
              </p>
            </div>
            <div className="settings-control-group">
              <label className="settings-control-label" htmlFor="settings-log-level">Log level</label>
              <select
                id="settings-log-level"
                className="settings-select"
                value={form.log_level ?? 'info'}
                onChange={(e) => updateDiagnosticsControl('log_level', e.target.value)}
              >
                {LOG_LEVEL_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="settings-control-group">
              <label className="settings-control-label" htmlFor="settings-trace-redaction">Trace redaction</label>
              <select
                id="settings-trace-redaction"
                className="settings-select"
                value={form.chat_trace_redaction_mode ?? 'minimal'}
                onChange={(e) => updateDiagnosticsControl('chat_trace_redaction_mode', e.target.value)}
              >
                {TRACE_REDACTION_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="settings-control-group">
              <label className="settings-control-label" htmlFor="settings-trace-retention-user">
                {renderLabelWithMutedParens('User trace retention (days)')}
              </label>
              <input
                id="settings-trace-retention-user"
                type="number"
                className="settings-input settings-input--number"
                min={0}
                max={3650}
                value={form.chat_trace_user_retention_days ?? 30}
                onChange={(e) => updateDiagnosticsControl('chat_trace_user_retention_days', clamp(parseInteger(e.target.value, 0), 0, 3650))}
              />
            </div>
            <div className="settings-control-group">
              <label className="settings-control-label" htmlFor="settings-trace-retention-eval">
                {renderLabelWithMutedParens('Evaluation trace retention (days)')}
              </label>
              <input
                id="settings-trace-retention-eval"
                type="number"
                className="settings-input settings-input--number"
                min={0}
                max={3650}
                value={form.chat_trace_evaluation_retention_days ?? 30}
                onChange={(e) => updateDiagnosticsControl('chat_trace_evaluation_retention_days', clamp(parseInteger(e.target.value, 0), 0, 3650))}
              />
            </div>
          </div>
        )}
        </section>

        <section className={sectionClass(activeTab === 'system')}>
        <div className="settings-section-header ui-title ui-title--section">
          <i className="ri-server-line section-icon" aria-hidden="true" />
          System
        </div>
        <p className="settings-section-description ui-description">General application utilities and configuration references.</p>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-cpu-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              CPU Responsiveness
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Controls CPU threads used by chat generation. Lower values keep the system more responsive. Requires restart.
            </p>
          </div>
          <div className="settings-slider-row">
            <span className="settings-slider-min">Responsive</span>
            <span className="settings-slider-label">
              <span className="settings-slider-current">{CHAT_CPU_RESPONSIVENESS_LABELS[chatCpuVal] || 'Balanced'}</span>
              {' '}({form.llm_cpu_threads ?? 4} threads)
            </span>
            <span className="settings-slider-max">Fastest</span>
          </div>
          <input
            type="range"
            className="settings-slider"
            min={1}
            max={3}
            step={1}
            value={chatCpuVal}
            onChange={handleChatCpuChange}
          />
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-keyboard-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Keyboard Shortcuts
            </div>
            <p className="settings-subsection-description ui-subsection-description">
              Press <kbd>{typeof navigator !== 'undefined' && navigator.platform?.toLowerCase().includes('mac') ? '⌘/' : 'Ctrl+/'}</kbd> to view all{' '}
              <button
                type="button"
                className="settings-link settings-link--button"
                onClick={() => window.dispatchEvent(new CustomEvent('open-keyboard-shortcuts'))}
              >
                keyboard shortcuts
              </button>
            </p>
          </div>
        </div>

        <div className="settings-subsection">
          <div className="settings-subsection-head ui-subsection-head">
            <div className="settings-subsection-title ui-subsection-title">
              <i className="ri-file-settings-line subsection-icon ui-subsection-icon" aria-hidden="true" />
              Application Configuration
            </div>
            <p className="settings-subsection-description ui-subsection-description"><Link to="/settings/configuration" className="settings-link">Environment variables</Link> and configuration file reference.</p>
          </div>
        </div>
        </section>

        {activeTab === 'system' && (
          <>
            <div className="settings-reset-card ui-card ui-card--warning">
              <div className="settings-reset-card-title ui-card__title">
                <i className="ri-alert-line" aria-hidden="true" /> Reset Settings
              </div>
              <p className="settings-reset-card-description ui-card__description">
                Reset all application settings to their default values. Your indexed files, chat history, and models will remain unchanged.
              </p>
              <button type="button" className="settings-btn settings-btn--warning" onClick={onResetSettings}>
                <i className="ri-restart-line" aria-hidden="true" /> Reset Settings
              </button>
            </div>

            <div className="settings-danger-card ui-card ui-card--danger">
              <div className="settings-danger-card-title ui-card__title">
                <i className="ri-error-warning-line" aria-hidden="true" /> Danger Zone
              </div>
              <p className="settings-danger-card-description ui-card__description">
                Permanently deletes all indexed data (files, chunks, embeddings, chats) and resets all settings. Your original files and downloaded models will not be affected.
              </p>
              <button type="button" className="settings-btn settings-btn--danger" onClick={onResetIndex}>
                <i className="ri-delete-bin-line" aria-hidden="true" /> Reset All
              </button>
            </div>
          </>
        )}
      </div>

      {activeTab !== 'system' && (
        <div className="settings-actions settings-actions--sticky">
          <button
            type="button"
            className="settings-btn settings-btn--primary"
            onClick={handleSave}
            disabled={!canSaveSettings}
          >
            {saving ? 'Saving…' : 'Save Settings'}
          </button>
          <button type="button" className="settings-btn settings-btn--secondary" onClick={handleDiscard}>
            Discard Changes
          </button>
        </div>
      )}
    </div>
  )
}
