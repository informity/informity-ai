/**
 * Informity AI — Settings page
 * Loads settings, wires Save/Discard/Reset, handles confirmations.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import type { WheelEvent } from 'react'
import {
  getSettings,
  getIndexStatus,
  getModelProfile,
  updateSettings,
  resetSettings,
  resetIndex,
  ApiError,
} from '../api'
import { SettingsView } from '../components/settings/SettingsView'
import { PageHeader } from '../components/PageHeader'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { showToast } from '../context/useToast'
import { useConfirm } from '../context/useConfirm'
import { useBackendStatus } from '../context/useBackendStatus'
import { isBackendConnectionError } from '../utils/networkErrors'
import { proxyWheelToContainer } from '../utils/wheelProxy'
import { normalizeUiTheme, UI_THEME_DEFAULT, UI_THEME_STORAGE_KEY } from '../utils/uiTheme'
import { setMenuBarIconEnabled } from '../tauriRuntime'
import '../pages/PlaceholderPage.css'

const UPDATABLE_KEYS = [
  'watched_directories',
  'ignore_patterns',
  'exclude_macos_system',
  'exclude_developer_data',
  'supported_extensions',
  'follow_symlinks',
  'chunk_size_tokens',
  'chunk_overlap_tokens',
  'embedding_batch_size',
  'embedding_max_threads',
  'llm_cpu_threads',
  'enable_ocr_for_images',
  'scan_file_timeout_seconds',
  'full_privacy',
  'adaptive_rag_tuning',
  'chat_history_messages',
  'llm_model_filename',
  'diagnostics_profile',
  'chat_trace_logging',
  'chat_trace_redaction_mode',
  'chat_trace_user_retention_days',
  'chat_trace_evaluation_retention_days',
  'enable_raw_output_control',
  'log_level',
  'ui_theme',
  'enable_menu_bar_icon',
  'default_response_mode',
] as const

interface FormState {
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
  scan_file_timeout_seconds?: number
  full_privacy?: boolean
  adaptive_rag_tuning?: boolean
  chat_history_messages?: number
  diagnostics_profile?: string
  chat_trace_logging?: boolean
  chat_trace_redaction_mode?: string
  chat_trace_user_retention_days?: number
  chat_trace_evaluation_retention_days?: number
  enable_raw_output_control?: boolean
  log_level?: string
  ui_theme?: string
  enable_menu_bar_icon?: boolean
  default_response_mode?: 'analysis' | 'research'
  llm_model_filename?: string
}

function normalizeSupportedModes(
  modes: Array<'analysis' | 'research'> | string[] | undefined,
): Array<'analysis' | 'research'> {
  if (!Array.isArray(modes)) return ['analysis']
  const filtered: Array<'analysis' | 'research'> = []
  for (const mode of modes) {
    if (mode === 'analysis' || mode === 'research') {
      filtered.push(mode)
    }
  }
  return filtered.length > 0 ? filtered : ['analysis']
}

function buildPayload(form: FormState): Record<string, unknown> {
  const payload: Record<string, unknown> = {}
  for (const key of UPDATABLE_KEYS) {
    if (form[key] !== undefined) {
      payload[key] = form[key]
    }
  }
  return payload
}

interface SettingsData extends FormState {
  file_type_options?: { id: string; label: string; extensions: string[] }[]
}

interface IndexStatusData {
  reset_in_progress?: boolean
  last_reset_result?: {
    error?: string
    storage_compacted?: boolean
    compaction_error?: string | null
  } | null
}

const RESET_POLL_INTERVAL_MS = 500
const RESET_POLL_TIMEOUT_MS = 300000

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

function getErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    const detail = `${err.detail || ''}`.trim()
    if (detail) return detail
    return `HTTP ${err.status}`
  }
  if (err instanceof Error) {
    const message = `${err.message || ''}`.trim()
    if (message) return message
  }
  return fallback
}

export function SettingsPage() {
  const confirm = useConfirm()
  const { offline } = useBackendStatus()
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const resetPollingCancelledRef = useRef(false)
  const pageScrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    resetPollingCancelledRef.current = false
    return () => {
      resetPollingCancelledRef.current = true
    }
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const settingsData = (await getSettings()) as SettingsData
      setSettings(settingsData)
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to load'
      const disconnected = isBackendConnectionError(err)
      setError(msg)
      if (!disconnected) {
        showToast('error', msg)
      }
      setSettings(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const handleSave = async (form: FormState) => {
    setSaving(true)
    try {
      const payload = buildPayload(form)
      const nextModelFilename = String(
        payload.llm_model_filename ?? settings?.llm_model_filename ?? '',
      ).trim()
      const nextResponseMode = payload.default_response_mode
      if (
        nextModelFilename &&
        (nextResponseMode === 'analysis' || nextResponseMode === 'research')
      ) {
        try {
          const profile = await getModelProfile(nextModelFilename) as {
            supported_modes?: Array<'analysis' | 'research'> | string[]
          }
          const supportedModes = normalizeSupportedModes(profile.supported_modes)
          if (!supportedModes.includes(nextResponseMode)) {
            payload.default_response_mode = supportedModes[0] ?? 'analysis'
          }
        } catch {
          // If profile lookup fails, keep user's current mode and let backend validation decide.
        }
      }
      const updated = (await updateSettings(payload)) as SettingsData
      setSettings(updated)
      showToast('success', 'Settings saved')
      if (typeof updated.enable_menu_bar_icon === 'boolean') {
        try {
          await setMenuBarIconEnabled(updated.enable_menu_bar_icon)
        } catch (err) {
          showToast('warning', `Menu bar icon update failed: ${getErrorMessage(err, 'Unknown error')}`)
        }
      }
      if (form.ui_theme) {
        const normalizedTheme = normalizeUiTheme(form.ui_theme) || UI_THEME_DEFAULT
        document.documentElement.setAttribute('data-accent', normalizedTheme)
        try {
          localStorage.setItem(UI_THEME_STORAGE_KEY, normalizedTheme)
        } catch {
          // ignore
        }
      }
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Save failed.'
      showToast('error', msg)
    } finally {
      setSaving(false)
    }
  }

  const handleDiscard = () => {
    // No-op; form state is reset locally
  }

  const handleResetSettings = async () => {
    const ok = await confirm({
      title:       'Reset Settings',
      message:     'Reset all settings to defaults? Your indexed files and chat history will remain unchanged.',
      confirmLabel: 'Reset',
      cancelLabel:  'Cancel',
      icon:       'ri-restart-line',
    })
    if (!ok) return
    setSaving(true)
    try {
      await resetSettings()
      const updated = (await getSettings()) as SettingsData
      setSettings(updated)
      showToast('success', 'Settings reset')
    } catch (err) {
      const msg = getErrorMessage(err, 'Reset failed.')
      showToast('error', msg)
    } finally {
      setSaving(false)
    }
  }

  const handleResetIndex = async () => {
    resetPollingCancelledRef.current = false
    const ok1 = await confirm({
      title:       'Reset All Data',
      message:     'Permanently delete all indexed data (files, chunks, embeddings, chats) and reset settings? This cannot be undone.',
      confirmLabel: 'Continue',
      cancelLabel:  'Cancel',
      variant:      'danger',
      icon:       'ri-delete-bin-line',
    })
    if (!ok1) return
    const ok2 = await confirm({
      title:       'Are You Sure?',
      message:     'Your original files and models will not be affected, but you will need to re-scan.',
      confirmLabel: 'Reset All',
      cancelLabel:  'Cancel',
      variant:      'danger',
      icon:       'ri-delete-bin-line',
    })
    if (!ok2) return
    setSaving(true)
    try {
      try {
        await resetIndex()
      } catch (err) {
        if (resetPollingCancelledRef.current) return
        if (err instanceof ApiError && err.status === 409) {
          const detail = `${err.detail || ''}`.toLowerCase()
          if (detail.includes('reset is already in progress')) {
            const status = (await getIndexStatus()) as IndexStatusData
            if (resetPollingCancelledRef.current) return
            if (status.reset_in_progress) {
              showToast('info', 'Reset already in progress. Waiting for completion...')
            } else {
              throw err
            }
          } else if (detail.includes('already running')) {
            const forceReset = await confirm({
              title:       'Scan Running',
              message:     'A scan is currently running. Stop it and continue with Reset All Data?',
              confirmLabel: 'Stop Scan and Reset',
              cancelLabel:  'Cancel',
              variant:      'danger',
              icon:       'ri-close-circle-line',
            })
            if (!forceReset) {
              setSaving(false)
              return
            }
            await resetIndex(true)
            showToast('info', 'Stopping scan and starting reset...')
          } else {
            throw err
          }
        } else {
          throw err
        }
      }
      const deadline = Date.now() + RESET_POLL_TIMEOUT_MS
      let completed = false
      let finalStatus: IndexStatusData | null = null
      while (Date.now() < deadline) {
        if (resetPollingCancelledRef.current) return
        try {
          const status = (await getIndexStatus()) as IndexStatusData
          if (resetPollingCancelledRef.current) return
          if (!status.reset_in_progress) {
            finalStatus = status
            completed = true
            break
          }
        } catch {
          // Reset task may briefly make status unavailable; keep polling.
        }
        if (resetPollingCancelledRef.current) return
        await sleep(RESET_POLL_INTERVAL_MS)
      }
      if (resetPollingCancelledRef.current) return
      if (!completed) {
        showToast('warning', 'Reset is still running. Settings will refresh automatically when it completes.')
        await load()
        return
      }
      const resetResult = finalStatus?.last_reset_result
      const resetError = `${resetResult?.error || ''}`.trim()
      if (resetError) {
        showToast('error', `Reset failed: ${resetError}`)
        await load()
        return
      }
      const compactionError = `${resetResult?.compaction_error || ''}`.trim()
      const updated = (await getSettings()) as SettingsData
      if (resetPollingCancelledRef.current) return
      setSettings(updated)
      const completionMessage =
        compactionError || resetResult?.storage_compacted === false
          ? 'All data reset.\nDatabase reset in progress...'
          : 'All data reset'
      showToast('success', completionMessage)
    } catch (err) {
      if (resetPollingCancelledRef.current) return
      const msg = getErrorMessage(err, 'Reset failed.')
      showToast('error', msg)
    } finally {
      if (!resetPollingCancelledRef.current) {
        setSaving(false)
      }
    }
  }

  useEffect(() => {
    const normalizedTheme = normalizeUiTheme(settings?.ui_theme)
    if (normalizedTheme) {
      document.documentElement.setAttribute('data-accent', normalizedTheme)
      try {
        localStorage.setItem(UI_THEME_STORAGE_KEY, normalizedTheme)
      } catch {
        // ignore
      }
    }
  }, [settings?.ui_theme])

  useEffect(() => {
    if (typeof settings?.enable_menu_bar_icon !== 'boolean') {
      return
    }
    setMenuBarIconEnabled(settings.enable_menu_bar_icon).catch((err) => {
      console.warn('menu bar icon startup sync failed', err)
    })
  }, [settings?.enable_menu_bar_icon])

  const handlePageWheel = useCallback((e: WheelEvent<HTMLDivElement>) => {
    const pageScroll = pageScrollRef.current
    const settingsContent = pageScroll?.querySelector('.settings-content') as HTMLElement | null
    const target = settingsContent && settingsContent.scrollHeight > settingsContent.clientHeight
      ? settingsContent
      : pageScroll
    proxyWheelToContainer(e, target)
  }, [])

  if (loading) {
    return (
      <div className="page" onWheel={handlePageWheel}>
        <PageHeader
          title="Settings"
          subtitle="Loading settings..."
          icon="ri-settings-3-line"
        />
        <div className="page__scroll" ref={pageScrollRef}>
          <p>Loading settings...</p>
        </div>
      </div>
    )
  }

  if (offline || error) {
    return (
      <div className="page" onWheel={handlePageWheel}>
        <PageHeader
          title="Settings"
          subtitle="Manage application preferences and behavior."
          icon="ri-settings-3-line"
        />
        <div className="page__scroll" ref={pageScrollRef}>
          {offline ? <ServiceUnavailableState /> : <p className="page__error">{error}</p>}
        </div>
      </div>
    )
  }

  return (
    <div className="page" onWheel={handlePageWheel}>
      <PageHeader
        title="Settings"
        subtitle="Manage application preferences and behavior."
        icon="ri-settings-3-line"
      />
      <div className="page__scroll" ref={pageScrollRef}>
        <SettingsView
          settings={settings}
          fileTypeOptions={settings?.file_type_options ?? []}
          onSave={handleSave}
          onDiscard={handleDiscard}
          onResetSettings={handleResetSettings}
          onResetIndex={handleResetIndex}
          saving={saving}
        />
      </div>
    </div>
  )
}
