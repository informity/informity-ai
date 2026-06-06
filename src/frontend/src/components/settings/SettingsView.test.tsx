import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SettingsView } from './SettingsView'
import { SETTINGS_ACTIVE_TAB_STORAGE_KEY } from '../../utils/storageKeys'

vi.mock('../../api', () => ({
  getRoles: vi.fn(async () => [
    { id: 'legal', name: 'Legal', description: 'Legal role', icon: 'ri-scales-3-line' },
    { id: 'security_compliance', name: 'Security & Compliance', description: 'Security role', icon: 'ri-shield-check-line' },
  ]),
  getModelProfile: vi.fn(async () => ({})),
  getModelsCatalog: vi.fn(async () => ({
    default_model_filename: 'main.gguf',
    models: [
      {
        tier: 'small',
        title: 'Small',
        model_filename: 'main.gguf',
        approx_size_gb: 5.5,
        quality: 'Good',
        speed: 'Fast',
        ram_profile: 'Lower RAM',
        description: 'Fastest setup with lower memory footprint.',
        installed: true,
        is_default: true,
      },
      {
        tier: 'balanced',
        title: 'Balanced',
        model_filename: 'alt.gguf',
        approx_size_gb: 9.8,
        quality: 'High',
        speed: 'Balanced',
        ram_profile: 'Medium RAM',
        description: 'Recommended quality and speed tradeoff.',
        installed: true,
        is_default: false,
      },
    ],
  })),
  getModelOperationEvents: vi.fn(async () => ({
    state: 'idle',
    stage: 'idle',
    model_filename: null,
    overall_pct: 0,
    bytes_done: 0,
    bytes_total: 0,
    speed_bps: 0,
    eta_sec: null,
    paused: false,
    error: null,
  })),
  downloadModel: vi.fn(async () => ({ accepted: true, detail: 'ok' })),
  cancelModelDownload: vi.fn(async () => ({ accepted: true, detail: 'ok' })),
  removeModel: vi.fn(async () => ({ accepted: true, detail: 'ok' })),
}))

afterEach(() => {
  cleanup()
  localStorage.clear()
})

const baseSettings = {
  watched_directories: ['/tmp/docs'],
  ignore_patterns: ['*.log'],
  exclude_macos_system: true,
  exclude_developer_data: true,
  supported_extensions: ['.md', '.txt'],
  follow_symlinks: false,
  chunk_size_tokens: 512,
  chunk_overlap_tokens: 60,
  embedding_batch_size: 32,
  embedding_max_threads: 6,
  llm_cpu_threads: 4,
  enable_ocr_for_images: true,
  scan_file_timeout_seconds: 300,
  full_privacy: true,
  adaptive_rag_tuning: true,
  chat_history_messages: 5,
  enable_chat_roles: true,
  enabled_chat_role_ids: ['legal', 'security_compliance'],
  log_level: 'info',
  diagnostics_profile: 'standard',
  chat_trace_logging: false,
  chat_trace_redaction_mode: 'minimal',
  chat_trace_user_retention_days: 30,
  chat_trace_evaluation_retention_days: 30,
  enable_raw_output_control: false,
  ui_theme: 'blue',
  translate_default_language: 'Spanish',
  translate_default_tone: 'natural',
  translate_pinned_languages: [],
  translate_pinned_languages_limit: 6,
  llm_model_filename: 'main.gguf',
  available_models: ['main.gguf', 'alt.gguf'],
  model_profile: {
    name: 'Qwen 14B',
  },
  embedding_model: 'embed.gguf',
  rag_reranker_model: 'reranker.gguf',
}

// Render SettingsView with an optional ?tab= URL param (replaces old tab-click navigation)
function renderSettingsView(options?: {
  tab?: string
  settings?: typeof baseSettings
  onRequestClearMcpTokenConfirm?: () => Promise<boolean>
  onRequestRemoveModelConfirm?: (modelName: string, modelSizeLabel?: string) => Promise<boolean>
}) {
  const onSave = vi.fn()
  const onDiscard = vi.fn()
  const onResetSettings = vi.fn()
  const onResetIndex = vi.fn()
  const tab = options?.tab
  const initialEntry = tab ? `/settings?tab=${tab}` : '/settings'

  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <SettingsView
        settings={options?.settings ?? baseSettings}
        fileTypeOptions={[{ id: 'docs', label: 'Docs', extensions: ['.md', '.txt'] }]}
        onSave={onSave}
        onRequestClearMcpTokenConfirm={options?.onRequestClearMcpTokenConfirm}
        onRequestRemoveModelConfirm={options?.onRequestRemoveModelConfirm}
        onDiscard={onDiscard}
        onResetSettings={onResetSettings}
        onResetIndex={onResetIndex}
        saving={false}
      />
    </MemoryRouter>,
  )

  return { onSave, onDiscard, onResetSettings, onResetIndex }
}

describe('SettingsView tabs and action bar behavior', () => {
  it('renders General tab content by default', () => {
    renderSettingsView()
    // General section has Full Privacy Mode and Theme
    expect(screen.getByText('Full Privacy Mode')).toBeInTheDocument()
    expect(screen.getByText('Theme')).toBeInTheDocument()
  })

  it('shows the correct section when tab param is set', () => {
    renderSettingsView({ tab: 'diagnostics' })
    expect(screen.getByText('Diagnostics Profile')).toBeInTheDocument()
    // General content is in a hidden section (settings-section--hidden class applied via CSS)
    const privacyLabel = screen.queryByText('Full Privacy Mode')
    expect(privacyLabel?.closest('.settings-section--hidden')).toBeTruthy()
  })

  it('restores active tab from localStorage when no URL param is present', () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'diagnostics')
    renderSettingsView()
    expect(screen.getByText('Diagnostics Profile')).toBeInTheDocument()
    const privacyLabel = screen.queryByText('Full Privacy Mode')
    expect(privacyLabel?.closest('.settings-section--hidden')).toBeTruthy()
  })

  it('shows Save/Discard on the System tab', () => {
    renderSettingsView({ tab: 'system' })

    expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Discard Changes' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset Settings/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset All/i })).toBeInTheDocument()
  })

  it('shows Save/Discard on the General tab', () => {
    renderSettingsView({ tab: 'general' })
    expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Discard Changes' })).toBeInTheDocument()
  })

  it('keeps model selection editable and saves selected model when tab state is restored', async () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'models')
    const { onSave } = renderSettingsView()

    const modelSelect = screen.getByLabelText('Main model') as HTMLSelectElement
    expect(modelSelect.value).toBe('main.gguf')

    fireEvent.change(modelSelect, { target: { value: 'alt.gguf' } })
    expect((screen.getByLabelText('Main model') as HTMLSelectElement).value).toBe('alt.gguf')

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ llm_model_filename: 'alt.gguf' }),
    )
  })

  it('includes installed models not present in catalog entries', async () => {
    const settingsWithExtraModel = {
      ...baseSettings,
      available_models: ['main.gguf', 'alt.gguf', 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'],
    }

    renderSettingsView({ tab: 'models', settings: settingsWithExtraModel })

    await waitFor(() => {
      const modelSelect = screen.getByLabelText('Main model') as HTMLSelectElement
      const optionValues = Array.from(modelSelect.options).map((option) => option.value)
      expect(optionValues).toContain('Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
    })
  })

  it('lets users pin translate languages and persists the selection on save', async () => {
    const { onSave } = renderSettingsView({
      tab: 'translate',
      settings: {
        ...baseSettings,
        translate_default_language: 'German',
      },
    })

    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'ger' } })
    fireEvent.click(screen.getByRole('button', { name: 'German' }))
    expect(screen.queryByRole('button', { name: 'Remove German' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'spa' } })
    fireEvent.click(screen.getByRole('button', { name: 'Spanish' }))
    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'fre' } })
    fireEvent.click(screen.getByRole('button', { name: 'French' }))

    expect(screen.getByRole('button', { name: 'Remove French' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove Spanish' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        translate_pinned_languages: ['French', 'Spanish'],
      }),
    )
  })

  it('caps pinned translate languages at the configured limit', async () => {
    const { onSave } = renderSettingsView({
      tab: 'translate',
      settings: {
        ...baseSettings,
        translate_default_language: 'German',
        translate_pinned_languages_limit: 2,
      },
    })

    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'fre' } })
    fireEvent.click(screen.getByRole('button', { name: 'French' }))
    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'spa' } })
    fireEvent.click(screen.getByRole('button', { name: 'Spanish' }))
    fireEvent.change(screen.getByLabelText('Additional Languages'), { target: { value: 'ita' } })
    fireEvent.click(screen.getByRole('button', { name: 'Italian' }))

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        translate_pinned_languages: ['French', 'Spanish'],
      }),
    )
  })

  it('hides advanced diagnostics controls when profile is not custom', () => {
    renderSettingsView({ tab: 'diagnostics' })

    expect(screen.queryByText('Advanced Diagnostics')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Log level')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Trace redaction')).not.toBeInTheDocument()
  })

  it('shows advanced diagnostics controls when profile is custom', () => {
    const settingsCustom = { ...baseSettings, diagnostics_profile: 'custom' as const }

    renderSettingsView({ tab: 'diagnostics', settings: settingsCustom })

    expect(screen.getByText('Advanced Diagnostics')).toBeInTheDocument()
    expect(screen.getByLabelText('Log level')).toBeInTheDocument()
    expect(screen.getByLabelText('Trace redaction')).toBeInTheDocument()
    expect(screen.getByLabelText('User trace retention (days)')).toBeInTheDocument()
    expect(screen.getByLabelText('Evaluation trace retention (days)')).toBeInTheDocument()
  })

  it('does not render hidden advanced tuning controls on the Chat tab', () => {
    renderSettingsView({ tab: 'chat' })
    expect(screen.queryByLabelText('Enable adaptive passage retrieval')).not.toBeInTheDocument()
  })

  it('does not render chunk size or embedding controls on the Indexing tab', () => {
    renderSettingsView({ tab: 'indexing' })
    expect(screen.queryByText(/Chunk size:/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Overlap:/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('embedding-batch-size')).not.toBeInTheDocument()
    expect(screen.queryByText('Embedding Batch Size')).not.toBeInTheDocument()
  })

  it('keeps hidden settings in save payload (no contract regression)', () => {
    const { onSave } = renderSettingsView()

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        chunk_size_tokens: baseSettings.chunk_size_tokens,
        chunk_overlap_tokens: baseSettings.chunk_overlap_tokens,
        embedding_batch_size: baseSettings.embedding_batch_size,
        adaptive_rag_tuning: baseSettings.adaptive_rag_tuning,
        enable_raw_output_control: baseSettings.enable_raw_output_control,
      }),
    )
  })

  it('shows CPU responsiveness on the System tab', () => {
    renderSettingsView({ tab: 'system' })
    expect(screen.getByText('CPU Performance')).toBeInTheDocument()
  })

  it('shows specialization plugins on the Chat tab', () => {
    renderSettingsView({ tab: 'chat' })
    expect(screen.getByText('Specialization Plugins')).toBeInTheDocument()
  })

  it('shows chat activity logs toggle on the Diagnostics tab (moved from Chat)', () => {
    renderSettingsView({ tab: 'diagnostics' })
    expect(screen.getByText('Save chat activity logs')).toBeInTheDocument()
  })

  it('chat activity logs toggle is in a hidden section on the Chat tab', () => {
    renderSettingsView({ tab: 'chat' })
    const logsLabel = screen.queryByText('Save chat activity logs')
    expect(logsLabel?.closest('.settings-section--hidden')).toBeTruthy()
  })

  it('renders updated plain-language settings labels on Chat and Indexing tabs', () => {
    renderSettingsView({ tab: 'chat' })
    expect(screen.getByText('Conversation Memory')).toBeInTheDocument()

    cleanup()

    renderSettingsView({ tab: 'indexing' })
    expect(screen.getByText('File Processing Timeout')).toBeInTheDocument()
  })

  it('clears MCP token when switching HTTP to STDIO after confirmation', async () => {
    const confirmClear = vi.fn(async () => true)
    const settings = {
      ...baseSettings,
      mcp_enabled: true,
      mcp_transport: 'http' as const,
      mcp_http_host: '127.0.0.1',
      mcp_http_port: 8431,
      mcp_scope_mode: 'metadata_only' as const,
      mcp_access_token: 'imcp_abcdefghijklmnopqrstuvwxyzABCDEF',
      mcp_token_configured: true,
    }

    const { onSave } = renderSettingsView({ tab: 'integrations', settings, onRequestClearMcpTokenConfirm: confirmClear })
    fireEvent.click(screen.getByRole('tab', { name: 'MCP Server' }))

    const transport = screen.getByLabelText('Transport') as HTMLSelectElement
    fireEvent.change(transport, { target: { value: 'stdio' } })
    await waitFor(() => expect(confirmClear).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({
      mcp_transport: 'stdio',
      mcp_access_token: '',
    }))
  })

  it('keeps MCP HTTP transport when token-clear confirmation is canceled', async () => {
    const confirmClear = vi.fn(async () => false)
    const settings = {
      ...baseSettings,
      mcp_enabled: true,
      mcp_transport: 'http' as const,
      mcp_http_host: '127.0.0.1',
      mcp_http_port: 8431,
      mcp_scope_mode: 'metadata_only' as const,
      mcp_access_token: 'imcp_abcdefghijklmnopqrstuvwxyzABCDEF',
      mcp_token_configured: true,
    }

    renderSettingsView({ tab: 'integrations', settings, onRequestClearMcpTokenConfirm: confirmClear })
    fireEvent.click(screen.getByRole('tab', { name: 'MCP Server' }))

    const transport = screen.getByLabelText('Transport') as HTMLSelectElement
    fireEvent.change(transport, { target: { value: 'stdio' } })
    await waitFor(() => expect(confirmClear).toHaveBeenCalledTimes(1))
    expect((screen.getByLabelText('Transport') as HTMLSelectElement).value).toBe('http')
  })

  it('shows remove button disabled for the currently active installed model', async () => {
    renderSettingsView({ tab: 'models' })
    const removeButton = await screen.findByRole('button', { name: 'Remove' })
    expect(removeButton).toBeDisabled()
  })

  it('removes a non-active installed model after confirmation', async () => {
    const confirmRemove = vi.fn(async () => true)
    renderSettingsView({ tab: 'models', onRequestRemoveModelConfirm: confirmRemove })
    fireEvent.change(screen.getByLabelText('Main model'), { target: { value: 'alt.gguf' } })

    const removeButton = await screen.findByRole('button', { name: 'Remove' })
    expect(removeButton).toBeEnabled()
    fireEvent.click(removeButton)

    const api = await import('../../api')
    await waitFor(() => expect(confirmRemove).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(api.removeModel).toHaveBeenCalledWith('alt.gguf'))
  })
})
