import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SettingsView } from './SettingsView'
import { SETTINGS_ACTIVE_TAB_STORAGE_KEY } from '../../utils/storageKeys'

vi.mock('../../api', () => ({
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
  log_level: 'info',
  diagnostics_profile: 'standard',
  chat_trace_logging: false,
  chat_trace_redaction_mode: 'minimal',
  chat_trace_user_retention_days: 30,
  chat_trace_evaluation_retention_days: 30,
  enable_raw_output_control: false,
  ui_theme: 'blue',
  llm_model_filename: 'main.gguf',
  available_models: ['main.gguf', 'alt.gguf'],
  model_profile: {
    name: 'Qwen 14B',
  },
  embedding_model: 'embed.gguf',
  rag_reranker_model: 'reranker.gguf',
}

function renderSettingsView() {
  const onSave = vi.fn()
  const onDiscard = vi.fn()
  const onResetSettings = vi.fn()
  const onResetIndex = vi.fn()

  render(
    <MemoryRouter>
      <SettingsView
        settings={baseSettings}
        fileTypeOptions={[{ id: 'docs', label: 'Docs', extensions: ['.md', '.txt'] }]}
        onSave={onSave}
        onDiscard={onDiscard}
        onResetSettings={onResetSettings}
        onResetIndex={onResetIndex}
        saving={false}
      />
    </MemoryRouter>,
  )

  return { onSave, onDiscard, onResetSettings, onResetIndex }
}

function sectionHeaderFor(element: HTMLElement): string {
  const section = element.closest('section')
  const header = section?.querySelector('.settings-section-header')
  return (header?.textContent || '').trim()
}

describe('SettingsView tabs and action bar behavior', () => {
  it('renders expected tabs and defaults to General tab selected', () => {
    renderSettingsView()

    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Data Sources' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'System' })).toBeInTheDocument()
  })

  it('switches tabs when clicked', async () => {
    renderSettingsView()

    fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }))
    expect(screen.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'false')
  })

  it('restores active tab from localStorage and persists updates', async () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'diagnostics')
    renderSettingsView()

    expect(screen.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'false')

    fireEvent.click(screen.getByRole('tab', { name: 'System' }))
    expect(localStorage.getItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY)).toBe('system')
  })

  it('shows Save/Discard on non-System tabs and hides them on System', async () => {
    renderSettingsView()

    expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Discard Changes' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'System' }))

    expect(screen.queryByRole('button', { name: 'Save Settings' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Discard Changes' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset Settings/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset All/i })).toBeInTheDocument()
  })

  it('keeps model selection editable and saves selected model when tab state is restored', async () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'models')
    const { onSave } = renderSettingsView()

    const modelSelect = screen.getByLabelText('Model') as HTMLSelectElement
    expect(modelSelect.value).toBe('main.gguf')

    fireEvent.change(modelSelect, { target: { value: 'alt.gguf' } })
    expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('alt.gguf')

    fireEvent.click(screen.getByRole('button', { name: 'Save Settings' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ llm_model_filename: 'alt.gguf' }),
    )
  })

  it('includes installed models not present in catalog entries', async () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'models')
    const settingsWithExtraModel = {
      ...baseSettings,
      available_models: ['main.gguf', 'alt.gguf', 'Qwen3.5-35B-A3B-Q4_K_M.gguf'],
    }

    render(
      <MemoryRouter>
        <SettingsView
          settings={settingsWithExtraModel}
          fileTypeOptions={[{ id: 'docs', label: 'Docs', extensions: ['.md', '.txt'] }]}
          onSave={vi.fn()}
          onDiscard={vi.fn()}
          onResetSettings={vi.fn()}
          onResetIndex={vi.fn()}
          saving={false}
        />
      </MemoryRouter>,
    )

    await waitFor(() => {
      const modelSelect = screen.getByLabelText('Model') as HTMLSelectElement
      const optionValues = Array.from(modelSelect.options).map((option) => option.value)
      expect(optionValues).toContain('Qwen3.5-35B-A3B-Q4_K_M.gguf')
    })
  })

  it('hides advanced diagnostics controls when profile is not custom', () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'diagnostics')
    renderSettingsView()

    expect(screen.queryByText('Advanced Diagnostics')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Log Level')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Trace Redaction')).not.toBeInTheDocument()
  })

  it('shows advanced diagnostics controls when profile is custom', () => {
    localStorage.setItem(SETTINGS_ACTIVE_TAB_STORAGE_KEY, 'diagnostics')
    const settingsCustom = { ...baseSettings, diagnostics_profile: 'custom' as const }

    render(
      <MemoryRouter>
        <SettingsView
          settings={settingsCustom}
          fileTypeOptions={[{ id: 'docs', label: 'Docs', extensions: ['.md', '.txt'] }]}
          onSave={vi.fn()}
          onDiscard={vi.fn()}
          onResetSettings={vi.fn()}
          onResetIndex={vi.fn()}
          saving={false}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Advanced Diagnostics')).toBeInTheDocument()
    expect(screen.getByLabelText('Log Level')).toBeInTheDocument()
    expect(screen.getByLabelText('Trace Redaction')).toBeInTheDocument()
    expect(screen.getByLabelText('User Trace Retention (Days)')).toBeInTheDocument()
    expect(screen.getByLabelText('Evaluation Trace Retention (Days)')).toBeInTheDocument()
  })

  it('does not render hidden advanced tuning controls in chat/indexing/diagnostics', () => {
    renderSettingsView()

    fireEvent.click(screen.getByRole('tab', { name: 'Chat' }))
    expect(screen.queryByLabelText('Enable adaptive passage retrieval')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Indexing' }))
    expect(screen.queryByText(/Chunk size:/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Overlap:/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('embedding-batch-size')).not.toBeInTheDocument()
    expect(screen.queryByText('Embedding Batch Size')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }))
    expect(screen.queryByLabelText('Enable raw output view')).not.toBeInTheDocument()
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

  it('shows CPU responsiveness in System and not in Chat', () => {
    renderSettingsView()
    const cpuLabel = screen.getByText('CPU Responsiveness')
    expect(sectionHeaderFor(cpuLabel)).toContain('System')
  })

  it('shows chat activity logs toggle in Chat and not in Diagnostics', () => {
    renderSettingsView()
    const logsLabel = screen.getByText('Save chat activity logs')
    expect(sectionHeaderFor(logsLabel)).toContain('Chat')
  })

  it('renders updated plain-language settings labels', () => {
    renderSettingsView()

    fireEvent.click(screen.getByRole('tab', { name: 'Chat' }))
    expect(screen.getByText('Conversation Memory')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Indexing' }))
    expect(screen.getByText('File Processing Timeout')).toBeInTheDocument()
  })
})
