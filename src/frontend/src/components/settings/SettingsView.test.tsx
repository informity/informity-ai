import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SettingsView } from './SettingsView'

vi.mock('../../api', () => ({
  getModelProfile: vi.fn(async () => ({})),
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
    localStorage.setItem('informity.settings.activeTab', 'diagnostics')
    renderSettingsView()

    expect(screen.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'false')

    fireEvent.click(screen.getByRole('tab', { name: 'System' }))
    expect(localStorage.getItem('informity.settings.activeTab')).toBe('system')
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
    localStorage.setItem('informity.settings.activeTab', 'models')
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
})
