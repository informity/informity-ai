/**
 * Informity AI — Configuration page
 * Environment variables and application defaults reference.
 */
import { useState, useEffect } from 'react'
import { getEnvVars, getConfigReference, ApiError } from '../api'
import { PageHeader } from '../components/PageHeader'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { showToast } from '../context/useToast'
import { useBackendStatus } from '../context/useBackendStatus'
import { isBackendConnectionError } from '../utils/networkErrors'
import '../pages/PlaceholderPage.css'
import './ConfigurationPage.css'

const CONFIG_SECTION_ICONS: Record<string, string> = {
  'Server': 'ri-server-line',
  'Paths and Storage': 'ri-folder-line',
  'Privacy': 'ri-shield-check-line',
  'Appearance': 'ri-palette-line',
  'Data Sources': 'ri-folder-line',
  'Indexing': 'ri-stack-line',
  'Embeddings': 'ri-ai-generate-3d-line',
  'LLM and RAG': 'ri-robot-2-line',
  'Logging': 'ri-file-list-3-line',
  'Diagnostics': 'ri-pulse-line',
  'Advanced and Internal': 'ri-tools-line',
  'Runtime Environment': 'ri-terminal-box-line',
  'Preset Exclusion Patterns': 'ri-filter-off-line',
  'File Processing Limits': 'ri-file-line',
  'File Watcher': 'ri-eye-line',
  'Operation State': 'ri-time-line',
  'RAG Coverage Retrieval': 'ri-mind-map',
}

const SETTINGS_ALIGNED_GROUP_ORDER = [
  'Privacy',
  'LLM and RAG',
  'Data Sources',
  'Indexing',
  'Embeddings',
  'Diagnostics',
  'Advanced and Internal',
  'Runtime Environment',
  'Appearance',
  'Logging',
  'Paths and Storage',
  'Server',
  'File Processing Limits',
  'Preset Exclusion Patterns',
  'File Watcher',
  'Operation State',
  'RAG Coverage Retrieval',
] as const

const GROUP_ORDER_INDEX = new Map<string, number>(
  SETTINGS_ALIGNED_GROUP_ORDER.map((title, idx) => [title, idx]),
)

interface EnvVarItem {
  name: string
  default: string
  description: string
}

interface EnvVarGroup {
  title: string
  description: string
  variables: EnvVarItem[]
}

interface EnvVarsResponse {
  groups?: EnvVarGroup[]
}

interface ConfigConstant {
  name: string
  description: string
  default: string
}

interface ConfigRefGroup {
  title: string
  description: string
  constants: ConfigConstant[]
}

interface ConfigRefResponse {
  groups?: ConfigRefGroup[]
}

function sortGroupsByPreferredOrder<T extends { title: string }>(groups: T[]): T[] {
  return [...groups].sort((a, b) => {
    const aOrder = GROUP_ORDER_INDEX.get(a.title) ?? Number.MAX_SAFE_INTEGER
    const bOrder = GROUP_ORDER_INDEX.get(b.title) ?? Number.MAX_SAFE_INTEGER
    if (aOrder !== bOrder) return aOrder - bOrder
    return a.title.localeCompare(b.title)
  })
}

export function ConfigurationPage() {
  const { offline } = useBackendStatus()
  const [envVars, setEnvVars] = useState<EnvVarsResponse | null>(null)
  const [configRef, setConfigRef] = useState<ConfigRefResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([getEnvVars(), getConfigReference()])
      .then(([env, ref]) => {
        if (!cancelled) {
          setEnvVars(env as EnvVarsResponse)
          setConfigRef(ref as ConfigRefResponse)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          const msg = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Failed to load'
          const disconnected = isBackendConnectionError(err)
          setError(msg)
          if (!disconnected) {
            showToast('error', msg)
          }
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (loading) {
    return (
      <div className="page">
        <PageHeader
          title="Application Configuration"
          subtitle="Environment variables and application defaults reference"
          icon="ri-code-s-line"
        />
        <div className="page__scroll">
          <p>Loading application configuration...</p>
        </div>
      </div>
    )
  }

  if (offline || error) {
    return (
      <div className="page">
        <PageHeader
          title="Application Configuration"
          subtitle="Environment variables and application defaults reference"
          icon="ri-code-s-line"
        />
        <div className="page__scroll">
          {offline ? <ServiceUnavailableState /> : <p className="page__error">{error}</p>}
        </div>
      </div>
    )
  }

  const orderedEnvGroups = envVars?.groups ? sortGroupsByPreferredOrder(envVars.groups) : []
  const orderedConfigGroups = configRef?.groups ? sortGroupsByPreferredOrder(configRef.groups) : []

  return (
    <div className="page">
      <PageHeader
        title="Application Configuration"
        subtitle="Environment variables and application defaults reference"
        icon="ri-code-s-line"
      />

      <div className="page__scroll">
        <div className="config__content">
          {orderedEnvGroups.length > 0 && (
            <div className="config__main-section ui-section-divider">
              <h3 className="config__main-section-title">Environment Variables</h3>
              <p className="config__main-section-description">
                Application settings can be configured via environment variables. Note: persisted <code>config.json</code> values may take precedence for keys saved by the Settings UI.
              </p>
              <div className="config__groups">
                {orderedEnvGroups.map((group) => (
                  <div key={group.title} className="config__section">
                    <div className="config__section-head ui-subsection-head">
                      <div className="config__section-title ui-subsection-title">
                        {CONFIG_SECTION_ICONS[group.title] && (
                          <i className={`${CONFIG_SECTION_ICONS[group.title]} config__section-icon ui-subsection-icon`} aria-hidden="true" />
                        )}
                        {group.title}
                      </div>
                      <p className="config__section-desc ui-subsection-description">{group.description}</p>
                    </div>
                    <div className="config__var-list">
                      {group.variables.map((v) => (
                        <div key={v.name} className="config__var ui-card">
                          <div className="config__var-name">{v.name}</div>
                          <div className="config__var-desc">{v.description}</div>
                          <div className="config__var-default">{`Current value: ${v.default || '(unset)'}`}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {orderedConfigGroups.length > 0 && (
            <div className="config__main-section ui-section-divider">
              <h3 className="config__main-section-title">Application Defaults and Constants</h3>
              <p className="config__main-section-description">
                Reference values for constants and defaults that are not configurable via environment variables. These are fixed application behaviors and limits.
              </p>
              <div className="config__groups">
                {orderedConfigGroups.map((group) => (
                  <div key={group.title} className="config__section">
                    <div className="config__section-head ui-subsection-head">
                      <div className="config__section-title ui-subsection-title">
                        {CONFIG_SECTION_ICONS[group.title] && (
                          <i className={`${CONFIG_SECTION_ICONS[group.title]} config__section-icon ui-subsection-icon`} aria-hidden="true" />
                        )}
                        {group.title}
                      </div>
                      <p className="config__section-desc ui-subsection-description">{group.description}</p>
                    </div>
                    <div className="config__var-list">
                      {group.constants.map((c) => (
                        <div key={c.name} className="config__var ui-card">
                          <div className="config__var-name">{c.name}</div>
                          <div className="config__var-desc">{c.description}</div>
                          <div className="config__var-default">{c.default}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
