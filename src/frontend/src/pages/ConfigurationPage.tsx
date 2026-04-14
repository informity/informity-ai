/**
 * Informity AI — Environment Variables page
 * Reference for all INFORMITY_* environment variables.
 */
import { useState, useEffect } from 'react'
import { getEnvVars } from '../api'
import { PageHeader } from '../components/PageHeader'
import { ServiceUnavailableState } from '../components/ServiceUnavailableState'
import { showToast } from '../context/useToast'
import { useBackendStatus } from '../context/useBackendStatus'
import { isBackendConnectionError } from '../utils/networkErrors'
import { extractErrorMessage } from '../utils/errorMessages'
import '../pages/PlaceholderPage.css'
import './ConfigurationPage.css'

const CONFIG_SECTION_ICONS: Record<string, string> = {
  'Server':                    'ri-server-line',
  'Paths and Storage':         'ri-folder-line',
  'Privacy':                   'ri-shield-check-line',
  'Appearance':                'ri-palette-line',
  'Data Sources':              'ri-folder-open-line',
  'Indexing':                  'ri-stack-line',
  'Web Search':                'ri-global-line',
  'Embeddings':                'ri-ai-generate-3d-line',
  'LLM and RAG':               'ri-robot-2-line',
  'Logging':                   'ri-file-list-3-line',
  'Diagnostics':               'ri-pulse-line',
  'Retrieval Tuning':          'ri-equalizer-line',
  'Term Dictionary':           'ri-book-2-line',
  'Internal Constants':        'ri-settings-5-line',
  'Runtime Environment':       'ri-terminal-box-line',
}

const GROUP_ORDER: string[] = [
  'Privacy',
  'LLM and RAG',
  'Data Sources',
  'Indexing',
  'Web Search',
  'Embeddings',
  'Diagnostics',
  'Logging',
  'Appearance',
  'Paths and Storage',
  'Server',
  'Retrieval Tuning',
  'Term Dictionary',
  'Internal Constants',
  'Runtime Environment',
]

const GROUP_ORDER_INDEX = new Map<string, number>(
  GROUP_ORDER.map((title, idx) => [title, idx]),
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

function sortGroups(groups: EnvVarGroup[]): EnvVarGroup[] {
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
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getEnvVars()
      .then((env) => {
        if (!cancelled) setEnvVars(env as EnvVarsResponse)
      })
      .catch((err) => {
        if (!cancelled) {
          const msg = extractErrorMessage(err, 'Failed to load')
          const disconnected = isBackendConnectionError(err)
          setError(msg)
          if (!disconnected) showToast('error', msg)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  const pageHeader = (
    <PageHeader
      title="Environment Variables"
      subtitle="Reference for all INFORMITY_* environment variables. Values shown reflect the current runtime configuration."
      icon="ri-code-s-line"
    />
  )

  if (loading) {
    return (
      <div className="page">
        {pageHeader}
        <div className="page__scroll"><p>Loading...</p></div>
      </div>
    )
  }

  if (offline || error) {
    return (
      <div className="page">
        {pageHeader}
        <div className="page__scroll">
          {offline ? <ServiceUnavailableState /> : <p className="page__error">{error}</p>}
        </div>
      </div>
    )
  }

  const groups = envVars?.groups ? sortGroups(envVars.groups) : []

  return (
    <div className="page">
      {pageHeader}
      <div className="page__scroll">
        <div className="config__content">
          {groups.map((group) => (
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
    </div>
  )
}
