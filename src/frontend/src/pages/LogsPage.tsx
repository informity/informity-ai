import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getLogEvents } from '../api'
import { PageHeader } from '../components/PageHeader'
import { CenteredState } from '../components/CenteredState'
import { extractErrorMessage } from '../utils/errorMessages'
import '../pages/PlaceholderPage.css'
import '../styles/shared/tables.css'
import './LogsPage.css'

type LogsTab = 'application' | 'errors' | 'integrations'

interface LogEntry {
  id: string
  timestamp: string
  level: 'info' | 'warning' | 'error'
  source: string
  message: string
}

const LOGS_TABS: Array<{ id: LogsTab; label: string; icon: string }> = [
  { id: 'application', label: 'Application', icon: 'ri-window-2-line' },
  { id: 'errors', label: 'Errors', icon: 'ri-error-warning-line' },
  { id: 'integrations', label: 'Integrations', icon: 'ri-plug-3-line' },
]

const TAB_HEADER: Record<LogsTab, { icon: string; title: string; description: string }> = {
  application: {
    icon: 'ri-window-2-line',
    title: 'Application',
    description: 'Monitor file scans, indexing jobs, and system startup history.',
  },
  errors: {
    icon: 'ri-error-warning-line',
    title: 'Errors',
    description: 'Errors and warnings flagged for review.',
  },
  integrations: {
    icon: 'ri-plug-3-line',
    title: 'Integrations',
    description: 'MCP Server, Ollama, and other integration activity, connection health, and access scope limits.',
  },
}

function formatTimestampIso(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value || '')
  const pad2 = (n: number) => String(n).padStart(2, '0')
  const y = date.getFullYear()
  const m = pad2(date.getMonth() + 1)
  const d = pad2(date.getDate())
  const hh = pad2(date.getHours())
  const mm = pad2(date.getMinutes())
  const ss = pad2(date.getSeconds())
  return `${y}-${m}-${d} ${hh}:${mm}:${ss}`
}

function eventTypeBadgeClass(level: LogEntry['level']): string {
  if (level === 'error') return 'data-table__badge logs-event-badge logs-event-badge--error'
  if (level === 'warning') return 'data-table__badge logs-event-badge logs-event-badge--warning'
  return 'data-table__badge logs-event-badge logs-event-badge--info'
}

function emptyState(activeTab: LogsTab): { icon: string; title: string; description: string } {
  if (activeTab === 'errors') {
    return {
      icon: 'ri-error-warning-line',
      title: 'No errors yet.',
      description: 'Any new issues will show up here.',
    }
  }
  if (activeTab === 'integrations') {
    return {
      icon: 'ri-plug-3-line',
      title: 'No integration events yet.',
      description: 'MCP server and Ollama activity will appear here.',
    }
  }
  return {
    icon: 'ri-window-2-line',
    title: 'No application events yet.',
    description: 'Application activity will appear here.',
  }
}

export function LogsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const tabParam = searchParams.get('tab')
  const initialTab: LogsTab = tabParam === 'errors' || tabParam === 'integrations' ? tabParam : 'application'
  const [activeTab, setActiveTab] = useState<LogsTab>(initialTab)
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const urlTab: LogsTab = tabParam === 'errors' || tabParam === 'integrations' ? tabParam : 'application'
    if (urlTab !== activeTab) {
      setActiveTab(urlTab)
    }
  }, [activeTab, tabParam])

  const onTabClick = (tab: LogsTab) => {
    setActiveTab(tab)
    const next = new URLSearchParams(searchParams)
    next.set('tab', tab)
    setSearchParams(next, { replace: true })
  }

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    ;(async () => {
      try {
        const response = await getLogEvents({ channel: activeTab, limit: 200 })
        if (cancelled) return
        const mapped = (response.items || []).map((item) => ({
          level: (item.event_type === 'warning' || item.event_type === 'error') ? item.event_type : 'info',
          id: String(item.id),
          timestamp: item.timestamp || item.created_at,
          source: item.source,
          message: item.message,
        })) as LogEntry[]
        setEntries(mapped)
      } catch (err) {
        if (cancelled) return
        setEntries([])
        setError(extractErrorMessage(err, 'Failed to load logs'))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [activeTab])

  const empty = emptyState(activeTab)
  const header = TAB_HEADER[activeTab]
  const emptyDescription = useMemo(() => {
    if (loading) return 'Loading events...'
    if (error) return error
    return empty.description
  }, [empty.description, error, loading])

  return (
    <div className="page page--logs">
      <PageHeader
        title="Activity Logs"
        subtitle="Review application activity, errors, and integration events."
        icon="ri-file-list-2-line"
      />
      <div className="page__scroll">
        <div className="logs-tabs" role="tablist" aria-label="Logs tabs">
          {LOGS_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`logs-tab ${activeTab === tab.id ? 'logs-tab--active' : ''}`}
              onClick={() => onTabClick(tab.id)}
            >
              <i className={tab.icon} aria-hidden />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
        {entries.length === 0 ? (
          <CenteredState icon={empty.icon} title={empty.title} description={emptyDescription} />
        ) : (
          <section className="logs-table-wrap">
            <div className="logs-section-header ui-title ui-title--section">
              <i className={`${header.icon} section-icon`} aria-hidden="true" />
              {header.title}
            </div>
            <p className="logs-section-description ui-description">{header.description}</p>
            <div className="data-table">
              <div className="data-table__scroll">
                <table className="data-table__table">
                  <thead>
                    <tr>
                      <th className="data-table__th logs-col--time">Timestamp</th>
                      <th className="data-table__th logs-col--level">Event Type</th>
                      <th className="data-table__th logs-col--source">Source</th>
                      <th className="data-table__th">Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((entry) => (
                      <tr key={entry.id} className="data-table__row">
                        <td className="data-table__td logs-col--time">{formatTimestampIso(entry.timestamp)}</td>
                        <td className="data-table__td logs-col--level">
                          <span className={eventTypeBadgeClass(entry.level)}>{entry.level}</span>
                        </td>
                        <td className="data-table__td logs-col--source">{entry.source}</td>
                        <td className="data-table__td">{entry.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}
