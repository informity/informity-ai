import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getModelOperationEvents, type ModelOperationEventResponse } from '../api'
import { formatFileSize } from '../utils/formatFileSize'
import './ModelOperationBanner.css'

const ACTIVE_STATES = new Set<ModelOperationEventResponse['state']>([
  'in_progress',
  'paused',
  'failed',
])

function formatSpeed(speedBps: number): string {
  if (!speedBps || speedBps <= 0) return '--'
  return `${formatFileSize(speedBps)}/s`
}

export function ModelOperationBanner() {
  const navigate = useNavigate()
  const [event, setEvent] = useState<ModelOperationEventResponse | null>(null)

  useEffect(() => {
    let mounted = true
    const poll = async () => {
      try {
        const next = await getModelOperationEvents()
        if (mounted) setEvent(next)
      } catch {
        // Ignore transient API errors; this banner is optional UI telemetry.
      }
    }
    void poll()
    const id = window.setInterval(() => { void poll() }, 2000)
    return () => {
      mounted = false
      window.clearInterval(id)
    }
  }, [])

  if (!event || !ACTIVE_STATES.has(event.state)) return null

  const progress = Math.max(0, Math.min(100, event.overall_pct || 0))
  const stateLabel = event.state === 'in_progress' ? 'Downloading model' : event.state === 'paused' ? 'Model download paused' : 'Model download failed'
  const modelLabel = event.model_filename ?? 'model'
  const detail = event.error || `${formatFileSize(event.bytes_done || 0)} of ${formatFileSize(event.bytes_total || 0)} at ${formatSpeed(event.speed_bps || 0)}`

  return (
    <button
      type="button"
      className={`model-operation-banner model-operation-banner--${event.state}`}
      onClick={() => {
        try {
          localStorage.setItem('informity.settings.activeTab', 'models')
        } catch {
          // Ignore localStorage access failures.
        }
        navigate('/settings')
      }}
      aria-label="Open model operations in settings"
    >
      <div className="model-operation-banner__row">
        <span className="model-operation-banner__title">{stateLabel}: {modelLabel}</span>
        <span className="model-operation-banner__pct">{progress}%</span>
      </div>
      <div className="model-operation-banner__track" aria-hidden>
        <div className="model-operation-banner__fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="model-operation-banner__detail">{detail}</div>
    </button>
  )
}
