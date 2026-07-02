import { StateMessage } from './StateMessage'
import './BootOverlay.css'

interface BootOverlayProps {
  title: string
  elapsedSeconds: number | null
  description?: string | null
}

export function BootOverlay({ title, elapsedSeconds, description = null }: BootOverlayProps) {
  const resolvedTitle = elapsedSeconds === null || elapsedSeconds <= 0
    ? title
    : `${title} ${elapsedSeconds}s`

  return (
    <div className="boot-overlay" aria-live="polite" aria-label="Application startup">
      <StateMessage
        icon="ri-loader-4-line"
        title={resolvedTitle}
        description={description ?? ''}
        className="boot-overlay__message"
      />
    </div>
  )
}
