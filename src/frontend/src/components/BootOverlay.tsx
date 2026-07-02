import { StateMessage } from './StateMessage'
import './BootOverlay.css'

interface BootOverlayProps {
  title: string
  elapsedSeconds: number | null
  description?: string | null
}

export function BootOverlay({ title, elapsedSeconds, description = null }: BootOverlayProps) {
  const resolvedDescription = description ?? (elapsedSeconds === null ? '' : `${elapsedSeconds}s elapsed`)

  return (
    <div className="boot-overlay" aria-live="polite" aria-label="Application startup">
      <StateMessage
        icon="ri-loader-4-line"
        title={title}
        description={resolvedDescription}
        className="boot-overlay__message"
      />
    </div>
  )
}
