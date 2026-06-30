import { StateMessage } from './StateMessage'
import './BootOverlay.css'

interface BootOverlayProps {
  title: string
  elapsedSeconds: number | null
}

export function BootOverlay({ title, elapsedSeconds }: BootOverlayProps) {
  const description = elapsedSeconds === null ? '' : `${elapsedSeconds}s elapsed`

  return (
    <div className="boot-overlay" aria-live="polite" aria-label="Application startup">
      <StateMessage
        icon="ri-loader-4-line"
        title={title}
        description={description}
        className="boot-overlay__message"
      />
    </div>
  )
}
