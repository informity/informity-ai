import './StateMessage.css'

interface StateMessageProps {
  title?: string
  description?: string
  icon?: string
  className?: string
}

const DEFAULT_TITLE = 'Service unavailable.'
const DEFAULT_DESCRIPTION = 'Start or restart Informity AI, then try again.'
const DEFAULT_ICON = 'ri-server-line'

export function StateMessage({
  title = DEFAULT_TITLE,
  description = DEFAULT_DESCRIPTION,
  icon = DEFAULT_ICON,
  className = '',
}: StateMessageProps) {
  const rootClassName = ['state-message', className].filter(Boolean).join(' ')

  return (
    <div className={rootClassName} role="status" aria-live="polite">
      <i className={`state-message__icon ${icon}`} aria-hidden="true" />
      <p className="state-message__title">{title}</p>
      <p className="state-message__description">{description}</p>
    </div>
  )
}
