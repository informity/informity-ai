import { StateMessage } from './StateMessage'
import './CenteredState.css'

interface CenteredStateProps {
  icon?: string
  title: string
  description: string
  className?: string
}

export function CenteredState({ icon, title, description, className = '' }: CenteredStateProps) {
  const rootClassName = ['centered-state', className].filter(Boolean).join(' ')

  return (
    <div className={rootClassName}>
      <StateMessage
        icon={icon}
        title={title}
        description={description}
      />
    </div>
  )
}
