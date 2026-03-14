import { CenteredState } from './CenteredState'

interface ServiceUnavailableStateProps {
  className?: string
}

export function ServiceUnavailableState({ className = '' }: ServiceUnavailableStateProps) {
  return (
    <CenteredState
      className={className}
      icon="ri-server-line"
      title="Service unavailable."
      description="Start or restart Informity AI, then try again."
    />
  )
}
