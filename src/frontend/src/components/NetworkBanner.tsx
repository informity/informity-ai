/**
 * Informity AI — Service unavailable banner
 * Uses shared backend reachability state and chat network errors.
 */
import { useChatContext } from '../context/useChatContext'
import { useBackendStatus } from '../context/useBackendStatus'
import { SERVICE_UNAVAILABLE_MESSAGE } from '../utils/networkErrors'
import { TopBanner } from './TopBanner'

export function NetworkBanner() {
  const { offline } = useBackendStatus()
  const { error } = useChatContext()

  const showServiceUnavailable = offline || error === SERVICE_UNAVAILABLE_MESSAGE
  if (!showServiceUnavailable) return null

  return (
    <TopBanner tone="danger" iconClassName="ri-server-line">
      {SERVICE_UNAVAILABLE_MESSAGE}
    </TopBanner>
  )
}
