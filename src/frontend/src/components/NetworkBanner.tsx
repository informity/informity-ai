/**
 * Informity AI — Service unavailable banner
 * Uses shared backend reachability state.
 */
import { useBackendStatus } from '../context/useBackendStatus'
import './NetworkBanner.css'

export function NetworkBanner() {
  const { offline } = useBackendStatus()

  if (!offline) return null

  return (
    <div className="network-banner" role="alert">
      <i className="ri-server-line" aria-hidden style={{ fontSize: '1.125rem' }} />
      <span>Service unavailable. Start or restart Informity AI, then try again.</span>
    </div>
  )
}
