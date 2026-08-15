/**
 * Informity AI — Top banner
 * Shared top-of-page banner for alerts and status messages.
 */
import './TopBanner.css'

type TopBannerTone = 'danger' | 'info'

interface TopBannerProps {
  tone?: TopBannerTone
  iconClassName?: string
  children: string
}

export function TopBanner({
  tone = 'info',
  iconClassName = 'ri-information-line',
  children,
}: TopBannerProps) {
  return (
    <div className={`top-banner top-banner--${tone}`} role="alert">
      <i className={iconClassName} aria-hidden style={{ fontSize: '1.125rem' }} />
      <span>{children}</span>
    </div>
  )
}
