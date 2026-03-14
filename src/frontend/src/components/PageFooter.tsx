/**
 * Informity AI — Page footer
 * Unified footer for all pages (except chat). Matches Settings footer.
 */
import { useState, useEffect } from 'react'
import { getHealth } from '../api'
import { logApiError } from '../utils/logApiError'
import './PageFooter.css'

interface HealthResponse {
  version?: string
}

export function PageFooter() {
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    getHealth()
      .then((res) => {
        const r = res as HealthResponse
        if (mounted && r?.version) setVersion(r.version)
      })
      .catch((err) => logApiError(err, 'PageFooter.getHealth'))
    return () => { mounted = false }
  }, [])

  return (
    <p className="page-footer">
      © {new Date().getFullYear()}{' '}
      <a href="https://www.informity.com/" target="_blank" rel="noopener noreferrer" className="page-footer__link">
        Informity
      </a>
      {' · All Rights Reserved'}
      {version != null && ` · v${version}`}
    </p>
  )
}
