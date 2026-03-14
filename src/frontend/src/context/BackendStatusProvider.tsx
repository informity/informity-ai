import { useEffect, useState, type ReactNode } from 'react'
import { getHealth } from '../api'
import { BackendStatusContext } from './backendStatusContext'

const POLL_INTERVAL_MS = 5000

interface BackendStatusProviderProps {
  children: ReactNode
}

export function BackendStatusProvider({ children }: BackendStatusProviderProps) {
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    let mounted = true

    const check = async () => {
      try {
        await getHealth()
        if (mounted) setOffline(false)
      } catch {
        if (mounted) setOffline(true)
      }
    }

    check()
    const id = setInterval(check, POLL_INTERVAL_MS)
    return () => {
      mounted = false
      clearInterval(id)
    }
  }, [])

  return (
    <BackendStatusContext.Provider value={{ offline }}>
      {children}
    </BackendStatusContext.Provider>
  )
}

