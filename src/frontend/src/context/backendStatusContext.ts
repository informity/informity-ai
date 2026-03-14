import { createContext } from 'react'

interface BackendStatusContextValue {
  offline: boolean
}

export const BackendStatusContext = createContext<BackendStatusContextValue>({ offline: false })

