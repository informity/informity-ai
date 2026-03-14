import { useContext } from 'react'
import { BackendStatusContext } from './backendStatusContext'

export function useBackendStatus() {
  return useContext(BackendStatusContext)
}

