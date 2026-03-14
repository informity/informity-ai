/**
 * Informity AI — Toast context (internal)
 * Shared context instance for ToastProvider and useToast.
 */
import { createContext } from 'react'

export interface ToastContextValue {
  showToast: (type: string, message: string) => void
  dismiss: (id: string) => void
}

export const ToastContext = createContext<ToastContextValue | null>(null)
