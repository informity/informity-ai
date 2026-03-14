/**
 * Informity AI — Confirm dialog context (internal)
 * Shared context instance for ConfirmProvider and useConfirm.
 */
import { createContext } from 'react'

export interface ConfirmOptions {
  title?: string
  message: string
  subtext?: string
  confirmLabel?: string
  cancelLabel?: string
  variant?: 'default' | 'danger'
  icon?: string
}

export interface ConfirmContextValue {
  confirm: (options: ConfirmOptions) => Promise<boolean>
}

export const ConfirmContext = createContext<ConfirmContextValue | null>(null)
