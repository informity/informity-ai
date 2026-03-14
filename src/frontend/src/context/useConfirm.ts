/**
 * Informity AI — Confirm dialog hook
 * Access confirm() for consistent branded confirmation modals.
 */
import { useContext } from 'react'
import { ConfirmContext, type ConfirmContextValue } from './confirmContext'

export function useConfirm(): ConfirmContextValue['confirm'] {
  const ctx = useContext(ConfirmContext)
  if (!ctx) throw new Error('useConfirm must be used within ConfirmProvider')
  return ctx.confirm
}
