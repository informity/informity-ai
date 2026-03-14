/**
 * Informity AI — Toast hook and standalone show
 * useToast: access toast API from within ToastProvider.
 * showToast: dispatch toast from anywhere (e.g. outside React, in catch blocks).
 */
import { useContext } from 'react'
import { ToastContext } from './toastContext'

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}

export function showToast(type: string, message: string): void {
  window.dispatchEvent(new CustomEvent('show-toast', { detail: { type, message } }))
}
