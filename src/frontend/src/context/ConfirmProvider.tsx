/**
 * Informity AI — Confirm dialog provider
 * Provides consistent branded confirmation modals.
 */
import { useState, useCallback, type ReactNode } from 'react'
import { ConfirmContext } from './confirmContext'
import { ConfirmDialog } from '../components/ConfirmDialog'
import type { ConfirmOptions } from './confirmContext'

interface ConfirmState extends ConfirmOptions {
  resolve: (value: boolean) => void
}

interface ConfirmProviderProps {
  children: ReactNode
}

export function ConfirmProvider({ children }: ConfirmProviderProps) {
  const [state, setState] = useState<ConfirmState | null>(null)

  const confirm = useCallback((options: ConfirmOptions) => {
    return new Promise<boolean>((resolve) => {
      setState({
        title:       options.title ?? 'Confirm',
        message:     options.message,
        subtext:     options.subtext,
        confirmLabel: options.confirmLabel ?? 'OK',
        cancelLabel:  options.cancelLabel ?? 'Cancel',
        variant:      options.variant ?? 'default',
        icon:         options.icon,
        resolve,
      })
    })
  }, [])

  const handleConfirm = useCallback(() => {
    if (state) {
      state.resolve(true)
      setState(null)
    }
  }, [state])

  const handleCancel = useCallback(() => {
    if (state) {
      state.resolve(false)
      setState(null)
    }
  }, [state])

  return (
    <ConfirmContext.Provider value={{ confirm }}>
      {children}
      {state && (
        <ConfirmDialog
          open
          title={state.title}
          message={state.message}
          subtext={state.subtext}
          confirmLabel={state.confirmLabel}
          cancelLabel={state.cancelLabel}
          variant={state.variant}
          icon={state.icon}
          onConfirm={handleConfirm}
          onCancel={handleCancel}
        />
      )}
    </ConfirmContext.Provider>
  )
}
