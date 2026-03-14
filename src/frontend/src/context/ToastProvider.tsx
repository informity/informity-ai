/**
 * Informity AI — Toast notification provider
 * Bottom-right toasts for success, error, info, warning.
 */
import { useState, useCallback, useEffect, useRef, type ReactNode } from 'react'
import { ToastContext } from './toastContext'
import './Toast.css'

interface Toast {
  id: string
  type: string
  message: string
}

const TOAST_ICONS: Record<string, string> = {
  success: 'ri-checkbox-circle-line',
  error:   'ri-close-circle-line',
  info:    'ri-information-line',
  warning: 'ri-error-warning-line',
}

const AUTO_DISMISS_MS = 5000

interface ToastProviderProps {
  children: ReactNode
}

export function ToastProvider({ children }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const dismissTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  const clearDismissTimer = useCallback((id: string) => {
    const timer = dismissTimersRef.current.get(id)
    if (!timer) return
    clearTimeout(timer)
    dismissTimersRef.current.delete(id)
  }, [])

  const showToast = useCallback((type: string, message: string) => {
    const id = crypto.randomUUID?.() || Date.now().toString()
    setToasts((prev) => [...prev, { id, type, message }])
    clearDismissTimer(id)
    const timer = setTimeout(() => {
      dismissTimersRef.current.delete(id)
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, AUTO_DISMISS_MS)
    dismissTimersRef.current.set(id, timer)
  }, [clearDismissTimer])

  const dismiss = useCallback((id: string) => {
    clearDismissTimer(id)
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [clearDismissTimer])

  useEffect(() => {
    const timerMap = dismissTimersRef.current
    const handler = (e: Event) => {
      const { type, message } = (e as CustomEvent).detail || {}
      if (type && message) {
        showToast(type, message)
      }
    }
    window.addEventListener('show-toast', handler)
    return () => {
      window.removeEventListener('show-toast', handler)
      for (const timer of timerMap.values()) {
        clearTimeout(timer)
      }
      timerMap.clear()
    }
  }, [showToast])

  return (
    <ToastContext.Provider value={{ showToast, dismiss }}>
      {children}
      <div className="toast-container" aria-live="polite">
        {toasts.map((t) => {
          const iconClass = TOAST_ICONS[t.type] || 'ri-information-line'
          return (
            <div
              key={t.id}
              className={`toast toast--${t.type}`}
              role="alert"
            >
              <i className={iconClass + ' toast__icon'} aria-hidden style={{ fontSize: '1.125rem' }} />
              <span className="toast__message">{t.message}</span>
              <button
                type="button"
                className="toast__close"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss"
              >
                <i className="ri-close-line" aria-hidden style={{ fontSize: '0.875rem' }} />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
