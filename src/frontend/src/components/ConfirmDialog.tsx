/**
 * Informity AI — Confirm dialog
 * Branded confirmation modal. Use via useConfirm() from context/useConfirm.
 */
import { useEffect, useRef } from 'react'
import './ConfirmDialog.css'

interface ConfirmDialogProps {
  open: boolean
  title?: string
  message: string
  subtext?: string
  confirmLabel?: string
  cancelLabel?: string
  variant?: 'default' | 'danger'
  icon?: string
  onConfirm?: () => void
  onCancel?: () => void
}

export function ConfirmDialog({
  open,
  title,
  message,
  subtext,
  confirmLabel = 'OK',
  cancelLabel = 'Cancel',
  variant = 'default',
  icon,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const confirmRef = useRef<HTMLButtonElement>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopImmediatePropagation()
        onCancel?.()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, onCancel])

  useEffect(() => {
    if (!open) return
    const el = variant === 'danger' ? cancelRef.current : confirmRef.current
    if (el) el.focus()
  }, [open, variant])

  if (!open) return null

  return (
    <div
      className="confirm-dialog__backdrop"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby="confirm-dialog-message"
    >
      <div
        className={`confirm-dialog ${icon ? 'confirm-dialog--centered' : ''}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className={`confirm-dialog__header ${icon ? 'confirm-dialog__header--centered' : ''}`}
        >
          {icon && (
            <div className={`confirm-dialog__icon confirm-dialog__icon--${variant}`}>
              <i className={icon} aria-hidden style={{ fontSize: '2.5rem' }} />
            </div>
          )}
          <h2 id="confirm-dialog-title" className="confirm-dialog__title">
            {title}
          </h2>
        </div>
        <div className="confirm-dialog__body">
          <p id="confirm-dialog-message" className="confirm-dialog__message">
            {message}
          </p>
          {subtext && (
            <p className="confirm-dialog__subtext">
              {subtext}
            </p>
          )}
        </div>
        <div className="confirm-dialog__footer">
          <button
            ref={cancelRef}
            type="button"
            className="confirm-dialog__btn confirm-dialog__btn--cancel"
            onClick={onCancel}
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={`confirm-dialog__btn confirm-dialog__btn--primary confirm-dialog__btn--${variant}`}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
