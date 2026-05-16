/**
 * Informity AI — Keyboard shortcuts help modal
 * Shown on Cmd+/ (or Ctrl+/). Lists all shortcuts.
 */
import { useState, useEffect, useCallback } from 'react'
import './KeyboardShortcutsModal.css'

const MOD_KEY = typeof navigator !== 'undefined' && navigator.platform?.toLowerCase().includes('mac')
  ? '⌘'
  : 'Ctrl'

const SHORTCUTS = [
  { keys: 'Command+Alt+Ctrl+;', desc: 'Focus app window' },
  { keys: 'Ctrl+1', desc: 'Go to Chat' },
  { keys: 'Ctrl+2', desc: 'Go to History' },
  { keys: 'Ctrl+3', desc: 'Go to Files' },
  { keys: 'Ctrl+4', desc: 'Go to Dashboard' },
  { keys: 'Ctrl+5', desc: 'Go to Settings' },
  { keys: `${MOD_KEY}+N`, desc: 'New Chat' },
  { keys: 'Enter', desc: 'Send Chat Message' },
  { keys: 'Shift+Enter', desc: 'New line in chat input' },
  { keys: `${MOD_KEY}+B`, desc: 'Toggle Sidebar' },
  { keys: `${MOD_KEY}+,`, desc: 'Open Settings' },
  { keys: `${MOD_KEY}+/`, desc: 'Show Keyboard Shortcuts' },
  { keys: 'Esc', desc: 'Close panel or modal' },
  { keys: `${MOD_KEY}+Shift+H`, desc: 'Toggle Chat History Panel' },
]

export function KeyboardShortcutsModal() {
  const [open, setOpen] = useState(false)

  const close = useCallback(() => setOpen(false), [])

  useEffect(() => {
    const handleOpen = () => setOpen(true)
    window.addEventListener('open-keyboard-shortcuts', handleOpen)
    return () => window.removeEventListener('open-keyboard-shortcuts', handleOpen)
  }, [])

  useEffect(() => {
    if (!open) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopImmediatePropagation()
        close()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, close])

  if (!open) return null

  return (
    <div
      className="keyboard-shortcuts-modal__backdrop"
      onClick={close}
      role="dialog"
      aria-modal="true"
      aria-labelledby="keyboard-shortcuts-title"
      aria-describedby="keyboard-shortcuts-body"
    >
      <div
        className="keyboard-shortcuts-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="keyboard-shortcuts-modal__header">
          <h2 id="keyboard-shortcuts-title">Keyboard Shortcuts</h2>
          <button
            type="button"
            className="keyboard-shortcuts-modal__close"
            onClick={close}
            aria-label="Close"
          >
            <i className="ri-close-line" aria-hidden style={{ fontSize: '1.25rem' }} />
          </button>
        </div>
        <div id="keyboard-shortcuts-body" className="keyboard-shortcuts-modal__body">
          <dl className="keyboard-shortcuts-list">
            {SHORTCUTS.map(({ keys, desc }) => (
              <div key={keys} className="keyboard-shortcuts-list__row">
                <dt className="keyboard-shortcuts-list__keys">
                  <kbd>{keys}</kbd>
                </dt>
                <dd className="keyboard-shortcuts-list__desc">{desc}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  )
}
