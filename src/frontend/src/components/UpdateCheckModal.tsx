import './ConfirmDialog.css'

type UpdateCheckState = 'checking' | 'up_to_date' | 'update_available' | 'error'

interface UpdateCheckModalProps {
  open: boolean
  state: UpdateCheckState
  currentVersion?: string | null
  latestVersion?: string | null
  releaseNotes?: string | null
  errorMessage?: string | null
  checking: boolean
  onClose: () => void
  onRetry: () => void
  onDownload: () => void
}

function getTitle(state: UpdateCheckState): string {
  if (state === 'checking') return 'Checking for updates...'
  if (state === 'up_to_date') return "You're up to date"
  if (state === 'update_available') return 'Update Available'
  return 'Unable to Check for Updates'
}

function getIconClass(state: UpdateCheckState, checking: boolean): string {
  if (state === 'checking' || checking) return 'ri-loader-4-line'
  if (state === 'up_to_date') return 'ri-checkbox-circle-line'
  if (state === 'update_available') return 'ri-download-2-line'
  return 'ri-error-warning-line'
}

export function UpdateCheckModal({
  open,
  state,
  currentVersion,
  latestVersion,
  releaseNotes,
  errorMessage,
  checking,
  onClose,
  onRetry,
  onDownload,
}: UpdateCheckModalProps) {
  if (!open) return null

  const title = getTitle(state)
  const iconClass = getIconClass(state, checking)
  const disableButtons = checking

  return (
    <div
      className="confirm-dialog__backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="update-check-modal-title"
      aria-describedby="update-check-modal-message"
    >
      <div className="confirm-dialog confirm-dialog--centered" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-dialog__header confirm-dialog__header--centered">
          <div className="confirm-dialog__icon confirm-dialog__icon--default">
            <i className={iconClass} aria-hidden style={{ fontSize: '2.5rem' }} />
          </div>
          <h2 id="update-check-modal-title" className="confirm-dialog__title">{title}</h2>
        </div>
        <div className="confirm-dialog__body">
          {state === 'checking' && (
            <p id="update-check-modal-message" className="confirm-dialog__message">
              Looking up the latest release information...
            </p>
          )}
          {state === 'up_to_date' && (
            <p id="update-check-modal-message" className="confirm-dialog__message">
              You are running version {currentVersion || '--'}, which is the latest available.
            </p>
          )}
          {state === 'update_available' && (
            <>
              <p id="update-check-modal-message" className="confirm-dialog__message">
                Version {latestVersion || '--'} is available. You are currently on {currentVersion || '--'}.
              </p>
              {releaseNotes && (
                <p className="confirm-dialog__subtext">
                  {releaseNotes}
                </p>
              )}
            </>
          )}
          {state === 'error' && (
            <p id="update-check-modal-message" className="confirm-dialog__message">
              {errorMessage || "Couldn't check for updates right now. Please try again."}
            </p>
          )}
        </div>
        <div className="confirm-dialog__footer">
          {state === 'update_available' ? (
            <>
              <button
                type="button"
                className="confirm-dialog__btn confirm-dialog__btn--cancel"
                onClick={onClose}
                disabled={disableButtons}
              >
                Later
              </button>
              <button
                type="button"
                className="confirm-dialog__btn confirm-dialog__btn--primary"
                onClick={onDownload}
                disabled={disableButtons}
              >
                Download Update
              </button>
            </>
          ) : state === 'error' ? (
            <>
              <button
                type="button"
                className="confirm-dialog__btn confirm-dialog__btn--cancel"
                onClick={onClose}
                disabled={disableButtons}
              >
                Close
              </button>
              <button
                type="button"
                className="confirm-dialog__btn confirm-dialog__btn--primary"
                onClick={onRetry}
                disabled={disableButtons}
              >
                Try Again
              </button>
            </>
          ) : (
            <button
              type="button"
              className="confirm-dialog__btn confirm-dialog__btn--primary"
              onClick={onClose}
              disabled={disableButtons}
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
