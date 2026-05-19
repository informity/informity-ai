import { BrowserRouter, HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { ToastProvider } from './context/ToastProvider'
import { ConfirmProvider } from './context/ConfirmProvider'
import { ChatProvider } from './context/ChatProvider'
import { BackendStatusProvider } from './context/BackendStatusProvider'
import { ErrorBoundary } from './components/ErrorBoundary'
import { CenteredState } from './components/CenteredState'
import { Layout } from './components/Layout'
import {
  cancelSetup,
  getSetupEvents,
  getSetupStatus,
  retrySetup,
  startSetup,
  type SetupEventResponse,
  type SetupStatusResponse,
} from './api'
import { type SetupState, isSetupBlockingState } from './types/setupState'
import { extractErrorMessage } from './utils/errorMessages'
import './App.css'

const ChatPage = lazy(async () => ({ default: (await import('./pages/ChatPage')).ChatPage }))
const HistoryPage = lazy(async () => ({ default: (await import('./pages/HistoryPage')).HistoryPage }))
const FilesPage = lazy(async () => ({ default: (await import('./pages/FilesPage')).FilesPage }))
const DashboardPage = lazy(async () => ({ default: (await import('./pages/DashboardPage')).DashboardPage }))
const LogsPage = lazy(async () => ({ default: (await import('./pages/LogsPage')).LogsPage }))
const SettingsPage = lazy(async () => ({ default: (await import('./pages/SettingsPage')).SettingsPage }))
const ConfigurationPage = lazy(async () => ({ default: (await import('./pages/ConfigurationPage')).ConfigurationPage }))
const SetupRequiredPage = lazy(async () => ({ default: (await import('./pages/SetupRequiredPage')).SetupRequiredPage }))

interface AppProps {
  startupError?: string | null
}

function App({ startupError = null }: AppProps) {
  const [setupStatus, setSetupStatus] = useState<SetupStatusResponse | null>(null)
  const [setupEvent, setSetupEvent] = useState<SetupEventResponse | null>(null)
  const [setupError, setSetupError] = useState<string | null>(null)
  const [setupCancelled, setSetupCancelled] = useState(false)
  const [setupStartPending, setSetupStartPending] = useState(false)
  const [setupActionPending, setSetupActionPending] = useState(false)

  const refreshSetupStatus = useCallback(async () => {
    try {
      const status = await getSetupStatus()
      setSetupStatus(status)
      setSetupError(null)
    } catch (error) {
      const message = extractErrorMessage(error, 'Unable to determine setup status.')
      setSetupError(message)
    }
  }, [])

  useEffect(() => {
    void refreshSetupStatus()
  }, [refreshSetupStatus])

  const isSetupBlocking = Boolean(setupStatus && isSetupBlockingState(setupStatus.state))
  const setupBlockingState: Exclude<SetupState, 'ready'> | null = (
    setupStatus && isSetupBlockingState(setupStatus.state)
      ? setupStatus.state
      : null
  )

  useEffect(() => {
    if (!isSetupBlocking) return
    let mounted = true
    const poll = async () => {
      try {
        const [status, event] = await Promise.all([getSetupStatus(), getSetupEvents()])
        if (!mounted) return
        setSetupStatus(status)
        setSetupEvent(event)
      } catch {
        // keep prior state; explicit errors are handled by main status call
      }
    }
    void poll()
    const id = window.setInterval(() => { void poll() }, 2000)
    return () => {
      mounted = false
      window.clearInterval(id)
    }
  }, [isSetupBlocking])

  if (startupError) {
    return (
      <CenteredState
        icon="ri-alert-line"
        title="Backend startup failed."
        description={startupError}
      />
    )
  }

  if (setupError && !setupStatus) {
    return (
      <CenteredState
        icon="ri-error-warning-line"
        title="Unable to determine setup status."
        description={setupError}
      />
    )
  }

  if (setupCancelled) {
    return (
      <CenteredState
        icon="ri-close-circle-line"
        title="Setup canceled."
        description="You can close this window."
      />
    )
  }

  if (!setupStatus) {
    return (
      <CenteredState
        icon="ri-loader-4-line"
        title="Checking setup status..."
        description="Verifying required local models before loading the app."
      />
    )
  }

  const Router = window.__INFORMITY_DESKTOP__ ? HashRouter : BrowserRouter

  return (
    <ErrorBoundary>
      <ToastProvider>
        <ConfirmProvider>
          <ChatProvider>
            <BackendStatusProvider>
              <Router>
                <Suspense
                  fallback={(
                    <CenteredState
                      icon="ri-loader-4-line"
                      title="Loading page..."
                      description="Preparing application view."
                    />
                  )}
                >
                  <Routes>
                    {isSetupBlocking ? (
                      <>
                        <Route
                          path="/setup"
                          element={(
                            <SetupRequiredPage
                              state={setupBlockingState ?? 'setup_required'}
                              tierOptions={setupStatus.tier_options}
                              machineRamGb={setupStatus.machine_ram_gb}
                              recommendedTier={setupStatus.recommended_tier}
                              recommendedReason={setupStatus.recommended_reason}
                              event={setupEvent}
                              isStarting={setupStartPending}
                              isActing={setupActionPending}
                              onStartSetup={(tier, modelFilename) => {
                                setSetupStartPending(true)
                                void startSetup(tier, modelFilename)
                                  .then(() => refreshSetupStatus())
                                  .catch((error) => {
                                    const message = extractErrorMessage(error, 'Setup start failed.')
                                    setSetupError(message)
                                  })
                                  .finally(() => setSetupStartPending(false))
                              }}
                              onRetrySetup={() => {
                                setSetupActionPending(true)
                                void retrySetup()
                                  .then(() => Promise.all([refreshSetupStatus(), getSetupEvents().then(setSetupEvent)]))
                                  .catch((error) => {
                                    const message = extractErrorMessage(error, 'Setup retry failed.')
                                    setSetupError(message)
                                  })
                                  .finally(() => setSetupActionPending(false))
                              }}
                              onCancelDownload={() => {
                                setSetupActionPending(true)
                                void cancelSetup()
                                  .then(() => Promise.all([refreshSetupStatus(), getSetupEvents().then(setSetupEvent)]))
                                  .catch((error) => {
                                    const message = extractErrorMessage(error, 'Cancel download failed.')
                                    setSetupError(message)
                                  })
                                  .finally(() => setSetupActionPending(false))
                              }}
                              onCancel={() => {
                                setSetupActionPending(true)
                                void cancelSetup()
                                  .then(() => {
                                    if (window.__INFORMITY_DESKTOP__) {
                                      setSetupCancelled(true)
                                      window.close()
                                      return
                                    }
                                    setSetupCancelled(true)
                                  })
                                  .catch((error) => {
                                    const message = extractErrorMessage(error, 'Setup cancel failed.')
                                    setSetupError(message)
                                  })
                                  .finally(() => setSetupActionPending(false))
                              }}
                            />
                          )}
                        />
                        <Route path="*" element={<Navigate to="/setup" replace />} />
                      </>
                    ) : (
                      <>
                        <Route path="/setup" element={<Navigate to="/chat" replace />} />
                        <Route path="/" element={<Layout />}>
                          <Route index element={<Navigate to="/chat" replace />} />
                          <Route path="chat" element={<ChatPage />} />
                          <Route path="history" element={<HistoryPage />} />
                          <Route path="files" element={<FilesPage />} />
                          <Route path="dashboard" element={<DashboardPage />} />
                          <Route path="logs" element={<LogsPage />} />
                          <Route path="settings" element={<SettingsPage />} />
                          <Route path="settings/configuration" element={<ConfigurationPage />} />
                        </Route>
                        <Route path="*" element={<Navigate to="/chat" replace />} />
                      </>
                    )}
                  </Routes>
                </Suspense>
              </Router>
            </BackendStatusProvider>
          </ChatProvider>
        </ConfirmProvider>
      </ToastProvider>
    </ErrorBoundary>
  )
}

export default App
