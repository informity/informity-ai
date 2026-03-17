import { BrowserRouter, HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ToastProvider } from './context/ToastProvider'
import { ConfirmProvider } from './context/ConfirmProvider'
import { ChatProvider } from './context/ChatProvider'
import { BackendStatusProvider } from './context/BackendStatusProvider'
import { ErrorBoundary } from './components/ErrorBoundary'
import { CenteredState } from './components/CenteredState'
import { Layout } from './components/Layout'
import { ChatPage } from './pages/ChatPage'
import { HistoryPage } from './pages/HistoryPage'
import { FilesPage } from './pages/FilesPage'
import { DashboardPage } from './pages/DashboardPage'
import { SettingsPage } from './pages/SettingsPage'
import { ConfigurationPage } from './pages/ConfigurationPage'
import './App.css'

interface AppProps {
  startupError?: string | null
}

function App({ startupError = null }: AppProps) {
  if (startupError) {
    return (
      <CenteredState
        icon="ri-alert-line"
        title="Backend startup failed."
        description={startupError}
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
                <Routes>
                  <Route path="/" element={<Layout />}>
                    <Route index element={<Navigate to="/chat" replace />} />
                    <Route path="chat" element={<ChatPage />} />
                    <Route path="history" element={<HistoryPage />} />
                    <Route path="files" element={<FilesPage />} />
                    <Route path="dashboard" element={<DashboardPage />} />
                    <Route path="settings" element={<SettingsPage />} />
                    <Route path="settings/configuration" element={<ConfigurationPage />} />
                  </Route>
                  <Route path="*" element={<Navigate to="/chat" replace />} />
                </Routes>
              </Router>
            </BackendStatusProvider>
          </ChatProvider>
        </ConfirmProvider>
      </ToastProvider>
    </ErrorBoundary>
  )
}

export default App
