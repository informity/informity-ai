import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ChatView } from '../components/chat/ChatView'

interface LocationState {
  prefillMessage?: string
  chatId?: string
  scopedFileId?: number
  scopedFileName?: string
}

export function ChatPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const routeState = location.state as LocationState | undefined
  const [initialRouteState] = useState<LocationState>(() => routeState ?? {})
  const prefillMessage = initialRouteState.prefillMessage ?? ''
  const initialChatId = initialRouteState.chatId ?? null
  const initialScopedFile = (
    Number.isFinite(initialRouteState.scopedFileId)
      ? {
          fileId: Number(initialRouteState.scopedFileId),
          filename: String(initialRouteState.scopedFileName || '').trim() || `File ${Number(initialRouteState.scopedFileId)}`,
        }
      : null
  )

  useEffect(() => {
    if (!routeState?.chatId && !routeState?.prefillMessage && !routeState?.scopedFileId) return
    // Clear transient route state so reload restores current chat pointer from settings
    // instead of re-opening stale history selection.
    navigate(location.pathname, { replace: true, state: null })
  }, [location.pathname, navigate, routeState?.chatId, routeState?.prefillMessage, routeState?.scopedFileId])

  return (
    <div className="chat-page">
      <ChatView prefillMessage={prefillMessage} initialChatId={initialChatId} initialScopedFile={initialScopedFile} />
    </div>
  )
}
