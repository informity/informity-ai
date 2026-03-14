import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ChatView } from '../components/chat/ChatView'

interface LocationState {
  prefillMessage?: string
  chatId?: string
}

export function ChatPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const routeState = location.state as LocationState | undefined
  const [initialRouteState] = useState<LocationState>(() => routeState ?? {})
  const prefillMessage = initialRouteState.prefillMessage ?? ''
  const initialChatId = initialRouteState.chatId ?? null

  useEffect(() => {
    if (!routeState?.chatId && !routeState?.prefillMessage) return
    // Clear transient route state so reload restores current chat pointer from settings
    // instead of re-opening stale history selection.
    navigate(location.pathname, { replace: true, state: null })
  }, [location.pathname, navigate, routeState?.chatId, routeState?.prefillMessage])

  return (
    <div className="chat-page">
      <ChatView prefillMessage={prefillMessage} initialChatId={initialChatId} />
    </div>
  )
}
