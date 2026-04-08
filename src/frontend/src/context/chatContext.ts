/**
 * Informity AI — Chat context (internal)
 * Shared context instance for ChatProvider and useChatContext.
 */
import { createContext } from 'react'
import type { ChatMessageDisplay, ChatMode } from '../types/api'

export interface ChatContextValue {
  currentChatId: string | null
  setCurrentChatId: (id: string | null) => void
  activeGenerationChatId: string | null
  activeGenerationRequestId: string | null
  hasActiveGenerationForCurrentChat: boolean
  messages: ChatMessageDisplay[]
  isStreaming: boolean
  loadingChat: boolean
  error: string | null
  enableRawOutputControl: boolean
  chatWebSearchEnabled: boolean
  chatWebSearchPrivacyOverride: boolean
  setChatWebSearchPreferences: (prefs: { enabled: boolean; privacyOverride: boolean; persist?: boolean }) => Promise<void>
  selectChat: (chatId: string) => Promise<void>
  goToGeneratingChat: () => Promise<void>
  sendMessage: (
    text: string,
    options?: {
      isInternal?: boolean
      mode?: ChatMode
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  continueLastScope: (
    anchorMessageId?: number,
    options?: {
      mode?: ChatMode
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  stopStreaming: () => Promise<boolean>
  newChat: () => Promise<void>
  clearError: () => void
}

export const ChatContext = createContext<ChatContextValue | null>(null)
