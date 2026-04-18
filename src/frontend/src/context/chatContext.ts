/**
 * Informity AI — Chat context (internal)
 * Shared context instance for ChatProvider and useChatContext.
 */
import { createContext } from 'react'
import type { ChatFileScope, ChatMessageDisplay, ChatMode } from '../types/api'

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
  chatFileScope: ChatFileScope | null
  setChatWebSearchPreferences: (prefs: { enabled: boolean; privacyOverride: boolean; persist?: boolean }) => Promise<void>
  startScopedChat: (scope: ChatFileScope) => Promise<void>
  clearChatFileScope: () => void
  selectChat: (chatId: string) => Promise<void>
  goToGeneratingChat: () => Promise<void>
  sendMessage: (
    text: string,
    options?: {
      isInternal?: boolean
      mode?: ChatMode
      fileScope?: ChatFileScope | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  continueLastScope: (
    anchorMessageId?: number,
    options?: {
      mode?: ChatMode
      fileScope?: ChatFileScope | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  stopStreaming: () => Promise<boolean>
  newChat: () => Promise<void>
  clearError: () => void
}

export const ChatContext = createContext<ChatContextValue | null>(null)
