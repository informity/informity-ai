/**
 * Informity AI — Chat context (internal)
 * Shared context instance for ChatProvider and useChatContext.
 */
import { createContext } from 'react'
import type { ChatMessageDisplay } from '../types/api'

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
  selectChat: (chatId: string) => Promise<void>
  goToGeneratingChat: () => Promise<void>
  sendMessage: (
    text: string,
    options?: { isInternal?: boolean },
  ) => Promise<void>
  continueLastScope: (
    anchorMessageId?: number,
  ) => Promise<void>
  stopStreaming: () => Promise<boolean>
  newChat: () => Promise<void>
  clearError: () => void
}

export const ChatContext = createContext<ChatContextValue | null>(null)
