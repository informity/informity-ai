/**
 * Informity AI — Chat context (internal)
 * Shared context instance for ChatProvider and useChatContext.
 */
import { createContext } from 'react'
import type { ChatFileScope, ChatMessageDisplay, ChatMode, ChatUploadAttachment } from '../types/api'

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
  chatUploads: ChatUploadAttachment[]
  selectedUploadIds: string[]
  setChatWebSearchPreferences: (prefs: { enabled: boolean; privacyOverride: boolean; persist?: boolean }) => Promise<void>
  startScopedChat: (scope: ChatFileScope) => Promise<void>
  clearChatFileScope: () => void
  uploadFiles: (
    files: File[],
    options?: { onChatResolved?: (chatId: string) => void },
  ) => Promise<void>
  removeUploadedFile: (uploadId: string) => Promise<void>
  toggleUploadSelection: (uploadId: string) => void
  clearUploadSelection: () => void
  selectChat: (chatId: string) => Promise<void>
  goToGeneratingChat: () => Promise<void>
  sendMessage: (
    text: string,
    options?: {
      isInternal?: boolean
      mode?: ChatMode
      fileScope?: ChatFileScope | null
      scopedUploadIds?: string[] | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  continueLastScope: (
    anchorMessageId?: number,
    options?: {
      mode?: ChatMode
      fileScope?: ChatFileScope | null
      scopedUploadIds?: string[] | null
      chatWebSearchEnabled?: boolean
      chatWebSearchPrivacyOverride?: boolean
    },
  ) => Promise<void>
  stopStreaming: () => Promise<boolean>
  newChat: () => Promise<void>
  clearError: () => void
}

export const ChatContext = createContext<ChatContextValue | null>(null)
