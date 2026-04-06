import type { ChatMode } from '../types/api'

export const CHAT_MODE_LABELS: Record<ChatMode, string> = {
  assistant: 'Assistant',
  researcher: 'Researcher',
}

export const CHAT_MODE_ICONS: Record<ChatMode, string> = {
  assistant: 'ri-robot-2-line',
  researcher: 'ri-search-ai-3-line',
}
