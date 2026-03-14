/**
 * Informity AI — Chat context hook
 * Access current chat ID and setter for persistence across navigation.
 */
import { useContext } from 'react'
import { ChatContext } from './chatContext'

export function useChatContext() {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChatContext must be used within ChatProvider')
  return ctx
}
