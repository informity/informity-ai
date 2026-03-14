import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { HistoryTable } from './HistoryTable'

const navigateMock = vi.fn()
const selectChatMock = vi.fn(async () => {})

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => navigateMock,
  }
})

vi.mock('../../context/useChatContext', () => ({
  useChatContext: () => ({
    currentChatId: null,
    activeGenerationChatId: null,
    isStreaming: false,
    selectChat: selectChatMock,
    newChat: vi.fn(async () => {}),
    stopStreaming: vi.fn(async () => true),
  }),
}))

vi.mock('../../context/useConfirm', () => ({
  useConfirm: () => vi.fn(async () => true),
}))

vi.mock('../../context/useToast', () => ({
  showToast: vi.fn(),
}))

describe('HistoryTable', () => {
  it('opens selected history chat and primes chat context', () => {
    render(
      <HistoryTable
        chats={[
          {
            chat_id: 'chat-b',
            title: 'Chat B',
            last_message_preview: 'latest answer',
            first_user_message: 'question',
            message_count: 2,
            last_message_at: '2026-02-20T10:00:00.000Z',
            updated_at: '2026-02-20T10:00:00.000Z',
            last_generation_seconds: 1.23,
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByLabelText('Open chat Chat B'))

    expect(selectChatMock).toHaveBeenCalledWith('chat-b')
    expect(navigateMock).toHaveBeenCalledWith('/chat', { state: { chatId: 'chat-b' } })
  })
})
