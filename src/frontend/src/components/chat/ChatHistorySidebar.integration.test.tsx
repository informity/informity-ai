import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatProvider } from '../../context/ChatProvider'
import { ConfirmProvider } from '../../context/ConfirmProvider'
import { Sidebar } from '../Sidebar'
import { ChatPage } from '../../pages/ChatPage'
import { HistoryTable } from '../history/HistoryTable'

const {
  deleteChatMock,
  getChatMock,
  getCurrentChatMock,
  getFilesMock,
  getMessageRawMock,
  getRolesMock,
  getScanStatusMock,
  getSettingsMock,
  listChatUploadsMock,
  listFileReindexOperationsMock,
  setChatTitleMock,
  stopChatStreamMock,
  streamChatMock,
  updateChatPreferencesMock,
  updateCurrentChatMock,
  updateSettingsMock,
  uploadChatFileMock,
  deleteChatUploadMock,
} = vi.hoisted(() => ({
  deleteChatMock: vi.fn(),
  getChatMock: vi.fn(),
  getCurrentChatMock: vi.fn(),
  getFilesMock: vi.fn(),
  getMessageRawMock: vi.fn(),
  getRolesMock: vi.fn(),
  getScanStatusMock: vi.fn(),
  getSettingsMock: vi.fn(),
  listChatUploadsMock: vi.fn(),
  listFileReindexOperationsMock: vi.fn(),
  setChatTitleMock: vi.fn(),
  stopChatStreamMock: vi.fn(),
  streamChatMock: vi.fn(),
  updateChatPreferencesMock: vi.fn(),
  updateCurrentChatMock: vi.fn(),
  updateSettingsMock: vi.fn(),
  uploadChatFileMock: vi.fn(),
  deleteChatUploadMock: vi.fn(),
}))

vi.mock('../../api', () => {
  class MockApiError extends Error {
    status: number
    detail: string

    constructor(message: string, status: number, detail: string) {
      super(message)
      this.name = 'ApiError'
      this.status = status
      this.detail = detail
    }
  }

  return {
    ApiError: MockApiError,
    deleteChat: deleteChatMock,
    deleteChatUpload: deleteChatUploadMock,
    getChat: getChatMock,
    getCurrentChat: getCurrentChatMock,
    getFiles: getFilesMock,
    getMessageRaw: getMessageRawMock,
    getRoles: getRolesMock,
    getScanStatus: getScanStatusMock,
    getSettings: getSettingsMock,
    listChatUploads: listChatUploadsMock,
    listFileReindexOperations: listFileReindexOperationsMock,
    setChatTitle: setChatTitleMock,
    stopChatStream: stopChatStreamMock,
    streamChat: streamChatMock,
    updateChatPreferences: updateChatPreferencesMock,
    updateCurrentChat: updateCurrentChatMock,
    updateSettings: updateSettingsMock,
    uploadChatFile: uploadChatFileMock,
  }
})

describe('History open + sidebar navigation integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getSettingsMock.mockResolvedValue({ enable_raw_output_control: false, enable_chat_roles: true })
    getCurrentChatMock.mockResolvedValue({ current_chat_id: undefined })
    getFilesMock.mockResolvedValue({ files: [] })
    getMessageRawMock.mockResolvedValue({ raw_content: null })
    getRolesMock.mockResolvedValue([])
    getScanStatusMock.mockResolvedValue({ status: 'completed' })
    listChatUploadsMock.mockResolvedValue({ chat_id: 'chat-history-integration-1', attachments: [] })
    listFileReindexOperationsMock.mockResolvedValue({ running_count: 0, operations: [] })
    updateSettingsMock.mockResolvedValue({})
    updateCurrentChatMock.mockResolvedValue({})
    updateChatPreferencesMock.mockResolvedValue({})
    stopChatStreamMock.mockResolvedValue({ status: 'not_found', stopped: false })
    streamChatMock.mockResolvedValue(undefined)
    setChatTitleMock.mockResolvedValue({})
    deleteChatMock.mockResolvedValue({})
    uploadChatFileMock.mockResolvedValue({ chat_id: 'chat-history-integration-1', attachment: null })
    deleteChatUploadMock.mockResolvedValue({})
  })

  it('keeps chat usable after history-open, sidebar navigation away, and return to chat', async () => {
    getChatMock.mockResolvedValue({
      messages: [
        {
          id: 1201,
          role: 'user',
          content: 'History question',
          chat_mode: 'researcher',
          role_id: null,
          sources: [],
          created_at: '2026-05-10T10:00:00.000Z',
        },
        {
          id: 1202,
          role: 'assistant',
          content: 'History answer from selected chat',
          chat_mode: 'researcher',
          role_id: null,
          sources: [],
          created_at: '2026-05-10T10:00:02.000Z',
        },
      ],
    })

    render(
      <MemoryRouter initialEntries={['/history']}>
        <ConfirmProvider>
          <ChatProvider>
            <div>
              <Sidebar collapsed={false} onToggleCollapsed={() => {}} />
              <Routes>
                <Route
                  path="/history"
                  element={(
                    <HistoryTable
                      chats={[
                        {
                          chat_id: 'chat-history-integration-1',
                          title: 'History Integration Chat',
                          first_user_message: 'History question',
                          last_message_preview: 'History answer from selected chat',
                          message_count: 2,
                          last_message_at: '2026-05-10T10:00:02.000Z',
                          updated_at: '2026-05-10T10:00:02.000Z',
                          last_generation_seconds: 1.1,
                        },
                      ]}
                    />
                  )}
                />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/files" element={<div>Files Page</div>} />
              </Routes>
            </div>
          </ChatProvider>
        </ConfirmProvider>
      </MemoryRouter>,
    )

    fireEvent.click(screen.getByLabelText('Open chat History Integration Chat'))
    await waitFor(() => expect(getChatMock).toHaveBeenCalledWith('chat-history-integration-1'))
    expect(await screen.findByText('History answer from selected chat')).toBeInTheDocument()

    const sidebarNav = document.querySelector('.sidebar__nav')
    expect(sidebarNav).toBeTruthy()
    const sidebar = within(sidebarNav as HTMLElement)

    fireEvent.click(sidebar.getByRole('button', { name: /^Files$/i }))
    expect(await screen.findByText('Files Page')).toBeInTheDocument()

    fireEvent.click(sidebar.getByRole('button', { name: /^Chat$/i }))
    expect(await screen.findByText('History answer from selected chat')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Chat message input'), { target: { value: 'Follow-up after sidebar navigation' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(streamChatMock).toHaveBeenCalledTimes(1))
    expect(streamChatMock.mock.calls[0][1]).toBe('chat-history-integration-1')
    expect(streamChatMock.mock.calls[0][3]).toMatchObject({ mode: 'researcher' })
  })
})
