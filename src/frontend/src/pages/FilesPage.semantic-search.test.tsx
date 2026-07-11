import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FilesPage } from './FilesPage'

const {
  getFilesMock,
  getFileTypesMock,
  searchFilesMock,
  listFileReindexOperationsMock,
  getFileReindexOperationMock,
  openFileMock,
  reindexFileMock,
  removeFileMock,
  confirmMock,
} = vi.hoisted(() => ({
  getFilesMock: vi.fn(),
  getFileTypesMock: vi.fn(),
  searchFilesMock: vi.fn(),
  listFileReindexOperationsMock: vi.fn(),
  getFileReindexOperationMock: vi.fn(),
  openFileMock: vi.fn(async (path: string) => {
    void path
    return {}
  }),
  reindexFileMock: vi.fn(async () => ({ operation_id: 'op-1' })),
  removeFileMock: vi.fn(async () => ({})),
  confirmMock: vi.fn(async () => true),
}))

vi.mock('../api', () => ({
  getFiles: (...args: unknown[]) => getFilesMock(...args),
  getFileTypes: (...args: unknown[]) => getFileTypesMock(...args),
  searchFiles: (...args: unknown[]) => searchFilesMock(...args),
  listFileReindexOperations: (...args: unknown[]) => listFileReindexOperationsMock(...args),
  getFileReindexOperation: (...args: unknown[]) => getFileReindexOperationMock(...args),
  openFile: (...args: unknown[]) => openFileMock(...args),
  reindexFile: (...args: unknown[]) => reindexFileMock(...args),
  removeFile: (...args: unknown[]) => removeFileMock(...args),
}))

vi.mock('../context/useBackendStatus', () => ({
  useBackendStatus: () => ({ offline: false }),
}))

vi.mock('../context/useConfirm', () => ({
  useConfirm: () => confirmMock,
}))

describe('FilesPage semantic search', () => {
  afterEach(() => {
    cleanup()
    localStorage.clear()
    getFilesMock.mockReset()
    getFileTypesMock.mockReset()
    searchFilesMock.mockReset()
    listFileReindexOperationsMock.mockReset()
    getFileReindexOperationMock.mockReset()
    openFileMock.mockClear()
    reindexFileMock.mockClear()
    removeFileMock.mockClear()
    confirmMock.mockClear()
  })

  it('switches into semantic search mode and renders ranked file matches', async () => {
    getFilesMock.mockResolvedValue({
      files: [
        {
          id: 1,
          path: '/docs/quarterly-report.pdf',
          filename: 'Quarterly report.pdf',
          extension: '.pdf',
          size_bytes: 1024,
          content_hash: 'abc123',
          extracted_text_preview: 'Revenue overview',
          category: 'document',
          tags: [],
          modified_at: '2025-01-01T00:00:00Z',
          indexed_at: '2025-01-02T00:00:00Z',
        },
      ],
      total: 1,
    })
    getFileTypesMock.mockResolvedValue([])
    listFileReindexOperationsMock.mockResolvedValue({ status: 'ok', running_count: 0, operations: [] })
    searchFilesMock.mockResolvedValue({
      query: 'revenue forecast',
      total: 1,
      results: [
        {
          file_id: 1,
          filename: 'Quarterly report.pdf',
          path: '/docs/quarterly-report.pdf',
          extension: '.pdf',
          size_bytes: 1024,
          indexed_at: '2025-01-02T00:00:00Z',
          modified_at: '2025-01-01T00:00:00Z',
          content_hash: 'abc123',
          extracted_text_preview: 'Revenue overview',
          preview: 'Revenue increased year over year.',
          score: 0.18,
          category: 'document',
          chunk_id: 99,
          page_number: 12,
          section_path: 'Finance > Revenue',
          block_type: 'table',
        },
      ],
    })

    render(
      <MemoryRouter initialEntries={['/files']}>
        <FilesPage />
      </MemoryRouter>,
    )

    await waitFor(() => expect(getFilesMock).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Semantic' }))

    const searchInput = screen.getByPlaceholderText('Search document meaning…')
    fireEvent.change(searchInput, { target: { value: 'revenue forecast' } })

    await waitFor(() => {
      expect(searchFilesMock).toHaveBeenCalledWith({
        query: 'revenue forecast',
        limit: 20,
        fileTypes: undefined,
      })
    })

    await waitFor(() => expect(screen.getByText('Quarterly report.pdf')).toBeInTheDocument())
    expect(screen.getByText('Document')).toBeInTheDocument()
  })
})
