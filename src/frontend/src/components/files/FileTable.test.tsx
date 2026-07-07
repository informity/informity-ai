import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FileTable } from './FileTable'
import type { IndexedFile } from '../../types/api'

const openFileMock = vi.fn(async (path: string) => {
  void path
  return {}
})

vi.mock('../../api', () => ({
  openFile: (path: string) => openFileMock(path),
}))

afterEach(() => {
  cleanup()
  openFileMock.mockClear()
})

describe('FileTable filename links', () => {
  it('opens the underlying file when the filename is clicked', async () => {
    const file = {
      id: 1,
      path: '/Users/example/Documents/report.pdf',
      filename: 'report.pdf',
      extension: 'pdf',
      size_bytes: 1024,
      content_hash: 'abc123',
      extracted_text_preview: 'Preview',
      category: 'document',
      tags: [],
      modified_at: '2025-01-01T00:00:00Z',
      indexed_at: '2025-01-02T00:00:00Z',
    } as IndexedFile

    render(
      <FileTable
        files={[file]}
        total={1}
        offline={false}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open report.pdf' }))

    await waitFor(() => {
      expect(openFileMock).toHaveBeenCalledWith('/Users/example/Documents/report.pdf')
    })
  })
})
