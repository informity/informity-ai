import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FileSearchResults } from './FileSearchResults'

const openFileMock = vi.fn(async (path: string) => {
  void path
  return {}
})

vi.mock('../../api', () => ({
  openFile: (path: string) => openFileMock(path),
}))

vi.mock('../../context/useBackendStatus', () => ({
  useBackendStatus: () => ({ offline: false }),
}))

afterEach(() => {
  cleanup()
  openFileMock.mockClear()
})

describe('FileSearchResults', () => {
  it('renders file-table-like semantic result rows', () => {
    render(
      <FileSearchResults
        results={[
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
        ]}
      />,
    )

    expect(screen.getByText('Quarterly report.pdf')).toBeInTheDocument()
    expect(screen.getByText('Document')).toBeInTheDocument()
  })

  it('opens the underlying file from a semantic result row', async () => {
    render(
      <FileSearchResults
        results={[
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
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open Quarterly report.pdf' }))

    await waitFor(() => {
      expect(openFileMock).toHaveBeenCalledWith('/docs/quarterly-report.pdf')
    })
  })
})
