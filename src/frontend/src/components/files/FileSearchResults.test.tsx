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
  it('renders semantic result cards with location context', () => {
    render(
      <FileSearchResults
        results={[
          {
            file_id: 1,
            filename: 'Quarterly report.pdf',
            path: '/docs/quarterly-report.pdf',
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
    expect(screen.getByText('Page 12 · Finance > Revenue · table')).toBeInTheDocument()
    expect(screen.getByText('Revenue increased year over year.')).toBeInTheDocument()
    expect(screen.getByText('document')).toBeInTheDocument()
  })

  it('opens the underlying file from a semantic result card', async () => {
    render(
      <FileSearchResults
        results={[
          {
            file_id: 1,
            filename: 'Quarterly report.pdf',
            path: '/docs/quarterly-report.pdf',
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
