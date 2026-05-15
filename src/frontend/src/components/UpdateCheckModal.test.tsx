import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { UpdateCheckModal } from './UpdateCheckModal'

afterEach(() => {
  cleanup()
})

describe('UpdateCheckModal', () => {
  it('renders update available state with expected copy and actions', () => {
    render(
      <UpdateCheckModal
        open
        state="update_available"
        checking={false}
        latestVersion="9.9.9"
        currentVersion="0.12.0"
        onClose={vi.fn()}
        onRetry={vi.fn()}
        onDownload={vi.fn()}
      />,
    )

    expect(screen.getByText('Update Available')).toBeInTheDocument()
    expect(screen.getByText('New version 9.9.9 is available for download.')).toBeInTheDocument()
    expect(screen.getByText('Review the latest', { exact: false })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'release notes' })).toHaveAttribute(
      'href',
      'https://github.com/informity/informity-ai/releases',
    )
    expect(screen.getByRole('button', { name: 'Later' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download Update' })).toBeInTheDocument()
  })

  it('renders up-to-date state with expected copy', () => {
    render(
      <UpdateCheckModal
        open
        state="up_to_date"
        checking={false}
        currentVersion="0.12.0"
        onClose={vi.fn()}
        onRetry={vi.fn()}
        onDownload={vi.fn()}
      />,
    )

    expect(screen.getByText("You're up to date")).toBeInTheDocument()
    expect(screen.getByText('Informity AI is up to date (version 0.12.0).')).toBeInTheDocument()
    expect(screen.getByText('Review the latest', { exact: false })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'release notes' })).toHaveAttribute(
      'href',
      'https://github.com/informity/informity-ai/releases',
    )
  })

  it('renders generic error copy and supports retry action', () => {
    const onRetry = vi.fn()
    render(
      <UpdateCheckModal
        open
        state="error"
        checking={false}
        onClose={vi.fn()}
        onRetry={onRetry}
        onDownload={vi.fn()}
      />,
    )

    expect(screen.getByText('Update Check Failed')).toBeInTheDocument()
    expect(screen.getByText('Please try again later.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Try Again' }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('disables action buttons while checking', () => {
    render(
      <UpdateCheckModal
        open
        state="checking"
        checking
        onClose={vi.fn()}
        onRetry={vi.fn()}
        onDownload={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Close' })).toBeDisabled()
  })
})
