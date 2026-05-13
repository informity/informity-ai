import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ChatMessage } from './ChatMessage'

describe('ChatMessage edit interactions', () => {
  afterEach(() => {
    cleanup()
  })

  it('shows pencil control only when editable', () => {
    const { rerender } = render(
      <ChatMessage
        role="user"
        content="Editable user text"
        canEdit
      />,
    )

    expect(screen.getByRole('button', { name: 'Edit message' })).toBeInTheDocument()

    rerender(
      <ChatMessage
        role="user"
        content="Not editable user text"
        canEdit={false}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Edit message' })).toBeNull()
  })

  it('enters edit mode and submits via Enter with trimmed text', async () => {
    const onEditSubmit = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)

    render(
      <ChatMessage
        role="user"
        content="Original text"
        canEdit
        onEditSubmit={onEditSubmit}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit message' }))
    const textarea = screen.getByRole('textbox', { name: 'Edit message' })
    fireEvent.change(textarea, { target: { value: '  Updated text  ' } })
    fireEvent.keyDown(textarea, { key: 'Enter' })

    await waitFor(() => expect(onEditSubmit).toHaveBeenCalledTimes(1))
    const firstSubmitArg = onEditSubmit.mock.calls[0]?.[0]
    expect(firstSubmitArg).toBe('Updated text')
    await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Edit message' })).toBeNull())
  })

  it('submits via send button and does not submit empty drafts', async () => {
    const onEditSubmit = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)

    render(
      <ChatMessage
        role="user"
        content="Original text"
        canEdit
        onEditSubmit={onEditSubmit}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit message' }))
    const textarea = screen.getByRole('textbox', { name: 'Edit message' })
    fireEvent.change(textarea, { target: { value: '   ' } })
    expect(screen.getByRole('button', { name: 'Submit edited message' })).toBeDisabled()
    expect(onEditSubmit).not.toHaveBeenCalled()

    fireEvent.change(textarea, { target: { value: 'New value' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit edited message' }))

    await waitFor(() => expect(onEditSubmit).toHaveBeenCalledTimes(1))
    expect(onEditSubmit).toHaveBeenCalledWith('New value')
  })

  it('does not submit on Shift+Enter and keeps editor open', () => {
    const onEditSubmit = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)

    render(
      <ChatMessage
        role="user"
        content="Original text"
        canEdit
        onEditSubmit={onEditSubmit}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit message' }))
    const textarea = screen.getByRole('textbox', { name: 'Edit message' })
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })

    expect(onEditSubmit).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox', { name: 'Edit message' })).toBeInTheDocument()
  })

  it('cancels edit on Escape and calls cancel callback', () => {
    const onEditCancel = vi.fn()

    render(
      <ChatMessage
        role="user"
        content="Original text"
        canEdit
        onEditCancel={onEditCancel}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit message' }))
    const textarea = screen.getByRole('textbox', { name: 'Edit message' })
    fireEvent.change(textarea, { target: { value: 'Changed draft' } })
    fireEvent.keyDown(textarea, { key: 'Escape' })

    expect(onEditCancel).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('textbox', { name: 'Edit message' })).toBeNull()
    expect(screen.getByText('Original text')).toBeInTheDocument()
  })
})
