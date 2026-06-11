import { fireEvent, render, screen, within } from '@testing-library/react'
import { useState } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import * as api from '../api'

vi.mock('../context/useChatContext', () => ({
  useChatContext: () => ({
    isStreaming: true,
    activeGenerationChatId: 'chat-1',
    goToGeneratingChat: vi.fn(async () => {}),
    stopStreaming: vi.fn(async () => true),
  }),
}))

vi.mock('../api', () => ({
  getScanStatus: vi.fn(async () => ({ status: 'completed' })),
  listFileReindexOperations: vi.fn(async () => ({ running_count: 0, operations: [] })),
}))

import { Sidebar } from './Sidebar'

function SidebarHarness({ initialCollapsed }: { initialCollapsed: boolean }) {
  const [collapsed, setCollapsed] = useState(initialCollapsed)

  return (
    <MemoryRouter initialEntries={['/chat']}>
      <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((value) => !value)} />
    </MemoryRouter>
  )
}

describe('Sidebar', () => {
  it('shows low-noise generating spinner on chat item', () => {
    render(
      <MemoryRouter initialEntries={['/history']}>
        <Sidebar collapsed={false} onToggleCollapsed={() => {}} />
      </MemoryRouter>,
    )

    expect(screen.getByLabelText('Generating')).toBeInTheDocument()
  })

  it('does not render collapsed status indicator dot', () => {
    const { container } = render(
      <MemoryRouter initialEntries={['/history']}>
        <Sidebar collapsed={true} onToggleCollapsed={() => {}} />
      </MemoryRouter>,
    )

    expect(container.querySelector('.sidebar__status-dot')).toBeNull()
  })

  it('shows indexing spinner on files item when file reindex is running', async () => {
    vi.spyOn(api, 'listFileReindexOperations').mockResolvedValueOnce({
      status: 'ok',
      running_count: 1,
      operations: [],
    })

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Sidebar collapsed={false} onToggleCollapsed={() => {}} />
      </MemoryRouter>,
    )

    expect(await screen.findByLabelText('Indexing')).toBeInTheDocument()
  })

  it('shows a scan spinner on the settings item when scanning is running', async () => {
    vi.spyOn(api, 'getScanStatus').mockResolvedValueOnce({
      status: 'running',
    })

    render(
      <MemoryRouter initialEntries={['/settings?section=indexing']}>
        <Sidebar collapsed={false} onToggleCollapsed={() => {}} />
      </MemoryRouter>,
    )

    expect(await screen.findByLabelText('Scanning')).toBeInTheDocument()
  })

  it('expands the sidebar when clicking Settings while collapsed', () => {
    const { container } = render(<SidebarHarness initialCollapsed />)

    const collapsedSidebar = container.querySelector('.sidebar--collapsed')
    expect(collapsedSidebar).toBeTruthy()
    fireEvent.click(within(collapsedSidebar as HTMLElement).getByRole('button', { name: 'Settings' }))

    expect(collapsedSidebar).not.toHaveClass('sidebar--collapsed')
    expect(within(collapsedSidebar as HTMLElement).getByRole('button', { name: 'Toggle sidebar' })).toBeInTheDocument()
    expect(within(collapsedSidebar as HTMLElement).getByText('General')).toBeInTheDocument()
    expect(within(collapsedSidebar as HTMLElement).getByText('Data Sources')).toBeInTheDocument()
  })
})
