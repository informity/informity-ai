import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { SourceCard } from './SourceCard'

afterEach(() => {
  cleanup()
})

describe('SourceCard evidence rank display', () => {
  it('shows 100 for top-ranked source', () => {
    render(<SourceCard filename="example.pdf" rankIndex={0} rankTotal={5} />)
    expect(screen.getByText('100')).toBeInTheDocument()
  })

  it('shows 20 for last-ranked source', () => {
    render(<SourceCard filename="example.pdf" rankIndex={4} rankTotal={5} />)
    expect(screen.getByText('20')).toBeInTheDocument()
  })

  it('shows 100 for single source', () => {
    render(<SourceCard filename="example.pdf" rankIndex={0} rankTotal={1} />)
    expect(screen.getByText('100')).toBeInTheDocument()
  })
})
