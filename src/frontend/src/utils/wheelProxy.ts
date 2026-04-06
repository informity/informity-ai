import type { WheelEvent } from 'react'

/**
 * Route wheel delta to a specific scroll container, even when the cursor is
 * outside that container, while preserving nested scrollables.
 */
export function proxyWheelToContainer(
  e: WheelEvent<HTMLElement>,
  container: HTMLElement | null,
  options?: { excludeSelector?: string },
): void {
  if (!container) return

  const target = e.target as HTMLElement | null
  if (!target) return
  if (options?.excludeSelector && target.closest(options.excludeSelector)) return

  const deltaY = e.deltaY
  if (!Number.isFinite(deltaY) || deltaY === 0) return

  const findScrollableAncestor = (node: HTMLElement): HTMLElement | null => {
    let current: HTMLElement | null = node
    while (current && current !== container) {
      const style = window.getComputedStyle(current)
      const overflowY = style.overflowY
      const isScrollableY = (overflowY === 'auto' || overflowY === 'scroll') && current.scrollHeight > current.clientHeight
      if (isScrollableY) return current
      current = current.parentElement
    }
    return null
  }

  const nestedScrollable = findScrollableAncestor(target)
  if (nestedScrollable) {
    const maxScrollTop = nestedScrollable.scrollHeight - nestedScrollable.clientHeight
    const atTop = nestedScrollable.scrollTop <= 0
    const atBottom = nestedScrollable.scrollTop >= maxScrollTop - 1
    const scrollingDown = deltaY > 0
    const canNestedConsume = scrollingDown ? !atBottom : !atTop
    if (canNestedConsume) return
  }

  e.preventDefault()

  const maxContainerScroll = container.scrollHeight - container.clientHeight
  const next = Math.min(maxContainerScroll, Math.max(0, container.scrollTop + deltaY))
  if (next !== container.scrollTop) {
    container.scrollTop = next
  }
}
