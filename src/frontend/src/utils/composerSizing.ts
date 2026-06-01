/**
 * Shared composer sizing utilities.
 * Used by both ChatView (chat) and TranslatePage (translate) to keep
 * textarea auto-resize and chip-row offset logic identical.
 */

export const COMPOSER_TEXTAREA_MIN_HEIGHT = 104
export const COMPOSER_TEXTAREA_MAX_HEIGHT = 304
/** Added to both min and max when a chip row is present above the textarea. */
export const COMPOSER_SCOPED_EXTRA_HEIGHT = 52

/**
 * Auto-resize a composer textarea to fit its content, clamped to min/max.
 * Call on every input event and whenever chip state changes.
 *
 * @param ta       The textarea element.
 * @param isScoped True when a chip row is visible above the textarea (adds
 *                 COMPOSER_SCOPED_EXTRA_HEIGHT to both min and max).
 */
export function resizeComposerTextarea(
  ta: HTMLTextAreaElement,
  isScoped: boolean,
): void {
  ta.style.height = 'auto'
  const natural = ta.scrollHeight
  const min = isScoped
    ? COMPOSER_TEXTAREA_MIN_HEIGHT + COMPOSER_SCOPED_EXTRA_HEIGHT
    : COMPOSER_TEXTAREA_MIN_HEIGHT
  const max = isScoped
    ? COMPOSER_TEXTAREA_MAX_HEIGHT + COMPOSER_SCOPED_EXTRA_HEIGHT
    : COMPOSER_TEXTAREA_MAX_HEIGHT
  ta.style.height = `${Math.min(max, Math.max(min, natural))}px`
  ta.style.overflowY = natural > max ? 'auto' : 'hidden'
}

/**
 * Return the rendered height of a chip row element in pixels, or 0 if the
 * element is null or has no chip children.  Used to compute the CSS custom
 * property `--composer-scoped-top-padding` dynamically.
 */
export function getChipRowHeight(chipRowEl: HTMLElement | null): number {
  if (!chipRowEl) return 0
  if (chipRowEl.childElementCount === 0) return 0
  return chipRowEl.offsetHeight
}

/**
 * Apply `--composer-scoped-top-padding` to a composer input wrapper so the
 * textarea's top padding always matches the actual chip row height.
 *
 * @param wrapperEl  The `.composer__input-wrapper` element.
 * @param chipRowEl  The `.composer__chips` element (or null when no chips).
 * @param baseGap    Extra gap in px between chip row bottom and textarea text.
 */
export function applyComposerScopedPadding(
  wrapperEl: HTMLElement | null,
  chipRowEl: HTMLElement | null,
  baseGap = 8,
): void {
  if (!wrapperEl) return
  const chipHeight = getChipRowHeight(chipRowEl)
  if (chipHeight === 0) {
    wrapperEl.style.removeProperty('--composer-scoped-top-padding')
    return
  }
  const padding = chipHeight + baseGap
  wrapperEl.style.setProperty('--composer-scoped-top-padding', `${padding}px`)
}
