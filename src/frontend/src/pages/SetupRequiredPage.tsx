import './SetupRequiredPage.css'
import './PlaceholderPage.css'
import { SETUP_STATES, type SetupState } from '../types/setupState'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import type { SetupEventResponse, SetupTierOption } from '../api'

type SetupBlockingState = Exclude<SetupState, typeof SETUP_STATES.READY>

interface SetupRequiredPageProps {
  state: SetupBlockingState
  tierOptions: SetupTierOption[]
  machineRamGb: number | null
  recommendedTier: SetupTierOption['tier'] | null
  recommendedReason: string | null
  event: SetupEventResponse | null
  isStarting: boolean
  isActing: boolean
  onStartSetup: (tier: SetupTierOption['tier'], modelFilename: string) => void
  onRetrySetup: () => void
  onCancelDownload: () => void
  onCancel: () => void
}

function getCopy(state: SetupBlockingState): { title: string; description: string } {
  if (state === SETUP_STATES.IN_PROGRESS) {
    return {
      title: 'Downloading your model...',
      description: "Your model is downloading. Keep this window open until it's done.",
    }
  }
  if (state === SETUP_STATES.FAILED) {
    return {
      title: 'Download failed',
      description: 'Something went wrong while downloading your model. Check your internet connection and try again.',
    }
  }
  return {
    title: 'Welcome to Informity AI',
    description: 'You need to download at least one model to get started. Choose one below.',
  }
}

function formatMemoryProfile(value: string): string {
  return value.replace(/\s*RAM$/i, '').trim()
}

function getDisplayTierTitle(option: SetupTierOption): string {
  return option.tier === 'small' ? 'Light' : option.title
}

function formatRecommendation(
  machineRamGb: number | null,
  reason: string | null,
  option: SetupTierOption | undefined,
): ReactNode | null {
  if (!option) return null
  const tierTitle = getDisplayTierTitle(option)
  if (machineRamGb && machineRamGb > 0) {
    return (
      <>
        Your system has <span className="setup-required__reason-emphasis">{machineRamGb} GB</span> of memory.
        {' '}We recommend <span className="setup-required__reason-emphasis">{tierTitle}</span> for the best experience.
      </>
    )
  }
  if (reason && reason.trim().length > 0) {
    return (
      <>
        {reason.trim()} We recommend <span className="setup-required__reason-emphasis">{tierTitle}</span> for the best experience.
      </>
    )
  }
  return <>We recommend <span className="setup-required__reason-emphasis">{tierTitle}</span> for the best experience.</>
}

function getTierDescription(option: SetupTierOption): string {
  if (option.tier === 'quality') {
    return 'Best answer accuracy, slightly slower responses. Ideal for complex tasks.'
  }
  if (option.tier === 'balanced') {
    return 'Good quality with faster responses. A solid all-around choice.'
  }
  return 'Fastest setup, lowest memory footprint. Best for quick tasks and older hardware.'
}

function formatStageLabel(stage: string | null | undefined): string {
  if (!stage || !stage.trim()) return 'Preparing setup...'
  const key = stage.trim().toLowerCase()
  if (key === 'downloading_model') return 'Downloading model...'
  if (key === 'queued') return 'Preparing download...'
  if (key === 'finalizing') return 'Finalizing setup...'
  const normalized = stage
    .trim()
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .toLowerCase()
  return `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}...`
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 KB'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let unit = 0
  let next = value
  while (next >= 1024 && unit < units.length - 1) {
    next /= 1024
    unit += 1
  }
  const precision = next >= 100 || unit === 0 ? 0 : 1
  return `${next.toFixed(precision)} ${units[unit]}`
}

function getFriendlySetupError(error: string | null | undefined): string {
  const fallback = 'Something went wrong while downloading your model. Check your internet connection and try again.'
  if (!error || !error.trim()) return fallback
  const normalized = error.toLowerCase()

  if (
    normalized.includes('full privacy mode is enabled')
    || normalized.includes('models are not cached')
  ) {
    return 'Download could not start. Click Retry to try again.'
  }
  if (
    normalized.includes('enospc')
    || normalized.includes('no space left on device')
    || normalized.includes('disk full')
  ) {
    return 'There is not enough disk space to download this model. Free up space and try again.'
  }
  if (
    normalized.includes('timed out')
    || normalized.includes('timeout')
    || normalized.includes('connection')
    || normalized.includes('network')
    || normalized.includes('temporary failure in name resolution')
    || normalized.includes('name or service not known')
  ) {
    return 'Download failed due to a network issue. Check your internet connection and try again.'
  }
  if (
    normalized.includes('401')
    || normalized.includes('403')
    || normalized.includes('unauthorized')
    || normalized.includes('forbidden')
    || normalized.includes('gated')
    || normalized.includes('repository not found')
  ) {
    return 'Model download is currently unavailable. Please try again.'
  }
  if (
    normalized.includes('huggingface-hub is not installed')
    || normalized.includes("no module named 'httpx'")
  ) {
    return 'A required download component is unavailable. Restart the app and try again.'
  }

  return fallback
}

export function SetupRequiredPage({
  state,
  tierOptions,
  machineRamGb,
  recommendedTier,
  recommendedReason,
  event,
  isStarting,
  isActing,
  onStartSetup,
  onRetrySetup,
  onCancelDownload,
  onCancel,
}: SetupRequiredPageProps) {
  const copy = getCopy(state)
  const sortedTierOptions = useMemo(() => {
    const tierRank: Record<SetupTierOption['tier'], number> = {
      quality: 0,
      balanced: 1,
      small: 2,
    }
    return [...tierOptions].sort((a, b) => tierRank[a.tier] - tierRank[b.tier])
  }, [tierOptions])

  const initialTier = useMemo<SetupTierOption['tier']>(() => {
    const fallback = sortedTierOptions[0]?.tier ?? 'balanced'
    return recommendedTier ?? fallback
  }, [recommendedTier, sortedTierOptions])
  const [selectedTier, setSelectedTier] = useState<SetupTierOption['tier']>(initialTier)
  const [expandedTier, setExpandedTier] = useState<SetupTierOption['tier'] | null>(null)

  useEffect(() => {
    const activeArtifact = String(event?.artifact || '').trim()
    if (!activeArtifact) return
    const shouldSyncSelection = state === SETUP_STATES.IN_PROGRESS
      || state === SETUP_STATES.FAILED
      || (event?.overall_pct ?? 0) > 0
      || Boolean(event?.error)
    if (!shouldSyncSelection) return
    const matched = sortedTierOptions.find((option) => option.model_filename === activeArtifact)
    if (matched && matched.tier !== selectedTier) {
      setSelectedTier(matched.tier)
    }
  }, [event?.artifact, event?.overall_pct, event?.error, selectedTier, sortedTierOptions, state])

  const selectedOption = sortedTierOptions.find((option) => option.tier === selectedTier) ?? sortedTierOptions[0]
  const recommendedOption = sortedTierOptions.find((option) => option.tier === recommendedTier)
  const recommendationText = formatRecommendation(machineRamGb, recommendedReason, recommendedOption)
  const canStart = Boolean(selectedOption) && state !== SETUP_STATES.IN_PROGRESS && !isStarting
  const showProgress = state === SETUP_STATES.IN_PROGRESS || (event?.overall_pct ?? 0) > 0
  const progressPct = Math.max(0, Math.min(100, event?.overall_pct ?? 0))
  const friendlyProgressError = getFriendlySetupError(event?.error)
  const bytesDone = Math.max(0, event?.bytes_done ?? 0)
  const bytesTotal = Math.max(0, event?.bytes_total ?? 0)
  const speedBps = Math.max(0, event?.speed_bps ?? 0)
  const transferLine = bytesTotal > 0
    ? `${formatBytes(bytesDone)} / ${formatBytes(bytesTotal)}`
    : `${formatBytes(bytesDone)} downloaded`
  const speedLine = speedBps > 0 ? `${formatBytes(speedBps)}/s` : null
  const canRetrySetup = state === SETUP_STATES.FAILED && !isActing
  const isDownloadInProgress = state === SETUP_STATES.IN_PROGRESS
  const canCancelDownload = isDownloadInProgress && !isActing && !isStarting
  const isRetryMode = state === SETUP_STATES.FAILED
  const canPrimaryAction = isDownloadInProgress ? canCancelDownload : (isRetryMode ? canRetrySetup : canStart)

  return (
    <div className="setup-required">
      <main className="setup-required__panel">
        <header className="page-header setup-required__welcome">
          <div className="page-header__title-row">
            <span className="setup-required__logo-shell setup-required__logo-shell--welcome" aria-hidden>
              <img src="/logo.png" alt="" className="setup-required__logo" />
            </span>
            <h1 className="page-header__title">{copy.title}</h1>
          </div>
          <p className="page-header__subtitle setup-required__description">{copy.description}</p>
        </header>

        <section className="setup-required__tiers">
          <h2 className="setup-required__tiers-title">
            <i className="ri-robot-2-line" aria-hidden="true" />
            Choose Your Model
          </h2>
          {recommendationText ? <p className="setup-required__reason">{recommendationText}</p> : null}
          <div className="setup-required__tier-grid">
            {sortedTierOptions.map((option) => {
              const checked = option.tier === selectedTier
              const detailsOpen = expandedTier === option.tier
              return (
                <label
                  key={option.tier}
                  className={`setup-tier ui-card${checked ? ' setup-tier--selected' : ''}`}
                >
                  <div className="setup-tier__top">
                    <input
                      type="radio"
                      className="setup-tier__radio-input"
                      name="setup-tier"
                      checked={checked}
                      onChange={() => setSelectedTier(option.tier)}
                      disabled={state === SETUP_STATES.IN_PROGRESS || isStarting}
                    />
                    <span className="setup-tier__radio" aria-hidden>
                      <span className="setup-tier__radio-dot" />
                    </span>
                    <div>
                      <p className="setup-tier__title">
                        {getDisplayTierTitle(option)}
                      </p>
                      <p className="setup-tier__desc">{getTierDescription(option)}</p>
                    </div>
                  </div>
                  <div className="setup-tier__summary">
                    <button
                      type="button"
                      className={`setup-tier__details-toggle-inline${detailsOpen ? ' setup-tier__details-toggle-inline--expanded' : ''}`}
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        setExpandedTier((current) => (current === option.tier ? null : option.tier))
                      }}
                      aria-expanded={detailsOpen}
                    >
                      <i className="ri-arrow-right-s-line setup-tier__details-chevron" aria-hidden />
                      <span>{detailsOpen ? 'Hide Details' : 'Show Details'}</span>
                    </button>
                  </div>
                  {detailsOpen ? (
                    <div className="setup-tier__meta setup-tier__meta--details">
                      <span className="setup-tier__meta-item">
                        <i className="ri-robot-2-line setup-tier__meta-icon" aria-hidden />
                        <span className="setup-tier__meta-label">Model:</span>
                        <span className="setup-tier__meta-value">
                          {option.display_name}
                        </span>
                      </span>
                      <span className="setup-tier__meta-sep" aria-hidden>|</span>
                      <span className="setup-tier__meta-item">
                        <i className="ri-download-2-line setup-tier__meta-icon" aria-hidden />
                        <span className="setup-tier__meta-label">Size:</span>
                        <span className="setup-tier__meta-value">{option.approx_size_gb.toFixed(1)} GB</span>
                      </span>
                      <span className="setup-tier__meta-sep" aria-hidden>|</span>
                      <span className="setup-tier__meta-item">
                        <i className="ri-speed-up-line setup-tier__meta-icon" aria-hidden />
                        <span className="setup-tier__meta-label">Speed:</span>
                        <span className="setup-tier__meta-value">{option.speed}</span>
                      </span>
                      <span className="setup-tier__meta-sep" aria-hidden>|</span>
                      <span className="setup-tier__meta-item">
                        <i className="ri-cpu-line setup-tier__meta-icon" aria-hidden />
                        <span className="setup-tier__meta-label">Memory:</span>
                        <span className="setup-tier__meta-value">{formatMemoryProfile(option.ram_profile)}</span>
                      </span>
                    </div>
                  ) : null}
                  {checked && showProgress ? (
                    <div className="setup-tier__progress">
                      <div className="setup-tier__progress-row">
                        <p className="setup-tier__progress-label">
                          {formatStageLabel(event?.stage)}
                        </p>
                        <p className="setup-tier__progress-value">{progressPct}%</p>
                      </div>
                      <div className="setup-tier__progress-track">
                        <div className="setup-tier__progress-bar" style={{ width: `${progressPct}%` }} />
                      </div>
                      {event?.error || event?.paused || showProgress ? (
                        <p className="setup-tier__progress-meta">
                          {event?.error ? friendlyProgressError : ''}
                          {event?.error && (event?.paused || showProgress) ? ' • ' : ''}
                          {event?.paused ? 'Paused' : ''}
                          {(event?.error || event?.paused) && showProgress ? ' • ' : ''}
                          {showProgress ? transferLine : ''}
                          {showProgress && speedLine ? ` • ${speedLine}` : ''}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                </label>
              )
            })}
          </div>
        </section>

        <footer className="setup-required__actions ui-section-divider">
          <button
            type="button"
            className="settings-btn settings-btn--secondary"
            onClick={onCancel}
            disabled={isActing || isStarting}
          >
            Quit Setup
          </button>
          <button
            type="button"
            className="settings-btn settings-btn--primary"
            disabled={!canPrimaryAction}
            onClick={() => {
              if (isDownloadInProgress) {
                onCancelDownload()
                return
              }
              if (isRetryMode) {
                onRetrySetup()
                return
              }
              if (!selectedOption) return
              onStartSetup(selectedOption.tier, selectedOption.model_filename)
            }}
          >
            {isDownloadInProgress ? 'Cancel Download' : (isStarting ? 'Starting...' : (isRetryMode ? 'Retry' : 'Continue'))}
          </button>
        </footer>
      </main>
    </div>
  )
}
