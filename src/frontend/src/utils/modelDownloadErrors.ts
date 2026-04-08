export function getFriendlyModelDownloadError(error: string | null | undefined): string {
  const fallback = 'Something went wrong while downloading your model. Check your internet connection and try again.'
  if (!error || !error.trim()) return fallback
  const normalized = error.toLowerCase()

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
    || normalized.includes('cannot import name')
  ) {
    return 'A required download component is unavailable. Restart the app and try again.'
  }

  return fallback
}

