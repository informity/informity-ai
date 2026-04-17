interface BackendStartPayload {
  baseUrl: string
  sessionToken: string
  port: number
  launchMode: string
}

interface StartupStatusEventPayload {
  message?: unknown
}

interface MenuActionEventPayload {
  action?: unknown
}

type StartupStatusCallback = (message: string) => void
type MenuActionCallback = (action: string) => void

const BACKEND_STARTUP_STATUS_EVENT = 'informity://backend-startup-status'
const MENU_ACTION_EVENT = 'informity://menu-action'

function formatUnknownError(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  if (typeof error === 'string' && error.trim().length > 0) {
    return error
  }
  if (typeof error === 'object' && error !== null) {
    const candidate = error as Record<string, unknown>
    const directMessage = candidate.message
    if (typeof directMessage === 'string' && directMessage.trim().length > 0) {
      return directMessage
    }
    const causeMessage = candidate.cause
    if (typeof causeMessage === 'string' && causeMessage.trim().length > 0) {
      return causeMessage
    }
    try {
      return JSON.stringify(error)
    } catch {
      return String(error)
    }
  }
  return String(error)
}

function hasTauriInvoke(): boolean {
  const tauriApi = (window as Window & { __TAURI__?: { core?: { invoke?: unknown } } }).__TAURI__
  return typeof tauriApi?.core?.invoke === 'function'
}

async function listenStartupStatus(
  onStatus?: StartupStatusCallback,
): Promise<(() => void) | null> {
  if (!onStatus) {
    return null
  }

  const listen = (window as Window & {
    __TAURI__?: {
      event?: {
        listen?: (
          event: string,
          handler: (payload: { payload?: unknown }) => void,
        ) => Promise<() => void>
      }
    }
  }).__TAURI__?.event?.listen

  if (typeof listen !== 'function') {
    return null
  }

  try {
    return await listen(BACKEND_STARTUP_STATUS_EVENT, (event) => {
      const payload = event.payload as StartupStatusEventPayload | undefined
      if (typeof payload?.message === 'string' && payload.message.trim().length > 0) {
        onStatus(payload.message)
      }
    })
  } catch {
    return null
  }
}

async function invokeTauri<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  const invoke = (window as Window & {
    __TAURI__?: { core?: { invoke?: (cmd: string, params?: Record<string, unknown>) => Promise<unknown> } }
  }).__TAURI__?.core?.invoke

  if (typeof invoke !== 'function') {
    throw new Error('Tauri invoke API is unavailable')
  }

  return invoke(command, args) as Promise<T>
}

export function isDesktopRuntime(): boolean {
  return hasTauriInvoke()
}

export async function nativePickDirectoryDialog(title = 'Choose Folder'): Promise<string | null> {
  if (!isDesktopRuntime()) {
    return null
  }
  try {
    const dialog = await import('@tauri-apps/plugin-dialog')
    const selected = await dialog.open({
      title,
      directory: true,
      multiple: false,
    })
    return typeof selected === 'string' ? selected : null
  } catch {
    return null
  }
}

export async function openExternalUrl(rawUrl: string): Promise<boolean> {
  const url = String(rawUrl || '').trim()
  if (!url) return false

  if (isDesktopRuntime()) {
    try {
      await invokeTauri<void>('plugin:shell|open', { path: url })
      return true
    } catch {
      // Fall through to browser fallback.
    }
  }

  const opened = window.open(url, '_blank', 'noopener,noreferrer')
  return opened !== null
}

export async function bootstrapDesktopBackend(onStatus?: StartupStatusCallback): Promise<void> {
  if (!isDesktopRuntime()) {
    window.__INFORMITY_DESKTOP__ = false
    return
  }

  const unlisten = await listenStartupStatus(onStatus)

  try {
    const payload = await invokeTauri<BackendStartPayload>('backend_start')
    window.__INFORMITY_DESKTOP__ = true
    window.__INFORMITY_API_BASE__ = payload.baseUrl
    window.__INFORMITY_API_TOKEN__ = payload.sessionToken
  } catch (error) {
    const detail = formatUnknownError(error)
    throw new Error(`Backend startup failed: ${detail}`)
  } finally {
    if (unlisten) {
      try {
        unlisten()
      } catch {
        // ignore unlisten errors
      }
    }
  }
}

export async function setMenuBarIconEnabled(enabled: boolean): Promise<void> {
  if (!isDesktopRuntime()) {
    return
  }
  await invokeTauri<void>('set_menu_bar_icon_enabled', { enabled })
}

export async function listenDesktopMenuActions(
  onAction: MenuActionCallback,
): Promise<(() => void) | null> {
  if (!isDesktopRuntime()) {
    return null
  }
  const listen = (window as Window & {
    __TAURI__?: {
      event?: {
        listen?: (
          event: string,
          handler: (payload: { payload?: unknown }) => void,
        ) => Promise<() => void>
      }
    }
  }).__TAURI__?.event?.listen

  if (typeof listen !== 'function') {
    return null
  }

  try {
    return await listen(MENU_ACTION_EVENT, (event) => {
      const payload = event.payload as MenuActionEventPayload | undefined
      if (typeof payload?.action === 'string' && payload.action.trim().length > 0) {
        onAction(payload.action)
      }
    })
  } catch {
    return null
  }
}
