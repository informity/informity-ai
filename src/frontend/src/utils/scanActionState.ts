export const SCAN_ACTION_STATE_EVENT = 'informity:scan-action-state'

export interface ScanActionStateDetail {
  running: boolean
}

export function dispatchScanActionState(running: boolean): void {
  window.dispatchEvent(
    new CustomEvent<ScanActionStateDetail>(SCAN_ACTION_STATE_EVENT, {
      detail: { running },
    }),
  )
}
