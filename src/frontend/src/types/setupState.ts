export const SETUP_STATES = {
  READY: 'ready',
  REQUIRED: 'setup_required',
  IN_PROGRESS: 'setup_in_progress',
  FAILED: 'setup_failed',
} as const

export type SetupState = typeof SETUP_STATES[keyof typeof SETUP_STATES]

export function isSetupBlockingState(
  state: SetupState,
): state is Exclude<SetupState, typeof SETUP_STATES.READY> {
  return state !== SETUP_STATES.READY
}
