export interface StartupFailureInfo {
  reason: string | null
  detail: string | null
}

export function getStartupErrorMessage(reason?: string | null): string {
  switch (reason) {
    case 'downloading_classifier_model':
      return 'Could not download the classifier model. Please check your network connection.'
    case 'startup_cancelled':
      return 'Startup was canceled. Please relaunch application.'
    case 'startup_sequence':
      return 'Startup failed while preparing the app. Please check the logs and relaunch.'
    default:
      return 'Startup failed. Please check the logs and relaunch.'
  }
}
