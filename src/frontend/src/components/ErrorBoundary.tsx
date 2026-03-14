/**
 * Informity AI — Error boundary
 * Catches React errors and shows "Something went wrong" with retry.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'
import './ErrorBoundary.css'

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, errorInfo)
  }

  handleRetry = (): void => {
    this.setState({ error: null })
  }

  render() {
    if (this.state.error) {
      return (
        <div className="error-boundary">
          <div className="error-boundary__content">
            <i className="ri-error-warning-line error-boundary__icon" aria-hidden style={{ fontSize: '3rem' }} />
            <h2 className="error-boundary__title">Something went wrong</h2>
            <p className="error-boundary__message">
              An unexpected error occurred. You can try again or refresh the page.
            </p>
            <button
              type="button"
              className="error-boundary__retry"
              onClick={this.handleRetry}
            >
              <i className="ri-restart-line" aria-hidden style={{ fontSize: '1.125rem' }} />
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
