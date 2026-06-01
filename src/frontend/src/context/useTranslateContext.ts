import { useContext } from 'react'
import { TranslateContext } from './translateContext'

export function useTranslateContext() {
  const ctx = useContext(TranslateContext)
  if (!ctx) throw new Error('useTranslateContext must be used inside TranslateProvider')
  return ctx
}

/** Non-throwing variant — returns null when called outside TranslateProvider.
 *  Use in Sidebar and other components that render before the provider mounts. */
export function useOptionalTranslateContext() {
  return useContext(TranslateContext)
}
