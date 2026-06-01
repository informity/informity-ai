import { useContext } from 'react'
import { TranslateContext } from './translateContext'

export function useTranslateContext() {
  const ctx = useContext(TranslateContext)
  if (!ctx) throw new Error('useTranslateContext must be used inside TranslateProvider')
  return ctx
}
