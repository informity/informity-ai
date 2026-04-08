/**
 * Informity AI — Page header
 * Reusable header with title, subtitle, divider. Optional action slot (e.g. New Chat button).
 */
import type { ReactNode } from 'react'
import '../pages/PlaceholderPage.css'

interface PageHeaderProps {
  title: string
  subtitle?: ReactNode
  icon?: string
  action?: ReactNode
  className?: string
}

export function PageHeader({
  title,
  subtitle,
  icon = 'ri-file-line',
  action,
  className = '',
}: PageHeaderProps) {
  const headerClass = ['page-header', action && 'page-header--with-action', className].filter(Boolean).join(' ')
  const content = (
    <>
      <div className="page-header__title-row">
        <i className={`${icon} page-header__title-icon`} aria-hidden="true" />
        <h1 className="page-header__title ui-title ui-title--view">{title}</h1>
      </div>
      <div className="page-header__subtitle ui-subtitle">{subtitle}</div>
    </>
  )
  return (
    <header className={headerClass}>
      {action ? (
        <>
          <div className="page-header__content">{content}</div>
          <div className="page-header__action">{action}</div>
        </>
      ) : (
        content
      )}
    </header>
  )
}
