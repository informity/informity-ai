/**
 * Informity AI — Skeleton placeholder for loading states
 */
import './Skeleton.css'

interface SkeletonProps {
  className?: string
  width?: number | string
  height?: number | string
  style?: React.CSSProperties
}

export function Skeleton({ className = '', width, height, style = {} }: SkeletonProps) {
  const s: React.CSSProperties = { ...style }
  if (width !== undefined) s.width = typeof width === 'number' ? `${width}px` : width
  if (height !== undefined) s.height = typeof height === 'number' ? `${height}px` : height
  return <div className={`skeleton ${className}`.trim()} style={s} aria-hidden />
}
