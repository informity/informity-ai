/**
 * Informity AI — File table skeleton for loading state
 */
import { Skeleton } from '../Skeleton'
import './FileTableSkeleton.css'

const ROWS = 8

export function FileTableSkeleton() {
  return (
    <div className="file-table-skeleton">
      <div className="file-table-skeleton__header">
        <Skeleton width={24} height={24} />
        <Skeleton width="25%" height={16} />
        <Skeleton width="12%" height={16} />
        <Skeleton width="8%" height={16} />
        <Skeleton width="12%" height={16} />
        <Skeleton width="12%" height={16} />
        <Skeleton width="15%" height={16} />
      </div>
      {Array.from({ length: ROWS }).map((_, i) => (
        <div key={i} className="file-table-skeleton__row">
          <Skeleton width={24} height={24} />
          <Skeleton width="30%" height={14} />
          <Skeleton width="10%" height={14} />
          <Skeleton width="8%" height={14} />
          <Skeleton width="12%" height={14} />
          <Skeleton width="12%" height={14} />
          <Skeleton width="20%" height={14} />
        </div>
      ))}
    </div>
  )
}
