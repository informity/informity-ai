/**
 * Informity AI — Dashboard skeleton for loading state
 */
import { Skeleton } from '../Skeleton'
import { PageHeader } from '../PageHeader'
import './DashboardSkeleton.css'

export function DashboardSkeleton() {
  return (
    <div className="page page--dashboard">
      <PageHeader
        title="Dashboard"
        subtitle={<Skeleton width={240} height={18} />}
        icon="ri-layout-grid-line"
      />

      <div className="page__scroll">
        {/* Hero Card Skeleton */}
        <div className="dashboard-skeleton__hero">
          <Skeleton width={100} height={16} />
          <Skeleton width={120} height={48} style={{ marginTop: '1rem', marginBottom: '0.5rem' }} />
          <Skeleton width={200} height={20} />
          <Skeleton width={180} height={14} style={{ marginTop: '0.5rem', marginBottom: '1.5rem' }} />
          <Skeleton width={168} height={32} />
        </div>

        {/* Content Metrics */}
        <div className="dashboard-skeleton__section">
          <Skeleton width={140} height={24} style={{ marginBottom: '1rem' }} />
          <div className="dashboard-skeleton__cards">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="dashboard-skeleton__card">
                <Skeleton width={40} height={40} />
                <div className="dashboard-skeleton__card-content">
                  <Skeleton width={48} height={24} />
                  <Skeleton width={60} height={14} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="dashboard-skeleton__section">
          <Skeleton width={120} height={24} style={{ marginBottom: '1rem' }} />
          <div className="dashboard-skeleton__recent">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} width="100%" height={40} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
