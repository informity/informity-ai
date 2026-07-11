/**
 * Informity AI — Semantic file search results
 * Grid/list wrapper for semantic search result cards.
 */
import { FileSearchResultCard } from './FileSearchResultCard'
import type { FileSearchResult } from '../../types/api'
import { CenteredState } from '../CenteredState'
import { Skeleton } from '../Skeleton'
import './FileSearchResults.css'

interface FileSearchResultsProps {
  results: FileSearchResult[]
  loading?: boolean
}

export function FileSearchResults({ results, loading = false }: FileSearchResultsProps) {
  if (loading) {
    return (
      <div className="file-search-results">
        <div className="file-search-results__skeleton-grid" aria-label="Loading semantic search results">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="file-search-results__skeleton-card">
              <div className="file-search-results__skeleton-header">
                <Skeleton width={28} height={28} />
                <div className="file-search-results__skeleton-lines">
                  <Skeleton width="70%" height={14} />
                  <Skeleton width="48%" height={10} />
                </div>
              </div>
              <Skeleton width="100%" height={12} style={{ marginTop: '0.85rem' }} />
              <Skeleton width="85%" height={12} style={{ marginTop: '0.5rem' }} />
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <CenteredState
        icon="ri-search-line"
        title="No semantic matches."
        description="Try broader words or switch back to filename search."
        className="file-search-results__empty"
      />
    )
  }

  return (
    <div className="file-search-results">
      <div className="file-search-results__grid">
        {results.map((result) => (
          <FileSearchResultCard key={`${result.file_id}-${result.chunk_id ?? 'chunk'}`} result={result} />
        ))}
      </div>
    </div>
  )
}
