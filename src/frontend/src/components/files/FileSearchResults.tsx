/**
 * Informity AI — Semantic file search results
 * Table wrapper for semantic file search results.
 */
import { FileSearchResultsTable } from './FileSearchResultsTable'
import type { FileSearchResult, IndexedFile } from '../../types/api'
import { CenteredState } from '../CenteredState'
import { FileTableSkeleton } from './FileTableSkeleton'
import './FileSearchResults.css'

type SemanticSearchRow = IndexedFile & {
  score?: number
  preview?: string
  page_number?: number | null
  section_path?: string | null
  block_type?: string | null
}

function toRelevanceScore(distance: number): number {
  if (!Number.isFinite(distance)) return 0
  return Math.max(0, Math.min(100, Math.round((1 - distance) * 100)))
}

interface FileSearchResultsProps {
  results: FileSearchResult[]
  files?: IndexedFile[]
  loading?: boolean
  onChatAboutFile?: (file: IndexedFile) => void
  onTranslate?: (file: IndexedFile) => void
  onReindex?: (file: IndexedFile) => void
  onRemove?: (file: IndexedFile, e: React.MouseEvent) => void
  onOpenFile?: (file: IndexedFile) => void | Promise<void>
  reindexingFileIds?: Set<number>
  total?: number
}

function toSemanticRow(result: FileSearchResult): SemanticSearchRow {
  return {
    id: result.file_id,
    path: result.path,
    filename: result.filename,
    extension: result.extension,
    size_bytes: result.size_bytes,
    content_hash: result.content_hash,
    extracted_text_preview: result.extracted_text_preview,
    preview: result.preview,
    category: result.category,
    tags: [],
    indexed_at: result.indexed_at ?? undefined,
    modified_at: result.modified_at,
    score: toRelevanceScore(result.score),
    page_number: result.page_number,
    section_path: result.section_path,
    block_type: result.block_type,
  }
}

export function FileSearchResults({
  results,
  loading = false,
  files = [],
  onChatAboutFile,
  onTranslate,
  onReindex,
  onRemove,
  onOpenFile,
  reindexingFileIds,
  total,
}: FileSearchResultsProps) {
  if (loading) {
    return (
      <div className="file-search-results">
        <FileTableSkeleton />
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

  const filesById = new Map<number, IndexedFile>(files.map((file) => [file.id, file]))
  const semanticFiles = results.map((result) => {
    const existing = filesById.get(result.file_id)
    if (existing) {
      return {
        ...existing,
        score: toRelevanceScore(result.score),
        preview: result.preview,
        page_number: result.page_number,
        section_path: result.section_path,
        block_type: result.block_type,
      }
    }
    return toSemanticRow(result)
  }).sort((left, right) => (right.score ?? 0) - (left.score ?? 0))

  return (
    <div className="file-search-results">
      <FileSearchResultsTable
        files={semanticFiles}
        total={total ?? results.length}
        onChatAboutFile={onChatAboutFile}
        onTranslate={onTranslate}
        onReindex={onReindex}
        onRemove={onRemove}
        onOpenFile={onOpenFile}
        reindexingFileIds={reindexingFileIds}
      />
    </div>
  )
}
