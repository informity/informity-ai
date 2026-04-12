/**
 * Informity AI — API types
 * Shared types for API requests and responses.
 */

export interface ChatSourceReference {
  filename: string
  path: string
  chunk_preview: string
  relevance_score: number
  extraction_quality?: string | null
  extraction_note?: string | null
}

export interface DisplayTextBlock {
  type: 'text'
  markdown: string
}

export interface DisplayCodeBlock {
  type: 'code'
  code: string
  language?: string
}

export interface DisplayCalloutBlock {
  type: 'callout'
  text: string
  tone?: 'info' | 'warning' | 'success' | 'danger'
}

export interface DisplayMetricBlock {
  type: 'metric'
  label: string
  value: string
}

export interface DisplayTableBlock {
  type: 'table'
  columns: string[]
  rows: Array<Array<string | number | null>>
}

export interface DisplayUnknownBlock {
  type: string
  [key: string]: unknown
}

export type DisplayBlock =
  | DisplayTextBlock
  | DisplayCodeBlock
  | DisplayCalloutBlock
  | DisplayMetricBlock
  | DisplayTableBlock
  | DisplayUnknownBlock

export type CompletionMode = 'complete' | 'partial' | 'scoped_complete' | 'stopped'
export type ChatMode = 'assistant' | 'researcher'
export function isChatMode(value: unknown): value is ChatMode {
  return value === 'assistant' || value === 'researcher'
}
export type NextAction = 'none' | 'continue' | 'regenerate' | 'assistant_switch'
export type NextActionReason = 'stopped' | 'timeout' | 'unresolved_content' | 'budget_exhausted' | 'stalled' | 'out_of_corpus'
export type StreamStatusState = 'classifying' | 'retrieving' | 'searching' | 'generating' | 'continuing' | 'finalizing'

export interface PlanStepPayload {
  step_id?: number
  description?: string
  status?: 'running' | 'done' | 'empty'
}

export interface StreamStatusPayload {
  state?: StreamStatusState
  message?: string
  pass_index?: number
  pass_total?: number
  section_progress?: {
    completed?: string[]
    remaining?: string[]
    total?: number
  }
}

export interface StreamDonePayload {
  chat_mode?: ChatMode
  elapsed_seconds?: number
  request_id?: string
  timeout_occurred?: boolean
  timeout_reason?: string | null
  message_id?: number
  completion_mode?: CompletionMode
  stopped_by_user?: boolean
  has_remaining_scope?: boolean
  message_persisted?: boolean
  sources_count?: number
  display_blocks?: DisplayBlock[]
  budget_metrics?: Record<string, unknown>
  budget_checkpoints?: Array<Record<string, unknown>>
  web_search_used?: boolean
  continuation_passes?: number
  continuation_resolution_reason?: string | null
  next_action?: NextAction
  next_action_reason?: NextActionReason | null
  pass_details?: Array<Record<string, unknown>>
  status_transitions?: Array<Record<string, unknown>>
}

export interface ChatMessageApi {
  id?: number
  role: string
  content: string
  model_filename?: string | null
  sources?: ChatSourceReference[]
  display_blocks?: DisplayBlock[]
  is_internal?: boolean
  completion_mode?: CompletionMode
  stopped_by_user?: boolean
  has_remaining_scope?: boolean
  next_action?: NextAction
  next_action_reason?: NextActionReason | null
  created_at?: string
  generation_seconds?: number
  chat_mode?: ChatMode
}

export interface ChatMessageDisplay {
  id?: number
  role: string
  content: string
  modelFilename?: string | null
  sources?: ChatSourceReference[]
  displayBlocks?: DisplayBlock[]
  isInternal?: boolean
  isContinuation?: boolean
  isStreaming?: boolean
  streamStatusText?: string
  streamSectionProgress?: {
    completed: string[]
    remaining: string[]
    total: number
  }
  isPartial?: boolean
  hasRemainingScope?: boolean
  completionMode?: CompletionMode
  stoppedByUser?: boolean
  createdAt?: string
  generationSeconds?: number
  chatMode?: ChatMode
  nextAction?: NextAction
  nextActionReason?: NextActionReason | null
  continuationPasses?: number
  continueLabel?: 'Continue' | 'Continue Again'
  webSearchUsed?: boolean
  streamPlanSteps?: Array<{ step_id: number; description: string; status: 'running' | 'done' | 'empty' }>
}

export interface StreamChatCallbacks {
  onToken?: (token: string) => void
  onChatId?: (chatId: string) => void
  onStreamId?: (streamId: string) => void
  onRequestId?: (requestId: string) => void
  onSources?: (sources: ChatSourceReference[]) => void
  onCleaned?: (cleanedAnswer: string) => void
  onStatus?: (status: StreamStatusPayload) => void
  onPlanStep?: (payload: PlanStepPayload) => void
  onDone?: (data?: StreamDonePayload) => void
  onError?: (err: Error) => void
  signal?: AbortSignal
}

export interface IndexedFile {
  id: number
  path: string
  filename: string
  extension: string
  size_bytes: number
  content_hash: string
  extracted_text_preview: string
  category: string
  tags: string[]
  year?: number
  indexed_at?: string
  modified_at: string
  created_at?: string
  chunk_count?: number
}

export interface IndexStatus {
  total_files?: number
  total_chunks?: number
  total_embeddings?: number
  chat_count?: number
  indexed_content_size_bytes?: number
  db_size_bytes?: number
  vectors_size_bytes?: number
  model_size_bytes?: number
  last_scan_at?: string | null
  reset_in_progress?: boolean
  last_reset_result?: Record<string, unknown> | null
  source_scope_stats?: Array<Record<string, unknown>>
}

export interface ScanRecentError {
  path?: string
  filename?: string
  operation?: string
  error_code?: string | null
  error_message?: string
  is_timeout?: boolean
  created_at?: string | null
}

export interface ScanStatus {
  status?: string
  started_at?: string
  files_scanned?: number
  files_indexed?: number
  errors?: number
  timeout_errors?: number
  recent_errors?: ScanRecentError[]
  elapsed_seconds?: number
}

export type FileReindexOperationStatus = 'running' | 'completed' | 'failed'

export interface FileReindexOperation {
  operation_id: string
  operation_type: 'file_reindex'
  file_id: number
  filename: string
  status: FileReindexOperationStatus
  started_at: string
  completed_at?: string | null
  error?: string | null
  chunks_created?: number | null
}

export interface ChatListItem {
  chat_id: string
  title?: string
  last_message_preview?: string
  last_message_at?: string
  updated_at?: string
  message_count?: number
  first_user_message?: string
  last_generation_seconds?: number
}
