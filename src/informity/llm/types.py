from enum import StrEnum


class QueryType(StrEnum):
    METADATA = 'metadata'
    SIMPLE = 'simple'
    FOCUSED = 'focused'
    COVERAGE = 'coverage'


IntentLabel = QueryType


class IntentProfileId(StrEnum):
    METADATA_INVENTORY = 'metadata_inventory'
    TARGETED_FACT_LOOKUP = 'targeted_fact_lookup'
    STRUCTURED_FIELD_EXTRACTION = 'structured_field_extraction'
    CROSS_DOCUMENT_SYNTHESIS = 'cross_document_synthesis'
    COMPARATIVE_ANALYSIS = 'comparative_analysis'
    AUDIT_OR_COMPLIANCE_BRIEF = 'audit_or_compliance_brief'
    CONTINUATION_OR_REFINEMENT = 'continuation_or_refinement'
    CLARIFICATION_OR_DISAMBIGUATION = 'clarification_or_disambiguation'


class RetrievalMode(StrEnum):
    FOCUSED = 'focused'
    COVERAGE = 'coverage'


class OutputShape(StrEnum):
    STRUCTURED_EXTRACT = 'structured_extract'
    NARRATIVE_SYNTHESIS = 'narrative_synthesis'
    METADATA_TABLE = 'metadata_table'
    HYBRID = 'hybrid'


class ConfidenceBand(StrEnum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class QuerySubtype(StrEnum):
    EXTRACT_STRUCTURED_VALUES = 'extract_structured_values'
    AGGREGATE_BY_PERIOD = 'aggregate_by_period'
    FILE_INVENTORY = 'file_inventory'


class GroupBy(StrEnum):
    YEAR = 'year'
    CATEGORY = 'category'
    FILE = 'file'


class BlockType(StrEnum):
    TABLE = 'table'
    FORM = 'form'
    NARRATIVE = 'narrative'


class CompletionMode(StrEnum):
    COMPLETE = 'complete'
    PARTIAL = 'partial'
    SCOPED_COMPLETE = 'scoped_complete'
    STOPPED = 'stopped'


class NextAction(StrEnum):
    NONE = 'none'
    CONTINUE = 'continue'
    REGENERATE = 'regenerate'
    ASSISTANT_SWITCH = 'assistant_switch'


class TimeoutReason(StrEnum):
    QUEUE_WAIT_TIMEOUT = 'queue_wait_timeout'
    FIRST_TOKEN_WATCHDOG_TIMEOUT = 'first_token_watchdog_timeout'
    WALL_CLOCK_LIMIT = 'wall_clock_limit'
    UNKNOWN_TIMEOUT = 'unknown_timeout'


class FallbackReason(StrEnum):
    RESPONSE_SHAPE_NOT_ALLOWED_FOR_PROFILE = 'response_shape_not_allowed_for_profile'
    LOW_CONFIDENCE_ROUTE_GUARD = 'low_confidence_route_guard'
    CONTINUATION_ANCHOR_BIAS_APPLIED = 'continuation_anchor_bias_applied'
    RETRY_WITHOUT_MAX_SCORE = 'retry_without_max_score'
    VALIDATION_GATE_FAILED = 'validation_gate_failed'
    PRESERVE_COVERAGE_QUERY_TYPE_AFTER_FALLBACK = 'preserve_coverage_query_type_after_fallback'
    EMPTY_RETRIEVAL_RESULT = 'empty_retrieval_result'
    SCHEMA_CONTRACT_EMPTY_RETRIEVAL_RECOVERY = 'schema_contract_empty_retrieval_recovery'
    FOCUSED_ANCHOR_EMPTY_RETRIEVAL_RECOVERY = 'focused_anchor_empty_retrieval_recovery'
    FOCUSED_YEAR_SCOPE_EMPTY_RETRIEVAL_RECOVERY = 'focused_year_scope_empty_retrieval_recovery'
    COVERAGE_EVIDENCE_FLOOR_OVERRIDE = 'coverage_evidence_floor_override'
    FOCUSED_STRUCTURED_EVIDENCE_FLOOR_OVERRIDE = 'focused_structured_evidence_floor_override'
    SCHEMA_DRIVEN_GATE_BYPASS = 'schema_driven_gate_bypass'
    CONTINUATION_ANCHOR_GATE_BYPASS = 'continuation_anchor_gate_bypass'
    STRUCTURED_EXTRACTION_INSUFFICIENT = 'structured_extraction_insufficient'
    PRE_CLOSEOUT_QUALITY_CHECK_FAILED = 'insufficient_relevance_under_budget_pressure'


class FilterOperator(StrEnum):
    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GTE = 'gte'
    LT = 'lt'
    LTE = 'lte'
    IN = 'in'
    LIKE = 'like'
    CONTAINS_ANY = 'contains_any'


class ChatRole(StrEnum):
    USER = 'user'
    ASSISTANT = 'assistant'


class StreamSignalTag(StrEnum):
    CLASSIFICATION = '__classification__'
    SEARCHING_STATUS = '__searching_status__'
    TIMEOUT = '__timeout__'
    BUDGET_CHECKPOINT = '__budget_checkpoint__'
    PLAN_STEP = '__plan_step__'
    METRICS = '__metrics__'
    STREAM_SUMMARY = '__stream_summary__'
    FINISH_REASON = '__finish_reason__'


class StructuralGapReason(StrEnum):
    UNCLOSED_CODE_FENCE = 'unclosed_code_fence'
    TRUNCATED_MARKDOWN_TABLE_ROW = 'truncated_markdown_table_row'
    TRUNCATED_MARKDOWN_LIST_ITEM = 'truncated_markdown_list_item'


class ContinuationResolutionReason(StrEnum):
    DUPLICATE_CONTINUATION_DETECTED = 'duplicate_continuation_detected'
    CONTINUATION_PASS_BUDGET_EXHAUSTED = 'continuation_pass_budget_exhausted'
    STALLED = 'stalled'
    TIMEOUT = 'timeout'
    UNRESOLVED_CONTENT = 'unresolved_content'
    STOPPED = 'stopped'


class DiagnosticsQueryType(StrEnum):
    SIMPLE = 'simple'
    METADATA = 'metadata'
    FOCUSED = 'focused'
    COVERAGE = 'coverage'
    UNKNOWN = 'unknown'


__all__ = [
    'BlockType',
    'ChatRole',
    'CompletionMode',
    'ContinuationResolutionReason',
    'ConfidenceBand',
    'DiagnosticsQueryType',
    'FallbackReason',
    'FilterOperator',
    'GroupBy',
    'IntentLabel',
    'IntentProfileId',
    'NextAction',
    'OutputShape',
    'QuerySubtype',
    'QueryType',
    'RetrievalMode',
    'StreamSignalTag',
    'StructuralGapReason',
    'TimeoutReason',
]
