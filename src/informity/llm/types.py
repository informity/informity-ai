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


class OutputFormat(StrEnum):
    TABLE = 'table'
    LIST = 'list'
    BULLETS = 'bullets'
    NARRATIVE = 'narrative'
    CSV = 'csv'


class ConfidenceBand(StrEnum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class QuerySubtype(StrEnum):
    EXTRACT_STRUCTURED_VALUES = 'extract_structured_values'
    AGGREGATE_BY_PERIOD = 'aggregate_by_period'
    COMPARATIVE = 'comparative'


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
    COVERAGE_EVIDENCE_FLOOR_OVERRIDE = 'coverage_evidence_floor_override'


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
    'OutputFormat',
    'OutputShape',
    'QuerySubtype',
    'QueryType',
    'RetrievalMode',
    'StreamSignalTag',
    'StructuralGapReason',
    'TimeoutReason',
]
