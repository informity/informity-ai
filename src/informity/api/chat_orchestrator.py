# ==============================================================================
# Informity AI — Chat Orchestrator (Shell)
# Thin orchestration state model for /api/chat runtime decomposition.
# ==============================================================================

from dataclasses import dataclass, field

from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage
from informity.llm.types import (
    CompletionMode,
    ContinuationResolutionReason,
    DiagnosticsQueryType,
    StructuralGapReason,
    TimeoutReason,
)


@dataclass
class ChatOrchestratorState:
    answer_parts: list[str] = field(default_factory=list)
    sources: list[ChatSourceReference] = field(default_factory=list)
    generation_seconds: float | None = None
    timeout_occurred: bool = False
    timeout_reason: TimeoutReason | str | None = None
    assistant_message_id: int | None = None
    assistant_message_record: ChatMessage | None = None
    message_persisted: bool = False
    cleaned_answer: str = ''
    completion_mode_override: CompletionMode | str | None = None
    budget_metrics: dict[str, object] = field(default_factory=dict)
    budget_checkpoints: list[dict[str, object]] = field(default_factory=list)
    has_remaining_scope: bool = False
    stopped_by_user: bool = False
    finalized_sources: bool = False
    finalized_cleaned: bool = False
    metrics_query_type: str = DiagnosticsQueryType.UNKNOWN.value
    metrics_raw_chunks_count: int = 0
    continuation_passes: int = 0
    pass_details: list[dict[str, object]] = field(default_factory=list)
    continuation_resolution_reason: (
        ContinuationResolutionReason | StructuralGapReason | TimeoutReason | str | None
    ) = None
    continuation_progress_state: str | None = None
    status_transitions: list[dict[str, object]] = field(default_factory=list)
    resource_end_snapshot: dict[str, object] | None = None
    resource_metrics: dict[str, object] = field(default_factory=dict)
    previous_pass_raw_answer: str | None = None
    pre_classification_elapsed_ms: float | None = None
    sanitization_elapsed_ms: float | None = None


class ChatOrchestrator:
    def prepare_request(
        self,
        *,
        chat_id: str,
        stream_id: str,
        request_id: str | None,
    ) -> dict:
        return {
            'event': 'chat',
            'data': {
                'chat_id': chat_id,
                'stream_id': stream_id,
                'request_id': request_id if request_id else None,
            },
        }

    def initial_state(self) -> ChatOrchestratorState:
        return ChatOrchestratorState()
