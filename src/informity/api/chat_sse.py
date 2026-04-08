# ==============================================================================
# Informity AI — Chat SSE Helpers
# Contract tracking and status event emission helpers.
# ==============================================================================

import time

import structlog

from informity.utils.json_utils import serialize_api_response

SSE_PHASE_ORDER = {
    'chat': 1,
    'plan_step': 1,
    'token': 2,
    'budget': 2,
    'timeout': 2,
    'sources': 3,
    'cleaned': 4,
    'error': 4,
    'done': 5,
}

SSE_STATUS_ORDER = {
    'classifying': 1,
    'retrieving': 2,
    'searching': 3,
    'generating': 4,
    'continuing': 5,
    'finalizing': 6,
}

_log = structlog.get_logger(__name__)


class SseContractTracker:
    # Tracks expected SSE phase progression and reports out-of-order events.
    def __init__(self) -> None:
        self.current_phase = 0

    def update(self, event_name: str) -> bool:
        event_phase = SSE_PHASE_ORDER.get(event_name, 0)
        if event_phase < self.current_phase:
            return False
        self.current_phase = event_phase
        return True


class SseStatusEmitter:
    def __init__(self, *, chat_id: str, start_time: float) -> None:
        self.chat_id = chat_id
        self.start_time = start_time
        self.current_status_phase = 0
        self.current_status_state: str | None = None
        self.status_transitions: list[dict[str, object]] = []

    def build_event(
        self,
        state: str,
        *,
        message: str,
        pass_index: int | None = None,
        pass_total: int | None = None,
        section_progress: dict[str, object] | None = None,
        allow_same_state: bool = False,
    ) -> dict | None:
        if state == self.current_status_state and not allow_same_state:
            return None
        state_phase = SSE_STATUS_ORDER.get(state)
        if state_phase is None:
            return None
        if state_phase < self.current_status_phase:
            _log.warning(
                'chat_status_out_of_order',
                chat_id=self.chat_id,
                status_state=state,
                current_status_state=self.current_status_state,
                current_status_phase=self.current_status_phase,
                status_phase=state_phase,
            )
            return None
        self.current_status_phase = state_phase
        self.current_status_state = state
        payload: dict[str, object] = {
            'state': state,
            'message': message,
        }
        if pass_index is not None:
            payload['pass_index'] = pass_index
        if pass_total is not None:
            payload['pass_total'] = pass_total
        if section_progress is not None:
            payload['section_progress'] = section_progress
        self.status_transitions.append({
            'state': state,
            'elapsed_seconds': round(time.time() - self.start_time, 3),
            'pass_index': pass_index,
            'pass_total': pass_total,
            'section_progress_emitted': section_progress is not None,
        })
        return {'event': 'status', 'data': serialize_api_response(payload)}
