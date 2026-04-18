# ==============================================================================
# Informity AI — Chat Closeout Helpers
# Done-payload assembly and closeout-only numeric evidence helpers.
# ==============================================================================

def build_display_blocks(cleaned_answer: str) -> list[dict[str, str]]:
    if not cleaned_answer:
        return []
    return [{'type': 'text', 'markdown': cleaned_answer}]


def build_done_payload(
    *,
    elapsed_seconds: float | None,
    request_id: str | None,
    chat_mode: str,
    timeout_occurred: bool,
    timeout_reason: str | object | None,
    completion_mode: str | object,
    has_remaining_scope: bool,
    stopped_by_user: bool,
    next_action: str | object,
    next_action_reason: str | None,
    sources_count: int,
    message_persisted: bool,
    cleaned_answer: str,
    budget_metrics: dict[str, object],
    budget_checkpoints: list[dict[str, object]],
    continuation_passes: int,
    continuation_resolution_reason: str | object | None,
    continuation_progress_state: str | None,
    pass_details: list[dict[str, object]],
    status_transitions: list[dict[str, object]],
    resource_metrics: dict[str, object],
    message_id: int | None,
) -> dict:
    payload: dict[str, object] = {
        'elapsed_seconds': elapsed_seconds,
        'request_id': request_id if request_id else None,
        'chat_mode': chat_mode,
        'timeout_occurred': timeout_occurred,
        'timeout_reason': timeout_reason,
        'completion_mode': completion_mode,
        'has_remaining_scope': has_remaining_scope,
        'stopped_by_user': stopped_by_user,
        'next_action': next_action,
        'next_action_reason': next_action_reason,
        'sources_count': sources_count,
        'message_persisted': message_persisted,
        'display_blocks': build_display_blocks(cleaned_answer),
        'budget_metrics': budget_metrics,
        'web_search_used': bool(budget_metrics.get('web_search_used')),
        'budget_checkpoints': budget_checkpoints,
        'continuation_passes': continuation_passes,
        'continuation_resolution_reason': continuation_resolution_reason,
        'continuation_progress_state': continuation_progress_state,
        'pass_details': pass_details,
        'status_transitions': status_transitions,
        'resource_metrics': resource_metrics,
    }
    if message_id is not None:
        payload['message_id'] = message_id
    return payload


__all__ = [
    'build_display_blocks',
    'build_done_payload',
]
