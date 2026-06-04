# ==============================================================================
# Informity AI — Chat Orchestrator (Shell)
# Thin helper for /api/chat runtime decomposition.
# ==============================================================================


def prepare_chat_request(
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
