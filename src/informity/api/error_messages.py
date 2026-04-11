# ==============================================================================
# Informity AI — Client-safe error messages
# ==============================================================================

GENERIC_RUNTIME_ERROR_MESSAGE = 'Error: Something went wrong while processing your request. Please try again.'


def to_client_error_message(exc: Exception | BaseException) -> str:
    # Keep client messages generic; full details stay in server logs.
    _ = exc
    return GENERIC_RUNTIME_ERROR_MESSAGE
