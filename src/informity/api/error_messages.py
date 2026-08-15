# ==============================================================================
# Informity AI — Client-safe error messages
# ==============================================================================

"""Module for api error messages."""

GENERIC_RUNTIME_ERROR_MESSAGE = (
    "Error: Something went wrong while processing your request. Please try again."
)
LOW_MEMORY_RUNTIME_ERROR_MESSAGE = "insufficient memory available for this model"


def to_client_error_message(exc: Exception | BaseException) -> str:
    # Keep client messages generic; full details stay in server logs.
    """To client error message."""
    detail = str(getattr(exc, "detail", "") or "").strip()
    if detail == LOW_MEMORY_RUNTIME_ERROR_MESSAGE:
        return detail
    return GENERIC_RUNTIME_ERROR_MESSAGE
