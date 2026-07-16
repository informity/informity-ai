"""Module for api setup state."""

from enum import StrEnum


class SetupState(StrEnum):
    """Class docstring."""
    READY = "ready"
    REQUIRED = "setup_required"
    IN_PROGRESS = "setup_in_progress"
    FAILED = "setup_failed"


__all__ = ["SetupState"]
