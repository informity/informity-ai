from enum import StrEnum


class SetupState(StrEnum):
    READY = 'ready'
    REQUIRED = 'setup_required'
    IN_PROGRESS = 'setup_in_progress'
    FAILED = 'setup_failed'


__all__ = ['SetupState']
