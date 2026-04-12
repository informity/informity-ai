from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator

POLICY_SCOPE_FILESYSTEM_FILE = 'filesystem:file'
POLICY_SCOPE_MAIL_APPLE = 'mail.apple:mail'
POLICY_SCOPE_MAIL_OUTLOOK = 'mail.outlook:mail'
POLICY_MIN_SECONDS = 1
POLICY_MAX_SECONDS = 600
_MEBIBYTE = 1024 * 1024


class TimeoutPolicyLevel(BaseModel):
    base_seconds: int = 90
    seconds_per_mb: float = 22.0
    min_seconds: int = 30
    max_seconds: int = 600

    @model_validator(mode='after')
    def _validate_bounds(self) -> TimeoutPolicyLevel:
        if self.base_seconds < POLICY_MIN_SECONDS:
            raise ValueError(f'base_seconds must be >= {POLICY_MIN_SECONDS}')
        if self.seconds_per_mb < 0:
            raise ValueError('seconds_per_mb must be >= 0')
        if self.min_seconds < POLICY_MIN_SECONDS:
            raise ValueError(f'min_seconds must be >= {POLICY_MIN_SECONDS}')
        if self.max_seconds > POLICY_MAX_SECONDS:
            raise ValueError(f'max_seconds must be <= {POLICY_MAX_SECONDS}')
        if self.min_seconds > self.max_seconds:
            raise ValueError('min_seconds must be <= max_seconds')
        return self


class ScopedTimeoutPolicy(BaseModel):
    default: TimeoutPolicyLevel
    overrides: dict[str, TimeoutPolicyLevel] = Field(default_factory=dict)


def default_scoped_timeout_policy() -> ScopedTimeoutPolicy:
    default_level = TimeoutPolicyLevel(
        base_seconds=90,
        seconds_per_mb=22.0,
        min_seconds=30,
        max_seconds=600,
    )
    return ScopedTimeoutPolicy(
        default=default_level,
        overrides={
            POLICY_SCOPE_FILESYSTEM_FILE: default_level,
            POLICY_SCOPE_MAIL_APPLE: TimeoutPolicyLevel(
                base_seconds=75,
                seconds_per_mb=18.0,
                min_seconds=30,
                max_seconds=540,
            ),
            POLICY_SCOPE_MAIL_OUTLOOK: TimeoutPolicyLevel(
                base_seconds=75,
                seconds_per_mb=18.0,
                min_seconds=30,
                max_seconds=540,
            ),
        },
    )


def normalize_scope_key(source_provider: str, entity_type: str) -> str:
    provider = str(source_provider or '').strip().lower()
    entity = str(entity_type or '').strip().lower()
    return f'{provider}:{entity}'


def _clamp_timeout_seconds(timeout_seconds: float, *, minimum: int, maximum: int) -> int:
    bounded = min(maximum, max(minimum, timeout_seconds))
    return int(round(bounded))


def resolve_timeout_seconds(
    policy: ScopedTimeoutPolicy | Mapping[str, Any],
    *,
    scope_key: str,
    size_bytes: int,
) -> int:
    scoped = policy if isinstance(policy, ScopedTimeoutPolicy) else ScopedTimeoutPolicy.model_validate(policy)
    level = scoped.overrides.get(scope_key, scoped.default)
    size_mb = max(0.0, float(size_bytes) / float(_MEBIBYTE))
    raw_seconds = float(level.base_seconds) + (float(level.seconds_per_mb) * size_mb)
    return _clamp_timeout_seconds(raw_seconds, minimum=level.min_seconds, maximum=level.max_seconds)
