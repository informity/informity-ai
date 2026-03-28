from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import structlog

from informity.config import DirNames, settings
from informity.llm.types import QueryType

log = structlog.get_logger(__name__)

_CACHE_TTL_SECONDS = 600.0
_MAX_DB_ROWS = 500
_MAX_TRACE_FILES = 300
_SECONDS_PER_DAY = 86400
_MIN_TIMEOUT_DIVISOR_SECONDS = 1
_DEFAULT_ROLLOUT_STAGE = 'default_on'
_ROLLOUT_STAGE_OPTIONS = {'dev', 'power_users', 'default_on'}
_ROLLOUT_BUCKET_MODULUS = 100
_ROLLOUT_POWER_USERS_BUCKET_LIMIT = 35
_DEFAULT_TUNING_MIN_SAMPLES = 20
_DEFAULT_TUNING_LOOKBACK_DAYS = 14
_MIN_DYNAMIC_COMPLETION_SAMPLES = 5
_DEFAULT_SOFT_TOP_K = 0.60
_DEFAULT_SOFT_REASONING = 0.75
_DEFAULT_SOFT_OUTPUT = 0.85
_DEFAULT_SOFT_COVERAGE = 0.95
_DEFAULT_HARD_PRE_GENERATION = 1.10
_DEFAULT_STREAM_SOFT_LIMIT = 0.90
_DEFAULT_FIRST_TOKEN_LATE = 0.55
_P95_PERCENTILE = 0.95

_COVERAGE_LOW_SAMPLE_SOFT_TOP_K = 0.56
_COVERAGE_LOW_SAMPLE_SOFT_REASONING = 0.70
_COVERAGE_LOW_SAMPLE_SOFT_OUTPUT = 0.80
_COVERAGE_LOW_SAMPLE_SOFT_COVERAGE = 0.90
_COVERAGE_LOW_SAMPLE_HARD_PRE_GENERATION = 1.05
_COVERAGE_LOW_SAMPLE_STREAM_SOFT_LIMIT = 0.86
_COVERAGE_LOW_SAMPLE_FIRST_TOKEN_LATE = 0.50

_COMPLETION_RATIO_MIN = 0.30
_COMPLETION_RATIO_MAX = 0.98
_DYNAMIC_SOFT_TOP_K_MIN = 0.50
_DYNAMIC_SOFT_TOP_K_MAX = 0.72
_DYNAMIC_SOFT_TOP_K_MULTIPLIER = 0.80
_DYNAMIC_SOFT_REASONING_MIN_OFFSET = 0.10
_DYNAMIC_SOFT_REASONING_MAX = 0.88
_DYNAMIC_SOFT_REASONING_MULTIPLIER = 0.96
_DYNAMIC_SOFT_OUTPUT_MIN_OFFSET = 0.08
_DYNAMIC_SOFT_OUTPUT_MAX = 0.94
_DYNAMIC_SOFT_OUTPUT_MULTIPLIER = 1.08
_DYNAMIC_SOFT_COVERAGE_MIN_OFFSET = 0.06
_DYNAMIC_SOFT_COVERAGE_MAX = 0.98
_DYNAMIC_SOFT_COVERAGE_MULTIPLIER = 1.16
_DYNAMIC_HARD_PRE_GENERATION_MIN = 1.02
_DYNAMIC_HARD_PRE_GENERATION_MAX = 1.20
_DYNAMIC_HARD_PRE_GENERATION_OFFSET = 0.12
_DYNAMIC_STREAM_SOFT_LIMIT_MIN = 0.82
_DYNAMIC_STREAM_SOFT_LIMIT_MAX = 0.95
_DYNAMIC_STREAM_SOFT_LIMIT_MULTIPLIER = 0.92

_TIMEOUT_RATE_EMERGENCY_THRESHOLD = 0.12
_TIMEOUT_EMERGENCY_SOFT_TOP_K_DELTA = 0.04
_TIMEOUT_EMERGENCY_SOFT_REASONING_DELTA = 0.03
_TIMEOUT_EMERGENCY_SOFT_REASONING_MIN_OFFSET = 0.08
_TIMEOUT_EMERGENCY_SOFT_OUTPUT_DELTA = 0.02
_TIMEOUT_EMERGENCY_SOFT_OUTPUT_MIN_OFFSET = 0.06

_FIRST_TOKEN_MS_PER_SECOND = 1000.0
_FIRST_TOKEN_LATE_RATIO_MULTIPLIER = 1.10
_FIRST_TOKEN_LATE_MIN = 0.35
_FIRST_TOKEN_LATE_MAX = 0.70

_DEEP_ANALYSIS_SOFT_TOP_K_CAP = 0.88
_DEEP_ANALYSIS_SOFT_TOP_K_DELTA = 0.16
_DEEP_ANALYSIS_SOFT_REASONING_CAP = 0.92
_DEEP_ANALYSIS_SOFT_REASONING_DELTA = 0.14
_DEEP_ANALYSIS_SOFT_OUTPUT_CAP = 0.96
_DEEP_ANALYSIS_SOFT_OUTPUT_DELTA = 0.10
_DEEP_ANALYSIS_SOFT_COVERAGE_CAP = 1.02
_DEEP_ANALYSIS_SOFT_COVERAGE_DELTA = 0.10
_DEEP_ANALYSIS_HARD_PRE_GENERATION_CAP = 1.30
_DEEP_ANALYSIS_HARD_PRE_GENERATION_DELTA = 0.15
_DEEP_ANALYSIS_STREAM_SOFT_LIMIT_CAP = 0.98
_DEEP_ANALYSIS_STREAM_SOFT_LIMIT_DELTA = 0.06
_DEEP_ANALYSIS_FIRST_TOKEN_LATE_CAP = 0.80
_DEEP_ANALYSIS_FIRST_TOKEN_LATE_DELTA = 0.12


@dataclass(frozen=True)
class FitToBudgetPolicy:
    enabled: bool
    rollout_stage: str
    sample_count: int
    timeout_rate: float
    completion_p95_seconds: float | None
    first_token_p95_ms: float | None
    soft_top_k_threshold: float
    soft_reasoning_threshold: float
    soft_output_cap_threshold: float
    soft_coverage_to_focused_threshold: float
    hard_pre_generation_threshold: float
    stream_soft_limit_ratio: float
    first_token_late_ratio: float


@dataclass
class _PolicyCache:
    expires_at: float
    policy: FitToBudgetPolicy


_policy_cache: dict[str, _PolicyCache] = {}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * pct))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _resolve_rollout_enabled() -> tuple[str, bool]:
    stage = str(getattr(settings, 'fit_to_budget_rollout_stage', _DEFAULT_ROLLOUT_STAGE)).strip().lower()
    if stage not in _ROLLOUT_STAGE_OPTIONS:
        stage = _DEFAULT_ROLLOUT_STAGE
    if not bool(getattr(settings, 'fit_to_budget_enabled', True)):
        return stage, False
    if stage == _DEFAULT_ROLLOUT_STAGE:
        return stage, True
    if stage == 'dev':
        return stage, bool(settings.dev_reload)
    hint = f'{settings.app_data_dir}|{settings.log_level}|{settings.diagnostics_profile}'
    bucket = int(hashlib.sha256(hint.encode('utf-8')).hexdigest()[:8], 16) % _ROLLOUT_BUCKET_MODULUS
    return stage, bucket < _ROLLOUT_POWER_USERS_BUCKET_LIMIT


def _resolve_tuning_min_samples() -> int:
    return int(getattr(settings, 'fit_to_budget_tuning_min_samples', _DEFAULT_TUNING_MIN_SAMPLES))


def _extract_llm_step(trace_data: dict) -> dict | None:
    steps = trace_data.get('steps')
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and step.get('name') == 'llm' and isinstance(step.get('data'), dict):
            return step['data']
    return None


def _extract_intent_step(trace_data: dict) -> dict | None:
    steps = trace_data.get('steps')
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and step.get('name') == 'intent' and isinstance(step.get('data'), dict):
            return step['data']
    return None


def _collect_first_token_samples_sync(query_type: QueryType, lookback_days: int) -> list[float]:
    cutoff = time.time() - (max(1, lookback_days) * _SECONDS_PER_DAY)
    base_dirs: list[Path] = [
        settings.app_data_dir / DirNames.CHAT_LOGS,
        settings.diagnostics_dir / DirNames.RUNS,
    ]
    candidates: list[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for path in base_dir.rglob('*.json'):
            try:
                if path.stat().st_mtime >= cutoff:
                    candidates.append(path)
            except OSError:
                continue
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    samples: list[float] = []
    for path in candidates[:_MAX_TRACE_FILES]:
        try:
            trace_data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        intent_data = _extract_intent_step(trace_data)
        if not intent_data or str(intent_data.get('query_type', '')).strip().lower() != query_type:
            continue
        llm_data = _extract_llm_step(trace_data)
        if not llm_data:
            continue
        first_token_ms = llm_data.get('first_token_ms')
        if isinstance(first_token_ms, (int, float)) and first_token_ms > 0:
            samples.append(float(first_token_ms))
    return samples


async def _load_runtime_stats(
    db: aiosqlite.Connection,
    query_type: QueryType,
    lookback_days: int,
) -> tuple[int, float, float | None, float | None]:
    cursor = await db.execute(
        """
        SELECT generation_seconds, timeout_occurred
        FROM response_diagnostics_metrics
        WHERE query_type = ?
          AND generation_seconds IS NOT NULL
          AND created_at >= datetime('now', '-' || ? || ' days')
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (query_type, max(1, lookback_days), _MAX_DB_ROWS),
    )
    rows = await cursor.fetchall()
    completion_values = [float(row['generation_seconds']) for row in rows if row['generation_seconds'] is not None]
    timeout_count = sum(1 for row in rows if bool(row['timeout_occurred']))
    sample_count = len(completion_values)
    timeout_rate = (float(timeout_count) / float(sample_count)) if sample_count > 0 else 0.0
    completion_p95 = _percentile(completion_values, _P95_PERCENTILE)
    first_token_samples = await asyncio.to_thread(_collect_first_token_samples_sync, query_type, lookback_days)
    first_token_p95 = _percentile(first_token_samples, _P95_PERCENTILE)
    return sample_count, timeout_rate, completion_p95, first_token_p95


def _derive_thresholds(
    query_type: QueryType,
    timeout_seconds: int,
    sample_count: int,
    timeout_rate: float,
    completion_p95_seconds: float | None,
    first_token_p95_ms: float | None,
) -> tuple[float, float, float, float, float, float, float]:
    soft_top_k = _DEFAULT_SOFT_TOP_K
    soft_reasoning = _DEFAULT_SOFT_REASONING
    soft_output = _DEFAULT_SOFT_OUTPUT
    soft_coverage = _DEFAULT_SOFT_COVERAGE
    hard_pre_generation = _DEFAULT_HARD_PRE_GENERATION
    stream_soft_limit = _DEFAULT_STREAM_SOFT_LIMIT
    first_token_late = _DEFAULT_FIRST_TOKEN_LATE

    min_samples = _resolve_tuning_min_samples()
    if query_type == QueryType.COVERAGE and sample_count < min_samples:
        soft_top_k = _COVERAGE_LOW_SAMPLE_SOFT_TOP_K
        soft_reasoning = _COVERAGE_LOW_SAMPLE_SOFT_REASONING
        soft_output = _COVERAGE_LOW_SAMPLE_SOFT_OUTPUT
        soft_coverage = _COVERAGE_LOW_SAMPLE_SOFT_COVERAGE
        hard_pre_generation = _COVERAGE_LOW_SAMPLE_HARD_PRE_GENERATION
        stream_soft_limit = _COVERAGE_LOW_SAMPLE_STREAM_SOFT_LIMIT
        first_token_late = _COVERAGE_LOW_SAMPLE_FIRST_TOKEN_LATE

    if completion_p95_seconds is not None and sample_count >= max(_MIN_DYNAMIC_COMPLETION_SAMPLES, min_samples):
        completion_ratio = completion_p95_seconds / float(max(timeout_seconds, _MIN_TIMEOUT_DIVISOR_SECONDS))
        completion_ratio = max(_COMPLETION_RATIO_MIN, min(completion_ratio, _COMPLETION_RATIO_MAX))
        soft_top_k = max(_DYNAMIC_SOFT_TOP_K_MIN, min(_DYNAMIC_SOFT_TOP_K_MAX, completion_ratio * _DYNAMIC_SOFT_TOP_K_MULTIPLIER))
        soft_reasoning = max(
            soft_top_k + _DYNAMIC_SOFT_REASONING_MIN_OFFSET,
            min(_DYNAMIC_SOFT_REASONING_MAX, completion_ratio * _DYNAMIC_SOFT_REASONING_MULTIPLIER),
        )
        soft_output = max(
            soft_reasoning + _DYNAMIC_SOFT_OUTPUT_MIN_OFFSET,
            min(_DYNAMIC_SOFT_OUTPUT_MAX, completion_ratio * _DYNAMIC_SOFT_OUTPUT_MULTIPLIER),
        )
        soft_coverage = max(
            soft_output + _DYNAMIC_SOFT_COVERAGE_MIN_OFFSET,
            min(_DYNAMIC_SOFT_COVERAGE_MAX, completion_ratio * _DYNAMIC_SOFT_COVERAGE_MULTIPLIER),
        )
        hard_pre_generation = max(
            _DYNAMIC_HARD_PRE_GENERATION_MIN,
            min(_DYNAMIC_HARD_PRE_GENERATION_MAX, soft_coverage + _DYNAMIC_HARD_PRE_GENERATION_OFFSET),
        )
        stream_soft_limit = max(
            _DYNAMIC_STREAM_SOFT_LIMIT_MIN,
            min(_DYNAMIC_STREAM_SOFT_LIMIT_MAX, completion_ratio * _DYNAMIC_STREAM_SOFT_LIMIT_MULTIPLIER),
        )
    if timeout_rate >= _TIMEOUT_RATE_EMERGENCY_THRESHOLD:
        soft_top_k = max(_DYNAMIC_SOFT_TOP_K_MIN, soft_top_k - _TIMEOUT_EMERGENCY_SOFT_TOP_K_DELTA)
        soft_reasoning = max(
            soft_top_k + _TIMEOUT_EMERGENCY_SOFT_REASONING_MIN_OFFSET,
            soft_reasoning - _TIMEOUT_EMERGENCY_SOFT_REASONING_DELTA,
        )
        soft_output = max(
            soft_reasoning + _TIMEOUT_EMERGENCY_SOFT_OUTPUT_MIN_OFFSET,
            soft_output - _TIMEOUT_EMERGENCY_SOFT_OUTPUT_DELTA,
        )
    if first_token_p95_ms is not None:
        ratio = (first_token_p95_ms / _FIRST_TOKEN_MS_PER_SECOND) / float(max(timeout_seconds, _MIN_TIMEOUT_DIVISOR_SECONDS))
        first_token_late = max(_FIRST_TOKEN_LATE_MIN, min(_FIRST_TOKEN_LATE_MAX, ratio * _FIRST_TOKEN_LATE_RATIO_MULTIPLIER))
    return (
        soft_top_k,
        soft_reasoning,
        soft_output,
        soft_coverage,
        hard_pre_generation,
        stream_soft_limit,
        first_token_late,
    )


async def resolve_fit_to_budget_policy(
    db: aiosqlite.Connection,
    query_type: QueryType,
    timeout_seconds: int,
) -> FitToBudgetPolicy:
    normalized_mode = 'single'
    stage, enabled = _resolve_rollout_enabled()
    cache_key = f'{query_type}:{normalized_mode}:{stage}:{enabled}:{timeout_seconds}'
    now = time.time()
    cached = _policy_cache.get(cache_key)
    if cached is not None and cached.expires_at > now:
        return cached.policy

    lookback_days = int(getattr(settings, 'fit_to_budget_tuning_days', _DEFAULT_TUNING_LOOKBACK_DAYS))
    sample_count, timeout_rate, completion_p95_seconds, first_token_p95_ms = await _load_runtime_stats(
        db,
        query_type,
        lookback_days,
    )
    (
        soft_top_k,
        soft_reasoning,
        soft_output,
        soft_coverage,
        hard_pre_generation,
        stream_soft_limit,
        first_token_late,
    ) = _derive_thresholds(
        query_type=query_type,
        timeout_seconds=timeout_seconds,
        sample_count=sample_count,
        timeout_rate=timeout_rate,
        completion_p95_seconds=completion_p95_seconds,
        first_token_p95_ms=first_token_p95_ms,
    )
    policy = FitToBudgetPolicy(
        enabled=enabled,
        rollout_stage=stage,
        sample_count=sample_count,
        timeout_rate=round(timeout_rate, 4),
        completion_p95_seconds=completion_p95_seconds,
        first_token_p95_ms=first_token_p95_ms,
        soft_top_k_threshold=soft_top_k,
        soft_reasoning_threshold=soft_reasoning,
        soft_output_cap_threshold=soft_output,
        soft_coverage_to_focused_threshold=soft_coverage,
        hard_pre_generation_threshold=hard_pre_generation,
        stream_soft_limit_ratio=stream_soft_limit,
        first_token_late_ratio=first_token_late,
    )
    policy = FitToBudgetPolicy(
        enabled=policy.enabled,
        rollout_stage=policy.rollout_stage,
        sample_count=policy.sample_count,
        timeout_rate=policy.timeout_rate,
        completion_p95_seconds=policy.completion_p95_seconds,
        first_token_p95_ms=policy.first_token_p95_ms,
        soft_top_k_threshold=min(_DEEP_ANALYSIS_SOFT_TOP_K_CAP, policy.soft_top_k_threshold + _DEEP_ANALYSIS_SOFT_TOP_K_DELTA),
        soft_reasoning_threshold=min(
            _DEEP_ANALYSIS_SOFT_REASONING_CAP,
            policy.soft_reasoning_threshold + _DEEP_ANALYSIS_SOFT_REASONING_DELTA,
        ),
        soft_output_cap_threshold=min(
            _DEEP_ANALYSIS_SOFT_OUTPUT_CAP,
            policy.soft_output_cap_threshold + _DEEP_ANALYSIS_SOFT_OUTPUT_DELTA,
        ),
        soft_coverage_to_focused_threshold=min(
            _DEEP_ANALYSIS_SOFT_COVERAGE_CAP,
            policy.soft_coverage_to_focused_threshold + _DEEP_ANALYSIS_SOFT_COVERAGE_DELTA,
        ),
        hard_pre_generation_threshold=min(
            _DEEP_ANALYSIS_HARD_PRE_GENERATION_CAP,
            policy.hard_pre_generation_threshold + _DEEP_ANALYSIS_HARD_PRE_GENERATION_DELTA,
        ),
        stream_soft_limit_ratio=min(
            _DEEP_ANALYSIS_STREAM_SOFT_LIMIT_CAP,
            policy.stream_soft_limit_ratio + _DEEP_ANALYSIS_STREAM_SOFT_LIMIT_DELTA,
        ),
        first_token_late_ratio=min(
            _DEEP_ANALYSIS_FIRST_TOKEN_LATE_CAP,
            policy.first_token_late_ratio + _DEEP_ANALYSIS_FIRST_TOKEN_LATE_DELTA,
        ),
    )
    _policy_cache[cache_key] = _PolicyCache(
        expires_at=now + _CACHE_TTL_SECONDS,
        policy=policy,
    )
    log.debug(
        'fit_to_budget_policy_resolved',
        query_type=query_type,
        policy_variant=normalized_mode,
        rollout_stage=stage,
        enabled=enabled,
        sample_count=sample_count,
        timeout_rate=policy.timeout_rate,
        completion_p95_seconds=completion_p95_seconds,
        first_token_p95_ms=first_token_p95_ms,
    )
    return policy
