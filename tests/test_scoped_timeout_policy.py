from informity.timeout_policy import (
    POLICY_SCOPE_FILESYSTEM_FILE,
    ScopedTimeoutPolicy,
    TimeoutPolicyLevel,
    default_scoped_timeout_policy,
    normalize_scope_key,
    resolve_timeout_seconds,
)


def test_resolve_timeout_uses_scope_override_when_present() -> None:
    policy = ScopedTimeoutPolicy(
        default=TimeoutPolicyLevel(base_seconds=100, seconds_per_mb=10.0, min_seconds=20, max_seconds=400),
        overrides={
            'filesystem:file': TimeoutPolicyLevel(base_seconds=50, seconds_per_mb=5.0, min_seconds=10, max_seconds=300),
        },
    )
    # 10 MiB -> 50 + 50 = 100
    resolved = resolve_timeout_seconds(policy, scope_key='filesystem:file', size_bytes=10 * 1024 * 1024)
    assert resolved == 100


def test_resolve_timeout_falls_back_to_default_for_unknown_scope() -> None:
    policy = ScopedTimeoutPolicy(
        default=TimeoutPolicyLevel(base_seconds=100, seconds_per_mb=10.0, min_seconds=20, max_seconds=400),
        overrides={},
    )
    # 2 MiB -> 100 + 20 = 120
    resolved = resolve_timeout_seconds(policy, scope_key='mail.apple:mail', size_bytes=2 * 1024 * 1024)
    assert resolved == 120


def test_default_scoped_timeout_policy_is_valid() -> None:
    policy = default_scoped_timeout_policy()
    resolved = resolve_timeout_seconds(policy, scope_key='filesystem:file', size_bytes=0)
    assert resolved >= policy.default.min_seconds
    assert resolved <= policy.default.max_seconds


def test_normalize_scope_key_normalizes_case_and_whitespace() -> None:
    assert normalize_scope_key('  Filesystem ', ' File ') == POLICY_SCOPE_FILESYSTEM_FILE
    assert normalize_scope_key(None, None) == ':'


def test_resolve_timeout_accepts_mapping_policy_input() -> None:
    policy_mapping = {
        'default': {
            'base_seconds': 100,
            'seconds_per_mb': 10.0,
            'min_seconds': 20,
            'max_seconds': 400,
        },
        'overrides': {
            POLICY_SCOPE_FILESYSTEM_FILE: {
                'base_seconds': 50,
                'seconds_per_mb': 5.0,
                'min_seconds': 10,
                'max_seconds': 300,
            },
        },
    }
    resolved = resolve_timeout_seconds(
        policy_mapping,
        scope_key=POLICY_SCOPE_FILESYSTEM_FILE,
        size_bytes=10 * 1024 * 1024,
    )
    assert resolved == 100
