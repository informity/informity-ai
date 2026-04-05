from __future__ import annotations

from informity.llm.metrics_payload import build_metrics_payload


def test_build_metrics_payload_keeps_required_fields() -> None:
    payload = build_metrics_payload(query_type='simple', raw_chunks_count=0)
    assert payload == {'query_type': 'simple', 'raw_chunks_count': 0}


def test_build_metrics_payload_omits_none_optional_fields() -> None:
    payload = build_metrics_payload(
        query_type='coverage',
        raw_chunks_count=5,
        stream_duration_ms=None,
        answerability_passed=True,
    )
    assert payload['query_type'] == 'coverage'
    assert payload['raw_chunks_count'] == 5
    assert payload['answerability_passed'] is True
    assert 'stream_duration_ms' not in payload
