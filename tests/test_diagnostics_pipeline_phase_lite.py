from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('tools.diagnostics', reason='diagnostics tools package is not available in this checkout')
pytestmark = pytest.mark.diagnostics

from tools.diagnostics import pipeline


def test_select_pack_queries_filters_by_capability() -> None:
    queries = [
        {'question': 'route me', 'type': 'focused', 'expected_intent': 'focused'},
        {'question': 'retrieve me', 'type': 'coverage', 'min_sources_count': 2, 'expected_intent': 'coverage'},
        {'question': 'shape me', 'type': 'coverage', 'output_shape': {'required_headings': ['A']}, 'expected_intent': 'coverage'},
        {'question': 'long timeout style prompt one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen', 'type': 'coverage', 'expected_intent': 'coverage'},
    ]

    routing = pipeline._select_pack_queries(queries, 'routing')
    retrieval = pipeline._select_pack_queries(queries, 'retrieval')
    output_contract = pipeline._select_pack_queries(queries, 'output_contract')
    timeout = pipeline._select_pack_queries(queries, 'timeout')

    assert len(routing) == 2
    assert len(retrieval) == 4
    assert len(output_contract) == 1
    assert len(timeout) == 2


def test_build_run_diff_and_promote_recurring_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, 'settings', SimpleNamespace(diagnostics_dir=tmp_path))

    # Isolate bank/candidate files to tmp sandbox.
    regression_bank = tmp_path / 'regression_v1.json'
    regression_candidates = tmp_path / 'regression_candidates_v1.json'
    monkeypatch.setattr(pipeline, '_REGRESSION_BANK_FILE', regression_bank)
    monkeypatch.setattr(pipeline, '_REGRESSION_CANDIDATES_FILE', regression_candidates)

    regression_bank.write_text(
        '{"version":"v1","queries":[{"question":"already in bank"}]}',
        encoding='utf-8',
    )

    current_dir = tmp_path / 'runs' / 'current' / 'results'
    baseline_dir = tmp_path / 'runs' / 'baseline' / 'results'
    current_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir.joinpath('run.json').write_text(
        """
{
  "run_id": "baseline",
  "total_queries": 3,
  "results": [
    {"question":"Q1", "success": true, "regression_passed": false},
    {"question":"Q2", "success": true, "regression_passed": true},
    {"question":"already in bank", "success": true, "regression_passed": false}
  ]
}
""".strip(),
        encoding='utf-8',
    )
    current_dir.joinpath('run.json').write_text(
        """
{
  "run_id": "current",
  "total_queries": 3,
  "results": [
    {"question":"Q1", "success": true, "regression_passed": false},
    {"question":"Q2", "success": true, "regression_passed": false},
    {"question":"already in bank", "success": true, "regression_passed": false}
  ]
}
""".strip(),
        encoding='utf-8',
    )

    diff_payload = pipeline._build_run_diff('current', 'baseline')
    assert isinstance(diff_payload, dict)
    assert len(diff_payload['new_failures']) == 1
    assert len(diff_payload['resolved_failures']) == 0
    assert len(diff_payload['recurring_failures']) == 2

    promotion = pipeline._promote_recurring_failures_lite(
        current_run_id='current',
        baseline_run_id='baseline',
        diff_payload=diff_payload,
    )
    assert promotion['status'] == 'ok'
    # Q1 should be promoted; "already in bank" should be skipped.
    assert promotion['added'] == 1

    payload = pipeline._load_regression_candidates()
    questions = [entry.get('question') for entry in payload.get('queries', [])]
    assert 'Q1' in questions
    assert 'already in bank' not in questions


def test_write_fallback_pack_queries_routing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, 'settings', SimpleNamespace(diagnostics_dir=tmp_path))
    ok, reason = pipeline._write_fallback_pack_queries(
        'fallback-run',
        query_pack='routing',
        strategy='progressive',
        num_queries=12,
        seed=None,
    )
    assert ok is True
    assert reason.startswith('fallback_generated_queries=')
    payload = pipeline._load_queries_payload('fallback-run')
    assert isinstance(payload, dict)
    assert payload.get('fallback_mode') == 'routing_no_index'
    assert payload.get('query_pack') == 'routing'
    assert isinstance(payload.get('queries'), list)
    assert len(payload.get('queries')) == 12
