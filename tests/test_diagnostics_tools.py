from __future__ import annotations

import json
from random import Random

from tools.diagnostics import analyze, evaluate, generate_queries, pipeline, run_control

from informity.chat_trace import _ChatTraceWriter


def test_pipeline_writes_manifest_without_clobbering_run_json(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-1'
    results_dir = tmp_path / 'runs' / run_id / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json = results_dir / 'run.json'
    run_payload = {'results': [{'success': True, 'question': 'q1'}]}
    run_json.write_text(json.dumps(run_payload), encoding='utf-8')

    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(pipeline, 'run_generate_queries', lambda *_args, **_kwargs: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (True, ''),
    )
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    success = pipeline.run_pipeline(run_id=run_id, quiet=True, seed=42)

    assert success
    persisted = json.loads(run_json.read_text(encoding='utf-8'))
    assert persisted.get('results') == run_payload['results']

    manifest_file = results_dir / 'pipeline_manifest.json'
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert manifest['run_id'] == run_id
    assert manifest['seed'] == 42


def test_build_metrics_from_evaluation_results() -> None:
    results = [
        {
            'success': True,
            'query_type': 'focused',
            'model_filename': 'model.gguf',
            'issues': ['timeout'],
            'generation_seconds': 1.25,
            'answer_length': 220,
        },
        {'success': False, 'query_type': 'coverage', 'issues': ['empty_answer']},
    ]
    metrics = analyze.build_metrics_from_evaluation_results(results)

    assert len(metrics) == 1
    assert metrics[0]['query_type'] == 'focused'
    assert metrics[0]['model_filename'] == 'model.gguf'
    assert metrics[0]['detected_issues'] == ['timeout']
    assert metrics[0]['generation_seconds'] == 1.25
    assert metrics[0]['answer_length'] == 220


def test_merge_quality_failures_into_issue_frequency() -> None:
    aggregated = {
        'issue_frequency': {'timeout': 2},
        'by_issue_type': {'timeout': 2},
        'regression_failed': 1,
        'expectation_failures': 3,
    }
    merged = analyze.merge_quality_failures_into_issue_frequency(aggregated)

    assert merged['issue_frequency']['timeout'] == 2
    assert merged['issue_frequency']['regression_assertion_failures'] == 1
    assert merged['issue_frequency']['expectation_assertion_failures'] == 3
    assert merged['by_issue_type']['regression_assertion_failures'] == 1
    assert merged['by_issue_type']['expectation_assertion_failures'] == 3


def test_pipeline_uses_custom_queries_file(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-custom-queries'
    custom_queries_file = tmp_path / 'custom_queries.json'
    custom_payload = {
        'run_id': run_id,
        'queries': [
            {'question': 'What files are in the index?', 'expected_intent': 'metadata'},
        ],
    }
    custom_queries_file.write_text(json.dumps(custom_payload), encoding='utf-8')

    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (
            bool(queries_file and queries_file.exists()),
            '',
        ),
    )
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    def _should_not_run_generate(*_args, **_kwargs):
        raise AssertionError('run_generate_queries should not be called when --queries-file is used')

    monkeypatch.setattr(pipeline, 'run_generate_queries', _should_not_run_generate)

    success = pipeline.run_pipeline(
        run_id=run_id,
        queries_file=custom_queries_file,
        quiet=True,
    )
    assert success
    run_queries_file = tmp_path / 'runs' / run_id / 'queries' / 'queries.json'
    assert run_queries_file.exists()
    assert json.loads(run_queries_file.read_text(encoding='utf-8')) == custom_payload


def test_pipeline_marks_completed_with_failures_when_evaluate_finishes_with_quality_failures(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-quality-fail'
    run_queries_file = tmp_path / 'runs' / run_id / 'queries' / 'queries.json'
    run_queries_file.parent.mkdir(parents=True, exist_ok=True)
    run_queries_file.write_text(
        json.dumps({'run_id': run_id, 'queries': [{'question': 'q1'}]}),
        encoding='utf-8',
    )
    results_dir = tmp_path / 'runs' / run_id / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json = results_dir / 'run.json'
    run_json.write_text(
        json.dumps(
            {
                'run_id': run_id,
                'total_queries': 1,
                'completed_queries': 1,
                'successful': 0,
                'failed': 1,
                'regression_failed': 0,
                'results': [{'success': False}],
            }
        ),
        encoding='utf-8',
    )

    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_generate_queries',
        lambda _run_id, strategy, num_queries, seed=None, timeout_seconds=None: (True, ''),
    )
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (False, 'nonzero_exit_1'),
    )
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    success = pipeline.run_pipeline(run_id=run_id, quiet=True, seed=42)
    assert success is False

    manifest_file = results_dir / 'pipeline_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed_with_failures'


def test_pipeline_preflight_rejects_impossible_run_timeout(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-preflight-timeout'
    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(pipeline, 'run_generate_queries', lambda *_args, **_kwargs: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (True, ''),
    )
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    success = pipeline.run_pipeline(
        run_id=run_id,
        quiet=True,
        num_queries=20,
        query_timeout_seconds=120,
        run_timeout_seconds=30,
    )
    assert success is False

    manifest_file = tmp_path / 'runs' / run_id / 'results' / 'pipeline_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert manifest['status'] == 'failed'
    assert str(manifest.get('reason', '')).startswith('pipeline_timeout_preflight_failed:')


def test_pipeline_reconciles_orphaned_running_runs_before_start(tmp_path, monkeypatch) -> None:
    stale_run_id = 'run-stale-orphan'
    stale_results_dir = tmp_path / 'runs' / stale_run_id / 'results'
    stale_results_dir.mkdir(parents=True, exist_ok=True)
    (stale_results_dir / 'run.json').write_text(
        json.dumps(
            {
                'run_id': stale_run_id,
                'status': 'running',
                'total_queries': 3,
                'completed_queries': 1,
                'results': [{'success': True}],
            }
        ),
        encoding='utf-8',
    )

    run_id = 'run-test-reconcile'
    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(pipeline, 'run_generate_queries', lambda *_args, **_kwargs: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (True, ''),
    )
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    success = pipeline.run_pipeline(run_id=run_id, quiet=True)
    assert success

    stale_payload = json.loads((stale_results_dir / 'run.json').read_text(encoding='utf-8'))
    assert stale_payload.get('status') == 'aborted'
    assert stale_payload.get('aborted_reason') == 'orphaned_running_state_reconciled'


def test_pipeline_marks_failed_and_clears_lock_on_unhandled_step_exception(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-unhandled-exception'
    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(pipeline, 'run_generate_queries', lambda *_args, **_kwargs: (True, ''))
    monkeypatch.setattr(
        pipeline,
        'run_evaluate',
        lambda _run_id, queries_file=None, query_timeout_seconds=None, llm_model_filename=None, timeout_seconds=None: (True, ''),
    )

    def _raise_analyze(_run_id, timeout_seconds=None):
        raise RuntimeError('boom')

    monkeypatch.setattr(pipeline, 'run_analyze', _raise_analyze)

    success = pipeline.run_pipeline(run_id=run_id, quiet=True, seed=42)
    assert success is False

    run_file = tmp_path / 'runs' / run_id / 'results' / 'run.json'
    manifest_file = tmp_path / 'runs' / run_id / 'results' / 'pipeline_manifest.json'
    run_payload = json.loads(run_file.read_text(encoding='utf-8'))
    manifest_payload = json.loads(manifest_file.read_text(encoding='utf-8'))

    assert run_payload.get('status') == 'failed'
    assert str(run_payload.get('aborted_reason', '')).startswith('pipeline_unhandled_exception:analyze:RuntimeError')
    assert manifest_payload.get('status') == 'failed'
    assert str(manifest_payload.get('reason', '')).startswith('pipeline_unhandled_exception:analyze:RuntimeError')
    assert run_control.read_active_lock(tmp_path) is None


def test_pipeline_auto_adjusts_evaluate_step_timeout_floor(tmp_path, monkeypatch) -> None:
    run_id = 'run-test-evaluate-timeout-floor'
    monkeypatch.setattr(pipeline.settings, 'diagnostics_dir', tmp_path)
    monkeypatch.setattr(pipeline, 'run_golden_set', lambda timeout_seconds=None: (True, ''))
    monkeypatch.setattr(pipeline, 'run_generate_queries', lambda *_args, **_kwargs: (True, ''))
    monkeypatch.setattr(pipeline, 'run_analyze', lambda _run_id, timeout_seconds=None: (True, ''))

    captured: dict[str, object] = {}

    def _run_evaluate(
        _run_id,
        queries_file=None,
        query_timeout_seconds=None,
        llm_model_filename=None,
        timeout_seconds=None,
    ):
        captured['timeout_seconds'] = timeout_seconds
        return True, ''

    monkeypatch.setattr(pipeline, 'run_evaluate', _run_evaluate)

    success = pipeline.run_pipeline(
        run_id=run_id,
        quiet=True,
        num_queries=4,
        query_timeout_seconds=90,
        evaluate_step_timeout_seconds=240,
    )
    assert success
    assert captured.get('timeout_seconds') == 420.0


def test_generate_balanced_queries_deterministic() -> None:
    years = [2023, 2024]
    categories = ['tax', 'banking']
    extensions = ['.pdf', '.csv']
    files = [
        {'filename': 'a.pdf', 'extension': '.pdf'},
        {'filename': 'b.pdf', 'extension': '.pdf'},
        {'filename': 'c.csv', 'extension': '.csv'},
    ]

    queries_a = generate_queries.generate_balanced_queries(
        years=years,
        categories=categories,
        extensions=extensions,
        files=files,
        num_queries=10,
        rng=Random(99),
    )
    queries_b = generate_queries.generate_balanced_queries(
        years=years,
        categories=categories,
        extensions=extensions,
        files=files,
        num_queries=10,
        rng=Random(99),
    )

    assert queries_a == queries_b


def test_generate_progressive_queries_tiered_and_deterministic() -> None:
    years = [2023, 2024]
    categories = ['tax', 'banking']
    extensions = ['.pdf', '.csv']
    files = [
        {'filename': 'a.pdf', 'extension': '.pdf'},
        {'filename': 'b.pdf', 'extension': '.pdf'},
        {'filename': 'c.csv', 'extension': '.csv'},
    ]

    queries_a = generate_queries.generate_progressive_queries(
        years=years,
        categories=categories,
        extensions=extensions,
        files=files,
        num_queries=12,
        rng=Random(7),
    )
    queries_b = generate_queries.generate_progressive_queries(
        years=years,
        categories=categories,
        extensions=extensions,
        files=files,
        num_queries=12,
        rng=Random(7),
    )

    assert queries_a == queries_b
    assert len(queries_a) == 12
    assert all(isinstance(query.get('difficulty_tier'), str) for query in queries_a)

    rank = {'tier1_foundational': 1, 'tier2_intermediate': 2, 'tier3_advanced': 3}
    tiers = [rank[str(query.get('difficulty_tier'))] for query in queries_a]
    assert tiers == sorted(tiers)
    assert any(query.get('difficulty_tier') == 'tier3_advanced' for query in queries_a)


def test_validate_query_expectations_enforces_required_table_columns() -> None:
    failures = evaluate._validate_query_expectations(
        query_item={
            'output_shape': {
                'required_table_columns': ['Line Item', 'Amount', 'Source Snippet'],
            },
        },
        sections={},
        answer=(
            "| Field | Value | Source Snippet |\n"
            "| --- | ---: | --- |\n"
            "| a | 1 | src |\n"
        ),
        sources_count=1,
        unsupported_claim_count=0,
        evidence_coverage_rate=1.0,
        not_found_count=0,
    )
    assert any(str(item).startswith('required_table_columns_mismatch:') for item in failures)


def test_validate_query_expectations_does_not_fail_strict_gates_when_disabled() -> None:
    failures = evaluate._validate_query_expectations(
        query_item={
            'max_unsupported_claim_count': 0,
            'output_shape': {
                'required_table_columns': ['Line Item', 'Amount', 'Source Snippet'],
            },
        },
        sections={'llm': {}},
        answer=(
            "| Field | Value | Source Snippet |\n"
            "| --- | ---: | --- |\n"
            "| a | 1 | src |\n"
        ),
        sources_count=1,
        unsupported_claim_count=2,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert any(str(item).startswith('required_table_columns_mismatch:') for item in failures)
    assert any(str(item).startswith('unsupported_claim_count_above_threshold:') for item in failures)


def test_evaluate_summary_includes_latency_resource_metrics_and_budget_alert_total() -> None:
    summary = evaluate._build_summary(
        run_id='run-metrics',
        total_queries=3,
        results=[
            {
                'success': True,
                'generation_seconds': 10.0,
                'elapsed_seconds': 12.0,
                'first_token_seconds': 3.0,
                'resource_rss_delta_mb': 120.0,
                'budget_alerts': ['a', 'b'],
                'issues': [],
                'regression_passed': True,
            },
            {
                'success': True,
                'generation_seconds': 20.0,
                'elapsed_seconds': 22.0,
                'first_token_seconds': 5.0,
                'resource_rss_delta_mb': 320.0,
                'budget_alerts': ['c'],
                'issues': [],
                'regression_passed': True,
            },
            {
                'success': False,
                'issues': ['x'],
                'regression_passed': False,
            },
        ],
    )
    latency_metrics = summary.get('latency_metrics', {})
    resource_metrics = summary.get('resource_metrics', {})
    assert isinstance(latency_metrics, dict)
    assert isinstance(resource_metrics, dict)
    assert latency_metrics.get('elapsed_seconds_p50') == 12.0
    assert latency_metrics.get('elapsed_seconds_max') == 22.0
    assert latency_metrics.get('first_token_seconds_p50') == 3.0
    assert resource_metrics.get('rss_delta_mb_max') == 320.0
    assert summary.get('budget_alert_total') == 3


def test_regression_query_bank_filters_requires_multi_year() -> None:
    queries_single_year = generate_queries.load_regression_query_bank(years=[2024])
    queries_multi_year = generate_queries.load_regression_query_bank(years=[2023, 2024])

    assert any(q.get('skip_if') == 'requires_multi_year' for q in queries_multi_year)
    assert all(q.get('skip_if') != 'requires_multi_year' for q in queries_single_year)


def test_research_golden_query_bank_filters_requires_multi_year() -> None:
    queries_single_year = generate_queries.load_research_golden_query_bank(years=[2024])
    queries_multi_year = generate_queries.load_research_golden_query_bank(years=[2023, 2024])

    assert any(q.get('skip_if') == 'requires_multi_year' for q in queries_multi_year)
    assert all(q.get('skip_if') != 'requires_multi_year' for q in queries_single_year)


def test_pipeline_run_evaluate_uses_analysis_only_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_command(
        cmd: list[str],
        *,
        timeout_seconds: float | None = None,
        env_overrides: dict[str, str] | None = None,
    ):
        captured['cmd'] = cmd
        captured['timeout_seconds'] = timeout_seconds
        captured['env_overrides'] = env_overrides
        return True, ''

    monkeypatch.setattr(pipeline, '_run_command', _fake_run_command)

    ok, _ = pipeline.run_evaluate(
        run_id='run-test',
        queries_file=None,
        query_timeout_seconds=123.0,
        timeout_seconds=456.0,
    )

    assert ok
    cmd = captured.get('cmd')
    assert isinstance(cmd, list)
    assert '--response-mode' not in cmd


def test_validate_query_expectations_extended_output_shape() -> None:
    query_item = {
        'output_shape': {
            'section_order': ['Executive Summary', 'Action Checklist'],
            'required_numeric_count': 3,
            'required_years': [2022, 2023],
            'required_bullet_depth': 3,
            'must_call_out_missing_evidence': True,
        },
    }
    answer = """
Executive Summary
Value 100 and case 200 and id 300.
2022 and 2023 included.
  - level1
    - level2
      - level3
Evidence gap identified for missing records.
Action Checklist
""".strip()
    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - test helper behavior
        query_item=query_item,
        sections={},
        answer=answer,
        sources_count=3,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert failures == []


def test_section_order_uses_heading_anchors_not_body_mentions() -> None:
    query_item = {
        'output_shape': {
            'section_order': [
                'Executive Summary',
                'Year-by-Year Evidence Map',
                'Document Group Deep Dive',
                'Risks and Gaps',
                'Action Checklist',
            ],
        },
    }
    answer = """
## 1) Executive Summary
This summary references Risks and Gaps and the Action Checklist in narrative form.

## 2) Year-by-Year Evidence Map
Details by year.

## 3) Document Group Deep Dive
Detailed evidence.

## 4) Risks and Gaps
Open issues.

## 5) Action Checklist
Next actions.
""".strip()

    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - unit-test helper behavior
        query_item=query_item,
        sections={},
        answer=answer,
        sources_count=3,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert failures == []


def test_evaluation_summary_builder_counts_regression_failures() -> None:
    results = [
        {'success': True, 'regression_passed': True, 'issues': [], 'generation_seconds': 1.0},
        {'success': True, 'regression_passed': False, 'issues': [], 'generation_seconds': 2.0},
        {'success': False, 'issues': ['query_execution_timeout']},
    ]
    summary = evaluate._build_summary(  # noqa: SLF001 - unit-test helper behavior
        run_id='run-test',
        total_queries=3,
        results=results,
        aborted_reason=None,
    )
    assert summary['run_id'] == 'run-test'
    assert summary['total_queries'] == 3
    assert summary['successful'] == 2
    assert summary['failed'] == 1
    assert summary['regression_failed'] == 1


def test_trace_summary_envelope_builds_stable_diagnostics_fields() -> None:
    writer = _ChatTraceWriter(chat_id='chat-1', message_id='msg-1', chat_type='evaluation', run_id='run-1')
    writer.record('intent', {'intent': 'aggregate', 'subtype': 'aggregate_by_period', 'query_type': 'coverage'})
    writer.record('retrieval', {'raw_chunks_count': 12, 'matching_files': 5, 'files_covered_after_fallback': 5})
    writer.record('llm', {'total_elapsed_ms': 4200.0, 'token_count': 333})
    writer.record('sources', {'count': 4})
    writer.record('response', {'answer_length': 810, 'display_answer_length': 750, 'sources_count': 4})

    summary = writer.get_summary_envelope()
    diagnostics = summary['diagnostics']

    assert summary['schema'] == 'informity.chat_trace.summary'
    assert diagnostics['query_type'] == 'coverage'
    assert diagnostics['raw_chunks_count'] == 12
    assert diagnostics['sources_count'] == 4
    assert diagnostics['generation_seconds'] == 4.2
    assert diagnostics['answer_length'] == 810


def test_extract_diagnostics_fields_prefers_summary_envelope() -> None:
    sections = {
        'intent': {'query_type': 'focused'},
        'retrieval': {'raw_chunks_count': 2},
        'sources': {'count': 1},
        'response': {'answer_length': 50},
    }
    summary = {
        'diagnostics': {
            'query_type': 'coverage',
            'raw_chunks_count': 11,
            'sources_count': 3,
            'generation_seconds': 5.5,
            'answer_length': 120,
        },
    }
    extracted = evaluate._extract_diagnostics_fields_from_trace(  # noqa: SLF001 - unit-test helper behavior
        sections=sections,
        summary=summary,
        fallback_generation_seconds=9.9,
        fallback_answer_length=900,
    )

    assert extracted['query_type'] == 'coverage'
    assert extracted['raw_chunks_count'] == 11
    assert extracted['sources_count'] == 3
    assert extracted['generation_seconds'] == 5.5
    assert extracted['answer_length'] == 120


def test_validate_query_expectations_detects_continuation_drift_refusal() -> None:
    query_item = {
        'continuation_behavior': {
            'forbid_clarification_or_refusal': True,
        },
    }
    answer = 'No documents found for this continuation request. Could you clarify the scope?'
    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - unit-test helper behavior
        query_item=query_item,
        sections={},
        answer=answer,
        sources_count=0,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert any(item.startswith('continuation_drift_refusal:') for item in failures)


def test_validate_query_expectations_uses_absolute_min_words_floor() -> None:
    query_item = {
        'regression_case': 'analysis_word_floor',
        'output_shape': {
            'min_words': 900,
        },
    }
    answer = 'word ' * 1400
    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - unit-test helper behavior
        query_item=query_item,
        sections={},
        answer=answer,
        sources_count=1,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
        model_filename='Qwen3-30B-A3B-Q5_K_M.gguf',
    )
    assert not any(str(item).startswith('word_count_below_threshold:') for item in failures)


def test_validate_query_expectations_forbidden_terms_supports_regex() -> None:
    query_item = {
        'forbidden_terms': [
            r'regex:\b\d{3}-\d{2}-\d{4}\b',
        ],
    }
    answer = 'Extracted values include SSN-like identifier 000-00-0000 in one section.'
    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - unit-test helper behavior
        query_item=query_item,
        sections={},
        answer=answer,
        sources_count=1,
        unsupported_claim_count=0,
        evidence_coverage_rate=1.0,
        not_found_count=0,
    )
    assert any(str(item).startswith('contains_forbidden_term:') for item in failures)


def test_validate_query_expectations_forbidden_terms_invalid_regex_is_ignored() -> None:
    query_item = {
        'forbidden_terms': [
            'regex:[invalid',
        ],
    }
    failures = evaluate._validate_query_expectations(  # noqa: SLF001 - unit-test helper behavior
        query_item=query_item,
        sections={},
        answer='clean answer without forbidden content',
        sources_count=1,
        unsupported_claim_count=0,
        evidence_coverage_rate=1.0,
        not_found_count=0,
    )
    assert failures == []
