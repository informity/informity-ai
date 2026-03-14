from __future__ import annotations

import json
from pathlib import Path

from tools.diagnostics import run_control


def test_claim_active_run_creates_lock(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path
    ok, reason, previous = run_control.claim_active_run(
        diagnostics_dir,
        run_id='run-a',
        step='startup',
        stale_after_seconds=60,
        force_takeover=False,
    )
    assert ok is True
    assert reason in {'lock_acquired', 'lock_takeover'}
    assert previous is None
    lock = run_control.read_active_lock(diagnostics_dir)
    assert lock is not None
    assert lock.get('run_id') == 'run-a'


def test_claim_active_run_rejects_alive_non_stale_lock(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path
    run_control.write_active_lock(diagnostics_dir, run_id='run-existing', step='evaluate')
    ok, reason, previous = run_control.claim_active_run(
        diagnostics_dir,
        run_id='run-new',
        step='startup',
        stale_after_seconds=600,
        force_takeover=False,
    )
    assert ok is False
    assert reason.startswith('active_run_in_progress:')
    assert isinstance(previous, dict)
    assert previous.get('run_id') == 'run-existing'


def test_claim_active_run_takes_over_dead_pid_lock(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path
    lock_path = run_control.active_lock_path(diagnostics_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                'run_id': 'dead-run',
                'pid': 999999,  # best-effort non-existent pid
                'step': 'evaluate',
                'heartbeat_epoch': 0,
            }
        ),
        encoding='utf-8',
    )
    ok, reason, previous = run_control.claim_active_run(
        diagnostics_dir,
        run_id='run-new',
        step='startup',
        stale_after_seconds=1,
        force_takeover=False,
    )
    assert ok is True
    assert reason == 'lock_takeover'
    assert isinstance(previous, dict)
    current = run_control.read_active_lock(diagnostics_dir)
    assert current is not None
    assert current.get('run_id') == 'run-new'


def test_stop_active_run_marks_aborted_and_clears_lock(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path
    run_id = 'run-stop'
    run_control.write_active_lock(diagnostics_dir, run_id=run_id, step='evaluate', pid=999999)
    run_control.mark_run_status(
        diagnostics_dir,
        run_id=run_id,
        status='running',
    )
    manifest_file = diagnostics_dir / 'runs' / run_id / 'results' / 'pipeline_manifest.json'
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(
            {
                'run_id': run_id,
                'status': 'running',
                'completed': False,
                'reason': None,
            }
        ),
        encoding='utf-8',
    )

    result = run_control.stop_active_run(diagnostics_dir, reason='stopped_by_user')
    assert result.get('stopped') is True
    assert run_control.read_active_lock(diagnostics_dir) is None

    run_file = diagnostics_dir / 'runs' / run_id / 'results' / 'run.json'
    payload = json.loads(run_file.read_text(encoding='utf-8'))
    assert payload.get('status') == 'aborted'
    assert payload.get('aborted_reason') == 'stopped_by_user'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert manifest.get('status') == 'aborted'
    assert manifest.get('reason') == 'stopped_by_user'


def test_reconcile_orphaned_running_runs_marks_stale_runs_only(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path
    run_control.write_active_lock(diagnostics_dir, run_id='run-active', step='evaluate', pid=999999)
    run_control.mark_run_status(diagnostics_dir, run_id='run-active', status='running')
    run_control.mark_run_status(diagnostics_dir, run_id='run-stale', status='running')
    run_control.mark_run_status(diagnostics_dir, run_id='run-done', status='completed')

    reconciled = run_control.reconcile_orphaned_running_runs(
        diagnostics_dir,
        protected_run_id='run-new',
    )

    assert reconciled == ['run-stale']

    stale_payload = json.loads(
        (diagnostics_dir / 'runs' / 'run-stale' / 'results' / 'run.json').read_text(encoding='utf-8')
    )
    active_payload = json.loads(
        (diagnostics_dir / 'runs' / 'run-active' / 'results' / 'run.json').read_text(encoding='utf-8')
    )
    done_payload = json.loads(
        (diagnostics_dir / 'runs' / 'run-done' / 'results' / 'run.json').read_text(encoding='utf-8')
    )
    assert stale_payload.get('status') == 'aborted'
    assert stale_payload.get('aborted_reason') == 'orphaned_running_state_reconciled'
    assert active_payload.get('status') == 'running'
    assert done_payload.get('status') == 'completed'
