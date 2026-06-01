#!/usr/bin/env python3
# ==============================================================================
# Informity AI — Translation Pipeline Evaluation Matrix
#
# Selects a sample of indexed files across size and extension categories,
# runs each through the translation pipeline, and produces a Markdown report.
#
# Usage:
#   uv run python tools/diagnostics/translate_eval.py
#   uv run python tools/diagnostics/translate_eval.py --n 8 --language French
#   uv run python tools/diagnostics/translate_eval.py --max-mb 1 --language German --output /tmp/report.md
#   uv run python tools/diagnostics/translate_eval.py --api http://localhost:8420
#
# Options:
#   --api URL          API base URL (default: http://localhost:8420)
#   --n N              Number of files to translate (default: 5)
#   --language LANG    Target language (default: Spanish)
#   --tone TONE        Tone: natural, formal, literal (default: natural)
#   --max-mb N         Max file size in MB (default: 2)
#   --output PATH      Write report to this path (default: tools/diagnostics/evals/translate_eval_TIMESTAMP.md)
#   --timeout N        Per-job timeout in seconds (default: 1200)
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

os.environ['INFORMITY_SUPPRESS_CONSOLE_LOGS'] = '1'

import httpx  # noqa: E402  (after env setup)


# ==============================================================================
# Data classes
# ==============================================================================

@dataclass
class FileCandidate:
    file_id:   int
    filename:  str
    extension: str
    size_bytes: int
    page_count: int | None

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024

    @property
    def size_label(self) -> str:
        kb = self.size_kb
        if kb < 100:
            return 'small'
        if kb < 512:
            return 'medium'
        return 'large'


@dataclass
class SectionResult:
    index: int
    title: str | None
    text_len: int       # characters of translated output
    completed: bool
    error: str | None = None


@dataclass
class JobResult:
    file:             FileCandidate
    language:         str
    tone:             str
    job_id:           str | None          = None
    status:           str                 = 'unknown'
    error:            str | None          = None

    # Timing
    started_at:       float               = 0.0
    glossary_done_at: float | None        = None
    sections_ready_at: float | None       = None
    first_section_at: float | None        = None
    finished_at:      float | None        = None

    # Sections
    section_count:    int | None          = None
    sections:         list[SectionResult] = field(default_factory=list)
    glossary_terms:   int                 = 0
    retry_events:     int                 = 0

    # Output
    total_output_chars: int               = 0

    @property
    def elapsed_s(self) -> float | None:
        if self.finished_at and self.started_at:
            return self.finished_at - self.started_at
        return None

    @property
    def time_to_first_section_s(self) -> float | None:
        if self.first_section_at and self.started_at:
            return self.first_section_at - self.started_at
        return None

    @property
    def completed_sections(self) -> int:
        return sum(1 for s in self.sections if s.completed)

    @property
    def failed_sections(self) -> int:
        return sum(1 for s in self.sections if not s.completed)

    @property
    def success_rate(self) -> float:
        total = len(self.sections)
        return self.completed_sections / total if total else 0.0

    @property
    def chars_per_second(self) -> float | None:
        if self.elapsed_s and self.elapsed_s > 0 and self.total_output_chars > 0:
            return self.total_output_chars / self.elapsed_s
        return None

    @property
    def pages_per_minute(self) -> float | None:
        pages = self.file.page_count
        elapsed = self.elapsed_s
        if pages and elapsed and elapsed > 0:
            return (pages / elapsed) * 60
        return None


# ==============================================================================
# File selection
# ==============================================================================

def fetch_candidates(client: httpx.Client, api_base: str, max_bytes: int) -> list[FileCandidate]:
    resp = client.get(
        f'{api_base}/api/files',
        params={'limit': 200, 'sort': 'size_bytes', 'order': 'asc'},
        timeout=httpx.Timeout(15.0),
    )
    resp.raise_for_status()
    data = resp.json()
    files = data.get('files') or []
    candidates = []
    for f in files:
        sb = int(f.get('size_bytes') or 0)
        if sb <= 0 or sb > max_bytes:
            continue
        candidates.append(FileCandidate(
            file_id=int(f['id']),
            filename=str(f.get('filename') or f.get('path') or f'file-{f["id"]}'),
            extension=str(f.get('extension') or '').lower(),
            size_bytes=sb,
            page_count=f.get('page_count'),
        ))
    return candidates


def select_matrix_sample(candidates: list[FileCandidate], n: int) -> list[FileCandidate]:
    """
    Stratified sample across (size_label × extension) buckets.
    Falls back to random if not enough stratification is possible.
    """
    if len(candidates) <= n:
        return list(candidates)

    # Group by (size_label, extension)
    buckets: dict[tuple[str, str], list[FileCandidate]] = defaultdict(list)
    for c in candidates:
        buckets[(c.size_label, c.extension)].append(c)

    selected: list[FileCandidate] = []
    bucket_list = list(buckets.values())
    random.shuffle(bucket_list)

    # Round-robin across buckets
    idx = 0
    while len(selected) < n and any(bucket_list):
        bucket = bucket_list[idx % len(bucket_list)]
        if bucket:
            chosen = random.choice(bucket)
            bucket.remove(chosen)
            selected.append(chosen)
        idx += 1

    return selected[:n]


# ==============================================================================
# SSE parsing + job execution
# ==============================================================================

def _parse_sse_lines(lines_iter) -> tuple[str, str] | None:
    """Yield (event, data) pairs from raw SSE lines iterator."""
    current_event = ''
    data_parts: list[str] = []
    for raw_line in lines_iter:
        line = raw_line.strip() if raw_line else ''
        if line.startswith('event:'):
            if data_parts and current_event:
                yield current_event, '\n'.join(data_parts).strip()
                data_parts = []
            current_event = line[6:].strip()
        elif line.startswith('data:'):
            data_parts.append(line[5:].lstrip(' '))
        elif line == '':
            if data_parts and current_event:
                yield current_event, '\n'.join(data_parts).strip()
                data_parts = []
                current_event = ''


def wait_for_job_terminal(
    client: httpx.Client,
    api_base: str,
    job_id: str,
    timeout_s: float = 1800.0,
    poll_interval: float = 5.0,
) -> str:
    """Poll job status until terminal (done/failed/stalled). Returns final status."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = client.get(
                f'{api_base}/api/translate/jobs/{job_id}',
                timeout=httpx.Timeout(10.0),
            )
            if resp.status_code == 404:
                return 'not_found'
            resp.raise_for_status()
            status = resp.json().get('status', 'unknown')
            if status in ('done', 'failed', 'stalled'):
                return status
        except Exception:
            pass
        time.sleep(poll_interval)
    return 'timeout'


def run_translate_job(
    client: httpx.Client,
    api_base: str,
    candidate: FileCandidate,
    language: str,
    tone: str,
    timeout_s: float,
) -> JobResult:
    result = JobResult(file=candidate, language=language, tone=tone)
    result.started_at = time.monotonic()

    # Create job — poll until LLM is free (a previous section may be translating,
    # which holds the lock for up to TRANSLATE_SECTION_TIMEOUT_S = 120s).
    max_wait_s = 300  # 5 minutes max wait
    wait_deadline = time.monotonic() + max_wait_s
    poll_interval = 15  # seconds between retries
    job_created = False
    while True:
        try:
            resp = client.post(
                f'{api_base}/api/translate/jobs',
                json={'file_id': candidate.file_id, 'target_language': language,
                      'tone': tone, 'output_mode': 'markdown'},
                timeout=httpx.Timeout(15.0),
            )
            if resp.status_code == 409:
                remaining = int(wait_deadline - time.monotonic())
                if remaining <= 0:
                    result.status = 'skipped'
                    result.error = f'LLM busy for >{max_wait_s}s — giving up'
                    result.finished_at = time.monotonic()
                    return result
                print(f'         ⏳  LLM busy — retrying in {poll_interval}s ({remaining}s left)…', flush=True)
                time.sleep(poll_interval)
                continue
            resp.raise_for_status()
            result.job_id = resp.json()['job_id']
            job_created = True
            break
        except Exception as exc:
            result.status = 'create_failed'
            result.error = str(exc)
            result.finished_at = time.monotonic()
            return result

    if not job_created:
        return result

    # Stream SSE events
    try:
        with client.stream(
            'GET',
            f'{api_base}/api/translate/jobs/{result.job_id}/events',
            timeout=httpx.Timeout(float(timeout_s), connect=10.0),
        ) as resp:
            resp.raise_for_status()
            for event, raw in _parse_sse_lines(resp.iter_lines()):
                now = time.monotonic()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {}

                if event == 'ping':
                    pass
                elif event == 'glossary_done':
                    result.glossary_terms = int(data.get('term_count') or 0)
                    result.glossary_done_at = now
                elif event == 'sections_ready':
                    result.section_count = int(data.get('section_count') or 0)
                    result.sections_ready_at = now
                elif event == 'section_retry':
                    result.retry_events += 1
                elif event == 'section_done':
                    text = str(data.get('text') or '')
                    if result.first_section_at is None:
                        result.first_section_at = now
                    result.sections.append(SectionResult(
                        index=int(data.get('section_index') or 0),
                        title=data.get('section_title'),
                        text_len=len(text),
                        completed=True,
                    ))
                    result.total_output_chars += len(text)
                elif event == 'section_failed':
                    result.sections.append(SectionResult(
                        index=int(data.get('section_index') or 0),
                        title=None,
                        text_len=0,
                        completed=False,
                        error=str(data.get('error') or 'failed'),
                    ))
                elif event in ('job_done', 'job_failed', 'job_stalled'):
                    result.status = event.replace('job_', '')
                    if 'error' in data:
                        result.error = str(data['error'])
                    result.finished_at = now
                    break

        if result.finished_at is None:
            result.finished_at = time.monotonic()
            if result.completed_sections > 0:
                result.status = 'done'
            else:
                result.status = 'stream_ended'

    except httpx.TimeoutException:
        result.status = 'timeout'
        result.error = f'Stream timeout after {timeout_s}s'
        result.finished_at = time.monotonic()
    except Exception as exc:
        result.status = 'stream_error'
        result.error = str(exc)
        result.finished_at = time.monotonic()

    # If the stream ended unexpectedly but the job may still be running in the
    # backend (e.g. stream_error before first event), poll until terminal so the
    # next job doesn't hit a 409.
    if result.job_id and result.status in ('stream_error', 'stream_ended', 'timeout'):
        print('         ⏳  Stream ended early — polling job status until terminal…', flush=True)
        final = wait_for_job_terminal(client, api_base, result.job_id, timeout_s=float(timeout_s))
        if result.status == 'stream_error':
            result.status = final  # use the actual terminal status
        if result.finished_at is None:
            result.finished_at = time.monotonic()

    return result


# ==============================================================================
# Report generation
# ==============================================================================

def _fmt_s(v: float | None) -> str:
    return f'{v:.1f}s' if v is not None else '—'

def _fmt_n(v: int | None) -> str:
    return str(v) if v is not None else '—'

def _pct(v: float) -> str:
    return f'{v * 100:.0f}%'


def generate_report(results: list[JobResult], language: str, tone: str, started_wall: str) -> str:
    lines: list[str] = []

    # Header
    lines += [
        '# Translation Pipeline Evaluation Report',
        '',
        f'**Generated:** {started_wall}  ',
        f'**Language:** {language}  ',
        f'**Tone:** {tone}  ',
        f'**Files evaluated:** {len(results)}  ',
        '',
        '---',
        '',
    ]

    # Summary table
    lines += [
        '## Summary',
        '',
        '| # | File | Ext | Size | Pages | Sections | ✓ | ✗ | Retries | Glossary | First §  | Total | §/s output | Status |',
        '|---|------|-----|------|-------|----------|---|---|---------|----------|---------|-------|------------|--------|',
    ]

    total_sections = 0
    total_completed = 0
    total_failed = 0
    total_retries = 0
    total_elapsed: list[float] = []
    total_cps: list[float] = []

    for i, r in enumerate(results, 1):
        section_total = len(r.sections)
        total_sections += section_total
        total_completed += r.completed_sections
        total_failed += r.failed_sections
        total_retries += r.retry_events
        if r.elapsed_s:
            total_elapsed.append(r.elapsed_s)
        if r.chars_per_second:
            total_cps.append(r.chars_per_second)

        name = r.file.filename
        if len(name) > 35:
            name = '…' + name[-33:]

        status_icon = {
            'done': '✅', 'failed': '❌', 'stalled': '⚠️', 'timeout': '⏱️',
            'skipped': '⏭️', 'create_failed': '💥', 'stream_error': '💥',
        }.get(r.status, '❓')

        cps = f'{r.chars_per_second:.0f}' if r.chars_per_second else '—'

        lines.append(
            f'| {i} | `{name}` | {r.file.extension or "—"} '
            f'| {r.file.size_kb:.0f}KB '
            f'| {_fmt_n(r.file.page_count)} '
            f'| {_fmt_n(r.section_count)} '
            f'| {r.completed_sections} | {r.failed_sections} '
            f'| {r.retry_events} '
            f'| {r.glossary_terms} '
            f'| {_fmt_s(r.time_to_first_section_s)} '
            f'| {_fmt_s(r.elapsed_s)} '
            f'| {cps} '
            f'| {status_icon} {r.status} |'
        )

    lines += ['', '']

    # Aggregate stats
    success_count = sum(1 for r in results if r.status == 'done')
    avg_elapsed = sum(total_elapsed) / len(total_elapsed) if total_elapsed else None
    avg_cps = sum(total_cps) / len(total_cps) if total_cps else None
    section_success_rate = total_completed / total_sections if total_sections else 0

    lines += [
        '## Aggregate Metrics',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| Jobs completed successfully | {success_count} / {len(results)} ({_pct(success_count / len(results) if results else 0)}) |',
        f'| Total sections translated | {total_completed} / {total_sections} ({_pct(section_success_rate)}) |',
        f'| Total section failures | {total_failed} |',
        f'| Total retry events | {total_retries} |',
        f'| Avg job duration | {_fmt_s(avg_elapsed)} |',
        f'| Avg chars/second output | {f"{avg_cps:.0f}" if avg_cps else "—"} |',
        '',
        '---',
        '',
    ]

    # Per-file detail
    lines += ['## Per-File Detail', '']
    for i, r in enumerate(results, 1):
        lines += [
            f'### {i}. `{r.file.filename}`',
            '',
            f'- **Status:** {r.status}',
            f'- **Size:** {r.file.size_kb:.1f} KB ({r.file.page_count or "?"}p)',
            f'- **Glossary terms:** {r.glossary_terms}',
            f'- **Sections detected:** {r.section_count or "?"}',
            f'- **Sections completed / failed:** {r.completed_sections} / {r.failed_sections}',
            f'- **Retry events:** {r.retry_events}',
            f'- **Time to first section:** {_fmt_s(r.time_to_first_section_s)}',
            f'- **Total elapsed:** {_fmt_s(r.elapsed_s)}',
            f'- **Output chars/sec:** {f"{r.chars_per_second:.0f}" if r.chars_per_second else "—"}',
            f'- **Output chars total:** {r.total_output_chars:,}',
        ]
        if r.error:
            lines += [f'- **Error:** `{r.error}`']
        if r.sections:
            lines += ['']
            lines += ['| § | Title | Output chars | Result |']
            lines += ['|---|-------|-------------|--------|']
            for s in sorted(r.sections, key=lambda x: x.index):
                title = (s.title or '(untitled)')[:40]
                icon = '✅' if s.completed else '❌'
                err = f' — {s.error}' if s.error else ''
                lines.append(f'| {s.index} | {title} | {s.text_len:,} | {icon}{err} |')
        lines += ['']

    return '\n'.join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description='Translate pipeline evaluation matrix')
    parser.add_argument('--api', default='http://localhost:8420', metavar='URL')
    parser.add_argument('--n', type=int, default=5, metavar='N', help='Number of files to test')
    parser.add_argument('--language', default='Spanish', metavar='LANG')
    parser.add_argument('--tone', default='natural', choices=['natural', 'formal', 'literal'])
    parser.add_argument('--max-mb', type=float, default=2.0, metavar='MB')
    parser.add_argument('--output', default=None, metavar='PATH', help='Write report to file (also printed to stdout)')
    parser.add_argument('--timeout', type=int, default=1200, metavar='S', help='Per-job timeout seconds')
    parser.add_argument('--seed', type=int, default=None, metavar='N', help='Random seed for reproducible selection')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    max_bytes = int(args.max_mb * 1024 * 1024)
    started_wall = datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')

    print(f'\n🔍  Translate Eval Matrix  ·  {started_wall}', flush=True)
    print(f'    API:      {args.api}')
    print(f'    Language: {args.language}  |  Tone: {args.tone}')
    print(f'    Files:    {args.n}  (max {args.max_mb}MB each)\n', flush=True)

    with httpx.Client() as client:
        # Health check
        try:
            h = client.get(f'{args.api}/api/health', timeout=httpx.Timeout(5.0))
            h.raise_for_status()
        except Exception as exc:
            print(f'❌  Cannot reach API at {args.api}: {exc}', file=sys.stderr)
            print('    Make sure the Informity AI server is running.', file=sys.stderr)
            return 1

        # Fetch candidates
        print('📂  Fetching indexed files…', flush=True)
        try:
            candidates = fetch_candidates(client, args.api, max_bytes)
        except Exception as exc:
            print(f'❌  Failed to fetch files: {exc}', file=sys.stderr)
            return 1

        if not candidates:
            print(f'❌  No indexed files found under {args.max_mb}MB.', file=sys.stderr)
            return 1

        print(f'    Found {len(candidates)} eligible files.', flush=True)
        sample = select_matrix_sample(candidates, args.n)
        print(f'    Selected {len(sample)} for evaluation:\n', flush=True)
        for i, c in enumerate(sample, 1):
            print(f'    {i}. {c.filename} ({c.size_kb:.0f}KB, {c.extension})', flush=True)
        print('', flush=True)

        # Run jobs sequentially (single LLM)
        results: list[JobResult] = []
        for i, candidate in enumerate(sample, 1):
            print(f'[{i}/{len(sample)}] Translating: {candidate.filename}', flush=True)
            result = run_translate_job(
                client, args.api, candidate, args.language, args.tone, float(args.timeout)
            )
            elapsed = f'{result.elapsed_s:.0f}s' if result.elapsed_s else '—'
            sections_info = f'{result.completed_sections}/{len(result.sections)} sections'
            icon = '✅' if result.status == 'done' else '⚠️' if result.status in ('done', 'stalled') else '❌'
            print(f'         {icon}  {result.status}  |  {sections_info}  |  {elapsed}', flush=True)
            if result.error and result.status not in ('done',):
                print(f'         Error: {result.error}', flush=True)
            results.append(result)

        print('', flush=True)

    # Generate report
    report = generate_report(results, args.language, args.tone, started_wall)

    # Auto-save path
    ts = datetime.now(UTC).strftime('%Y%m%d_%H%M')
    # Default: tools/diagnostics/evals/ (gitignored via tools/)
    evals_dir = Path(__file__).parent / 'evals'
    evals_dir.mkdir(parents=True, exist_ok=True)
    auto_path = evals_dir / f'translate_eval_{ts}.md'
    output_path = Path(args.output) if args.output else auto_path

    output_path.write_text(report, encoding='utf-8')
    print(f'📄  Report saved: {output_path}', flush=True)
    print('', flush=True)

    # Print summary to stdout
    success = sum(1 for r in results if r.status == 'done')
    total_s = sum(1 for r in results if r.section_count)
    total_c = sum(r.completed_sections for r in results)
    print(f'Results: {success}/{len(results)} jobs completed  |  {total_c} sections translated')
    print('')

    # Also print report if it fits
    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
