# ==============================================================================
# Informity AI — Assistant Web Search
# Optional external web-search providers for assistant mode.
# ==============================================================================

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog

from informity.config import settings

log = structlog.get_logger(__name__)

_TAVILY_SEARCH_URL = str(os.getenv('INFORMITY_TAVILY_SEARCH_URL') or 'https://api.tavily.com/search').strip()
_TAVILY_USAGE_URL = str(os.getenv('INFORMITY_TAVILY_USAGE_URL') or 'https://api.tavily.com/usage').strip()
_LINKUP_SEARCH_URL = str(os.getenv('INFORMITY_LINKUP_SEARCH_URL') or 'https://api.linkup.so/v1/search').strip()
_MAX_QUERY_LENGTH = 512
_MAX_SNIPPET_LENGTH = 600
_DEFAULT_LINKUP_DEPTH = 'standard'
_DEFAULT_LINKUP_OUTPUT_TYPE = 'searchResults'
_RETRYABLE_FAILOVER_STATUSES: frozenset[str] = frozenset({
    'api_key_missing',
    'auth_invalid',
    'quota_exceeded',
    'rate_limited',
    'network_error',
    'provider_error',
})
SearchProviderName = Literal['tavily', 'linkup']


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebSearchOutcome:
    results: list[SearchResult]
    status: str = 'ok'
    usage_used: int | None = None
    usage_limit: int | None = None
    provider_attempted: str | None = None
    provider_used: str | None = None
    failover_applied: bool = False


class SearchProvider(Protocol):
    def search(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: float,
    ) -> WebSearchOutcome:
        ...


def has_any_provider_api_key() -> bool:
    return any((
        bool(str(settings.tavily_api_key or '').strip()),
        bool(str(settings.linkup_api_key or '').strip()),
    ))


class TavilyProvider:
    def __init__(self, api_key: str) -> None:
        self._api_key = str(api_key or '').strip()

    def search(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: float,
    ) -> WebSearchOutcome:
        if not self._api_key:
            return WebSearchOutcome(results=[], status='api_key_missing')
        safe_query = str(query or '').strip()[:_MAX_QUERY_LENGTH]
        if not safe_query:
            return WebSearchOutcome(results=[])

        payload: dict[str, object] = {
            'api_key': self._api_key,
            'query': safe_query,
            'max_results': int(max(1, min(10, max_results))),
            'search_depth': 'basic',
            'include_answer': False,
            'include_raw_content': False,
        }

        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            _TAVILY_SEARCH_URL,
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._api_key}',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            status = _classify_tavily_http_error(exc)
            log.warning('web_search_tavily_failed', error=str(exc))
            usage_used, usage_limit = self._fetch_usage(timeout_seconds=timeout_seconds)
            return WebSearchOutcome(
                results=[],
                status=status,
                usage_used=usage_used,
                usage_limit=usage_limit,
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            log.warning('web_search_tavily_failed', error=str(exc))
            usage_used, usage_limit = self._fetch_usage(timeout_seconds=timeout_seconds)
            return WebSearchOutcome(
                results=[],
                status='network_error',
                usage_used=usage_used,
                usage_limit=usage_limit,
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning('web_search_tavily_invalid_json')
            usage_used, usage_limit = self._fetch_usage(timeout_seconds=timeout_seconds)
            return WebSearchOutcome(
                results=[],
                status='provider_error',
                usage_used=usage_used,
                usage_limit=usage_limit,
            )

        rows = parsed.get('results')
        if not isinstance(rows, list):
            usage_used, usage_limit = self._fetch_usage(timeout_seconds=timeout_seconds)
            return WebSearchOutcome(
                results=[],
                status='provider_error',
                usage_used=usage_used,
                usage_limit=usage_limit,
            )

        results: list[SearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get('title') or '').strip()
            url = str(row.get('url') or '').strip()
            snippet = str(row.get('content') or '').strip()
            if not title or not url:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet[:_MAX_SNIPPET_LENGTH],
                ),
            )
        # Prefer live usage endpoint after each successful search so UI reflects
        # current Tavily credit counters; fall back to search payload when needed.
        payload_usage_used, payload_usage_limit = _extract_usage_from_search_response(parsed)
        live_usage_used, live_usage_limit = self._fetch_usage(timeout_seconds=timeout_seconds)
        usage_used = live_usage_used if live_usage_used is not None else payload_usage_used
        usage_limit = live_usage_limit if live_usage_limit is not None else payload_usage_limit
        return WebSearchOutcome(
            results=results,
            status='ok',
            usage_used=usage_used,
            usage_limit=usage_limit,
        )

    def _fetch_usage(self, *, timeout_seconds: float) -> tuple[int | None, int | None]:
        request = urllib.request.Request(
            _TAVILY_USAGE_URL,
            headers={
                'Authorization': f'Bearer {self._api_key}',
            },
            method='GET',
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, min(4.0, float(timeout_seconds)))) as response:
                raw = response.read().decode('utf-8')
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None, None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        key = payload.get('key')
        account = payload.get('account') if isinstance(payload.get('account'), dict) else {}
        if not isinstance(key, dict):
            key = {}
        usage = _safe_int(key.get('usage'))
        limit = _safe_int(key.get('limit'))
        if usage is None:
            usage = (
                _safe_int(key.get('search_usage'))
                or _safe_int(account.get('plan_usage'))
                or _safe_int(account.get('search_usage'))
            )
        if limit is None:
            limit = (
                _safe_int(account.get('plan_limit'))
                or _safe_int(account.get('limit'))
            )
        return usage, limit


def _safe_int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _extract_usage_from_search_response(payload: dict[str, object]) -> tuple[int | None, int | None]:
    key = payload.get('key')
    if isinstance(key, dict):
        usage = _safe_int(key.get('usage'))
        limit = _safe_int(key.get('limit'))
        if usage is not None and limit is not None:
            return usage, limit
    return None, None


def _classify_tavily_http_error(exc: urllib.error.HTTPError) -> str:
    code = int(getattr(exc, 'code', 0) or 0)
    if code == 429:
        return 'rate_limited'
    if code in {401, 403}:
        return 'auth_invalid'
    if code in {402, 432, 433}:
        return 'quota_exceeded'
    # Attempt best-effort content-based disambiguation.
    try:
        body = exc.read().decode('utf-8', errors='ignore').casefold()
    except Exception:  # noqa: BLE001
        body = ''
    if 'quota' in body or 'credit' in body or 'insufficient' in body:
        return 'quota_exceeded'
    if 'rate limit' in body or 'too many requests' in body:
        return 'rate_limited'
    if 'unauthorized' in body or 'invalid api key' in body or 'forbidden' in body:
        return 'auth_invalid'
    return 'provider_error'


class LinkupProvider:
    def __init__(self, api_key: str) -> None:
        self._api_key = str(api_key or '').strip()

    def search(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: float,
    ) -> WebSearchOutcome:
        if not self._api_key:
            return WebSearchOutcome(results=[], status='api_key_missing')
        safe_query = str(query or '').strip()[:_MAX_QUERY_LENGTH]
        if not safe_query:
            return WebSearchOutcome(results=[])

        payload: dict[str, object] = {
            'q': safe_query,
            'depth': _DEFAULT_LINKUP_DEPTH,
            'outputType': _DEFAULT_LINKUP_OUTPUT_TYPE,
            'maxResults': int(max(1, min(10, max_results))),
        }
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            _LINKUP_SEARCH_URL,
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._api_key}',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            log.warning('web_search_linkup_failed', error=str(exc))
            return WebSearchOutcome(results=[], status=_classify_linkup_http_error(exc))
        except (TimeoutError, urllib.error.URLError) as exc:
            log.warning('web_search_linkup_failed', error=str(exc))
            return WebSearchOutcome(results=[], status='network_error')

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning('web_search_linkup_invalid_json')
            return WebSearchOutcome(results=[], status='provider_error')

        return WebSearchOutcome(
            results=_extract_linkup_results(parsed),
            status='ok',
        )


def _classify_linkup_http_error(exc: urllib.error.HTTPError) -> str:
    code = int(getattr(exc, 'code', 0) or 0)
    if code == 429:
        return 'rate_limited'
    if code in {401, 403}:
        return 'auth_invalid'
    if code == 402:
        return 'quota_exceeded'
    try:
        body = exc.read().decode('utf-8', errors='ignore').casefold()
    except Exception:  # noqa: BLE001
        body = ''
    if 'quota' in body or 'credit' in body or 'insufficient' in body:
        return 'quota_exceeded'
    if 'rate limit' in body or 'too many requests' in body:
        return 'rate_limited'
    if 'unauthorized' in body or 'invalid api key' in body or 'forbidden' in body:
        return 'auth_invalid'
    return 'provider_error'


def _extract_linkup_results(payload: object) -> list[SearchResult]:
    if not isinstance(payload, dict):
        return []
    candidates: list[object] = []
    for key in ('results', 'sources', 'items'):
        rows = payload.get(key)
        if isinstance(rows, list):
            candidates.extend(rows)

    results: list[SearchResult] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        title = str(
            row.get('title')
            or row.get('name')
            or row.get('source')
            or row.get('domain')
            or '',
        ).strip()
        url = str(row.get('url') or row.get('link') or '').strip()
        snippet = str(
            row.get('snippet')
            or row.get('content')
            or row.get('description')
            or row.get('text')
            or '',
        ).strip()
        if not title and url:
            title = url
        if not title or not url:
            continue
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet[:_MAX_SNIPPET_LENGTH],
            ),
        )
    return results


def _resolve_provider_order() -> list[SearchProviderName]:
    primary = str(settings.web_search_primary_provider or 'tavily').strip().lower()
    if primary == 'linkup':
        return ['linkup', 'tavily']
    return ['tavily', 'linkup']


def _build_provider(provider_name: SearchProviderName) -> SearchProvider:
    if provider_name == 'linkup':
        return LinkupProvider(api_key=settings.linkup_api_key)
    return TavilyProvider(api_key=settings.tavily_api_key)


def _search_with_provider(
    provider_name: SearchProviderName,
    *,
    query: str,
) -> WebSearchOutcome:
    outcome = _build_provider(provider_name).search(
        query=query,
        max_results=settings.web_search_max_results,
        timeout_seconds=settings.web_search_timeout_seconds,
    )
    return WebSearchOutcome(
        results=outcome.results,
        status=outcome.status,
        usage_used=outcome.usage_used,
        usage_limit=outcome.usage_limit,
        provider_attempted=provider_name,
        provider_used=provider_name if outcome.status == 'ok' else None,
        failover_applied=False,
    )


def search_web(query: str, *, allow_privacy_override: bool = False) -> WebSearchOutcome:
    # Runtime gate: no network calls in full privacy mode.
    if settings.full_privacy and not allow_privacy_override:
        return WebSearchOutcome(results=[], status='privacy_blocked')
    if not has_any_provider_api_key():
        return WebSearchOutcome(results=[], status='api_key_missing')

    provider_order = _resolve_provider_order()
    primary_provider = provider_order[0]
    fallback_provider = provider_order[1]

    primary_outcome = _search_with_provider(primary_provider, query=query)
    if primary_outcome.status == 'ok':
        return primary_outcome

    if primary_outcome.status == 'privacy_blocked':
        return primary_outcome

    if primary_outcome.status not in _RETRYABLE_FAILOVER_STATUSES:
        return primary_outcome

    fallback_outcome = _search_with_provider(fallback_provider, query=query)
    if fallback_outcome.status == 'ok':
        return WebSearchOutcome(
            results=fallback_outcome.results,
            status='ok',
            usage_used=fallback_outcome.usage_used,
            usage_limit=fallback_outcome.usage_limit,
            provider_attempted=f'{primary_provider}->{fallback_provider}',
            provider_used=fallback_provider,
            failover_applied=True,
        )

    terminal_status = fallback_outcome.status
    if fallback_outcome.status == 'api_key_missing' and primary_outcome.status != 'api_key_missing':
        terminal_status = primary_outcome.status

    return WebSearchOutcome(
        results=[],
        status=terminal_status,
        usage_used=fallback_outcome.usage_used or primary_outcome.usage_used,
        usage_limit=fallback_outcome.usage_limit or primary_outcome.usage_limit,
        provider_attempted=f'{primary_provider}->{fallback_provider}',
        provider_used=None,
        failover_applied=False,
    )


def format_search_context(results: list[SearchResult]) -> str:
    if not results:
        return 'No web results were available for this query.'

    lines: list[str] = ['Web Search Results:']
    for idx, result in enumerate(results, start=1):
        lines.append(f'[{idx}] {result.title}')
        lines.append(f'URL: {result.url}')
        if result.snippet:
            lines.append(f'Snippet: {result.snippet}')
        lines.append('')
    return '\n'.join(lines).strip()
