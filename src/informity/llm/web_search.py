# ==============================================================================
# Informity AI — Assistant Web Search (Tavily)
# Optional external web-search provider for assistant mode.
# ==============================================================================

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import structlog

from informity.config import settings

log = structlog.get_logger(__name__)

_TAVILY_SEARCH_URL = 'https://api.tavily.com/search'
_MAX_QUERY_LENGTH = 512
_MAX_SNIPPET_LENGTH = 600


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    def search(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: float,
    ) -> list[SearchResult]:
        ...


class TavilyProvider:
    def __init__(self, api_key: str) -> None:
        self._api_key = str(api_key or '').strip()

    def search(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: float,
    ) -> list[SearchResult]:
        if not self._api_key:
            return []
        safe_query = str(query or '').strip()[:_MAX_QUERY_LENGTH]
        if not safe_query:
            return []

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
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
                raw = response.read().decode('utf-8')
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            log.warning('web_search_tavily_failed', error=str(exc))
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning('web_search_tavily_invalid_json')
            return []

        rows = parsed.get('results')
        if not isinstance(rows, list):
            return []

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
        return results


def search_web(query: str, *, allow_privacy_override: bool = False) -> list[SearchResult]:
    # Runtime gate: no network calls in full privacy mode.
    if settings.full_privacy and not allow_privacy_override:
        return []
    if not str(settings.tavily_api_key or '').strip():
        return []

    provider = TavilyProvider(api_key=settings.tavily_api_key)
    return provider.search(
        query=query,
        max_results=settings.web_search_max_results,
        timeout_seconds=settings.web_search_timeout_seconds,
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
