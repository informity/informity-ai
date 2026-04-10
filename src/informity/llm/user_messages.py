# ==============================================================================
# Informity AI — User-Facing Message Bank
# Centralized plain-language messages shown directly to end users.
# ==============================================================================

from __future__ import annotations

WEB_SEARCH_STATUS_MESSAGES: dict[str, str] = {
    'quota_exceeded': (
        'Web search is unavailable: Tavily API quota exceeded. '
        'Add credits or wait for your monthly reset.'
    ),
    'rate_limited': (
        'Web search is temporarily unavailable: Tavily rate limit reached. '
        'Please try again in a moment.'
    ),
    'auth_invalid': (
        'Web search failed: invalid Tavily API key. '
        'Update your key in Settings > Chat > Web Search.'
    ),
    'network_error': (
        'Web search is temporarily unavailable due to a network issue. '
        'Please try again.'
    ),
    'provider_error': 'Web search is temporarily unavailable. Please try again.',
}

EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE = (
    'Your knowledge base is empty. To get started with Researcher mode:\n'
    '1. Go to Settings > Data Sources and add your folders.\n'
    '2. Click Save Settings.\n'
    '3. Go to Dashboard and click Scan.\n\n'
    'Once indexing completes, come back and ask your question. '
    'Need a general answer now? Switch to Assistant mode.'
)

INSUFFICIENT_CONTEXT_RESEARCHER_MESSAGE = (
    'The available documents do not contain enough information to answer this question.'
)


def get_web_search_status_message(status: str) -> str:
    normalized_status = str(status or '').strip()
    return WEB_SEARCH_STATUS_MESSAGES.get(
        normalized_status,
        WEB_SEARCH_STATUS_MESSAGES['provider_error'],
    )
