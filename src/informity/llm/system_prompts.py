# ==============================================================================
# Informity AI — System Prompt Registry
# Backward-compatible exports sourced from centralized persona registry.
# ==============================================================================

from informity.llm.personas import get_mode_prompt

SIMPLE_ASSISTANT_SYSTEM_PROMPT = get_mode_prompt('assistant_default')
SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT = get_mode_prompt('assistant_web_search_synthesis')
SIMPLE_RESEARCHER_SYSTEM_PROMPT = get_mode_prompt('researcher_default')
SIMPLE_CHAT_SUMMARY_SYSTEM_PROMPT = get_mode_prompt('chat_summary')

__all__ = [
    'SIMPLE_ASSISTANT_SYSTEM_PROMPT',
    'SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT',
    'SIMPLE_RESEARCHER_SYSTEM_PROMPT',
    'SIMPLE_CHAT_SUMMARY_SYSTEM_PROMPT',
]
