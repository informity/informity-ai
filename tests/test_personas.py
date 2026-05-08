from informity.llm.personas import (
    MODE_REGISTRY,
    PERSONA_REGISTRY,
    ROLE_REGISTRY,
    compose_prompt,
    compose_persona_prompt,
    get_mode_prompt,
    get_persona_prompt,
    resolve_runtime_mode_id,
    resolve_runtime_persona_id,
)
from informity.llm.system_prompts import (
    SIMPLE_ASSISTANT_SYSTEM_PROMPT,
    SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT,
    SIMPLE_CHAT_SUMMARY_SYSTEM_PROMPT,
    SIMPLE_RESEARCHER_SYSTEM_PROMPT,
)

_EXPECTED_ASSISTANT_PROMPT = """You are Informity AI, a helpful AI assistant. Answer conversationally, clearly, and directly.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

You have no access to indexed documents, local files, or any private corpus unless the user explicitly provides content in this chat.
If asked to search files or cite corpus evidence, explain briefly that this is direct assistant chat without document retrieval.

Keep responses concise."""

_EXPECTED_ASSISTANT_WEB_SYNTHESIS_PROMPT = """You are Informity AI, a helpful AI assistant.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

Use provided web search context when relevant and answer directly.
If web context is insufficient, say what remains uncertain.
Keep responses concise."""

_EXPECTED_RESEARCHER_SIMPLE_PROMPT = """You are Informity AI, a helpful AI assistant. Answer questions conversationally and helpfully.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

You have access to a private document corpus.
Answer conversationally and directly. You do not need to cite documents for casual or conversational replies.
If asked about document search capabilities, describe them accurately but briefly.

Keep responses concise."""

_EXPECTED_CHAT_SUMMARY_PROMPT = """You are Informity AI, a helpful AI assistant.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

Task:
- Summarize this chat conversation only.
- Focus on topics discussed, key points, decisions, and open questions when present.
- Do not use external knowledge, web content, or document-corpus retrieval framing.
- If chat history is too limited, say that clearly and keep the response brief.

Keep responses concise."""


def test_registry_contains_core_default_personas() -> None:
    assert 'assistant_default' in MODE_REGISTRY
    assert 'researcher_default' in MODE_REGISTRY
    assert 'researcher_rag' in MODE_REGISTRY
    assert 'assistant_default' in PERSONA_REGISTRY
    assert 'researcher_default' in PERSONA_REGISTRY
    assert 'researcher_rag' in PERSONA_REGISTRY


def test_runtime_persona_resolution_by_mode() -> None:
    assert resolve_runtime_mode_id('assistant') == 'assistant_default'
    assert resolve_runtime_mode_id('researcher') == 'researcher_default'
    assert resolve_runtime_mode_id(None) == 'researcher_default'
    assert resolve_runtime_persona_id('assistant') == 'assistant_default'
    assert resolve_runtime_persona_id('researcher') == 'researcher_default'
    assert resolve_runtime_persona_id(None) == 'researcher_default'


def test_rag_persona_composition_adds_assistant_mode_policy_only_for_assistant() -> None:
    assistant_prompt = compose_persona_prompt(persona_id='researcher_rag', chat_mode='assistant')
    researcher_prompt = compose_persona_prompt(persona_id='researcher_rag', chat_mode='researcher')

    assert 'Answer using ONLY the available information from retrieved context' in assistant_prompt
    assert 'Assistant Mode Rules:' in assistant_prompt
    assert 'Assistant Mode Rules:' not in researcher_prompt


def test_legacy_prompt_exports_are_covered_by_registry_prompts() -> None:
    assert get_mode_prompt('assistant_default').startswith('You are Informity AI')
    assert get_persona_prompt('assistant_default').startswith('You are Informity AI')
    assert 'Summarize this chat conversation only.' in get_persona_prompt('chat_summary')


def test_persona_prompts_match_golden_baseline_exactly() -> None:
    assert get_mode_prompt('assistant_default') == _EXPECTED_ASSISTANT_PROMPT
    assert get_persona_prompt('assistant_default') == _EXPECTED_ASSISTANT_PROMPT
    assert get_persona_prompt('assistant_web_search_synthesis') == _EXPECTED_ASSISTANT_WEB_SYNTHESIS_PROMPT
    assert get_persona_prompt('researcher_default') == _EXPECTED_RESEARCHER_SIMPLE_PROMPT
    assert get_persona_prompt('chat_summary') == _EXPECTED_CHAT_SUMMARY_PROMPT


def test_compose_prompt_mode_only_matches_legacy_composer() -> None:
    assert compose_prompt(mode_id='researcher_rag', chat_mode='assistant') == compose_persona_prompt(
        persona_id='researcher_rag',
        chat_mode='assistant',
    )
    assert compose_prompt(mode_id='researcher_rag', chat_mode='researcher') == compose_persona_prompt(
        persona_id='researcher_rag',
        chat_mode='researcher',
    )


def test_role_registry_contains_builtin_roles() -> None:
    assert 'legal_us' in ROLE_REGISTRY
    assert 'security_compliance' in ROLE_REGISTRY
    assert 'financial_analyst' in ROLE_REGISTRY
    assert 'technical_reviewer' in ROLE_REGISTRY


def test_system_prompt_exports_match_golden_baseline_exactly() -> None:
    assert SIMPLE_ASSISTANT_SYSTEM_PROMPT == _EXPECTED_ASSISTANT_PROMPT
    assert SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT == _EXPECTED_ASSISTANT_WEB_SYNTHESIS_PROMPT
    assert SIMPLE_RESEARCHER_SYSTEM_PROMPT == _EXPECTED_RESEARCHER_SIMPLE_PROMPT
    assert SIMPLE_CHAT_SUMMARY_SYSTEM_PROMPT == _EXPECTED_CHAT_SUMMARY_PROMPT
