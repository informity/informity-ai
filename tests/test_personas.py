from informity.llm.personas import (
    PERSONA_REGISTRY,
    compose_persona_prompt,
    get_persona_prompt,
    resolve_runtime_persona_id,
)


def test_registry_contains_core_default_personas() -> None:
    assert 'assistant_default' in PERSONA_REGISTRY
    assert 'researcher_default' in PERSONA_REGISTRY
    assert 'researcher_rag' in PERSONA_REGISTRY


def test_runtime_persona_resolution_by_mode() -> None:
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
    assert get_persona_prompt('assistant_default').startswith('You are Informity AI')
    assert 'Summarize this chat conversation only.' in get_persona_prompt('chat_summary')
