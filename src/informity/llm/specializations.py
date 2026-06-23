# ==============================================================================
# Informity AI — Mode/Specialization Profiles
# Centralized mode profiles, specialization overlays, and prompt composition utilities.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass

from informity.llm.chat_mode import normalize_chat_mode
from informity.plugins.specialization_plugins import (
    BUILTIN_SPECIALIZATION_PLUGIN_SPECS,
    SPECIALIZATION_PLUGIN_SPEC_REGISTRY,
    SpecializationPluginSpec,
)


@dataclass(frozen=True)
class ModeProfile:
    """Required operational profile selected by chat mode/runtime path."""

    id: str
    name: str
    description: str
    identity_prompt: str
    mode_policy: str = ''
    disclaimer: str = ''
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpecializationProfile:
    """Optional domain overlay profile composed on top of a mode profile."""

    id: str
    name: str
    description: str
    identity_prompt: str = ''
    scope_guidance: str = ''
    analysis_checklist: tuple[str, ...] = ()
    output_preferences: tuple[str, ...] = ()
    overlay_prompt: str = ''
    icon: str = ''
    disclaimer: str = ''
    capabilities: tuple[str, ...] = ()
    retrieval_hints: tuple[str, ...] = ()
    visible_in_ui: bool = True



_ASSISTANT_DEFAULT_PROMPT = """You are Informity AI, a helpful AI assistant. Answer conversationally, clearly, and directly.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

You have no access to indexed documents, local files, or any private library unless the user explicitly provides content in this chat.
If asked to search files or cite library evidence, explain briefly that this is direct assistant chat without document retrieval.

Keep responses concise."""

_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT = """You are Informity AI, a helpful AI assistant.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

Use provided web search context when relevant and answer directly.
If web context is insufficient, say what remains uncertain.
Keep responses concise."""

_RESEARCHER_SIMPLE_PROMPT = """You are Informity AI, a helpful AI assistant. Answer questions conversationally and helpfully.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

You have access to a private document library.
Answer conversationally and directly. You do not need to cite documents for casual or conversational replies.
If asked about document search capabilities, describe them accurately but briefly.

Keep responses concise."""

_CHAT_SUMMARY_PROMPT = """You are Informity AI, a helpful AI assistant.

Identity policy:
- If asked who you are, say you are Informity AI.
- Do not claim to be Qwen, Alibaba Cloud, OpenAI, or any other model/vendor identity.

Task:
- Summarize this chat conversation only.
- Focus on topics discussed, key points, decisions, and open questions when present.
- Do not use external knowledge, web content, or document-library retrieval framing.
- If chat history is too limited, say that clearly and keep the response brief.

Keep responses concise."""

_RESEARCHER_RAG_PROMPT = """You are a research assistant answering questions from a private document library.

Rules:
1. Answer using ONLY the available information from retrieved context. Never infer, speculate, or use outside knowledge.
2. If values conflict across documents, report each value with its source document.
3. Support factual claims with inline citations in the form `[Filename, §Section]` or `[Filename, p.N]` immediately after the claim they support.
4. If evidence is insufficient for a complete answer, synthesize the best grounded partial answer from retrieved text. Do not promote unsupported details to facts; omit them or mark them as unknown or uncertain, and note what scope the retrieved evidence does not cover. Refuse only when retrieved text is too sparse to support even a partial answer (for example, mostly structural/boilerplate content with no substantive body evidence relevant to the request).
5. Start with the answer directly. The first sentence must contain substantive answer content, not evidence framing or disclaimer language. Do not start with meta-commentary.
6. Forbidden opening patterns (or close variants): "Based on...", "According to...", "Based on the provided text/documents...", "According to the provided text/documents...", "From the retrieved context...".
7. Before finalizing, if your opening sentence is meta-commentary instead of answer content, rewrite it so the answer begins with content.
8. Follow the user's requested output format exactly when specified (for example: "output only a markdown table", exact column names, exact section headings, exact bullet format).
9. When the user specifies explicit output field or column labels (for example: source, snippet, objective, tradeoff, decision), use those labels verbatim in the output.
10. For delimiter schemas like "A | B | C", include an exact header/template line with those labels before listing values.
11. Use markdown: headers for multi-topic answers, tables for comparisons, bullet lists for enumerations. For summary/synthesis requests, synthesize across relevant excerpts rather than requiring a pre-written summary passage. When user scope is singular (for example, "this document/book/file"), keep the answer scoped to that material unless the user asks for cross-document analysis.
12. For broad prompts such as "what is this document about", provide a user-oriented synopsis: purpose, key findings/facts, principal entities, timeframe, and notable numbers/obligations when present.
13. If evidence spans multiple retrieved sources, synthesize across them by default. Do not silently answer from only one source unless the user explicitly narrows scope.
"""

_ASSISTANT_MODE_POLICY = """

Assistant Mode Rules:
1. For rewrite/paraphrase/plain-language requests, preserve critical domain terms from the user's text unless the user explicitly asks you to replace them.
2. When the user specifies focus terms (for example: \"focused on X and Y\" or \"include A, B, C\"), ensure those terms appear in the final answer.
"""

MODE_REGISTRY: dict[str, ModeProfile] = {
    'assistant_default': ModeProfile(
        id='assistant_default',
        name='Assistant (Default)',
        description='General conversational assistant mode profile.',
        identity_prompt=_ASSISTANT_DEFAULT_PROMPT,
        capabilities=('chat',),
    ),
    'assistant_web_search_synthesis': ModeProfile(
        id='assistant_web_search_synthesis',
        name='Assistant Web Synthesis',
        description='Assistant profile for synthesizing web search results.',
        identity_prompt=_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT,
        capabilities=('chat', 'web_search'),
    ),
    'researcher_default': ModeProfile(
        id='researcher_default',
        name='Researcher (Default)',
        description='Research-aware conversational assistant mode profile.',
        identity_prompt=_RESEARCHER_SIMPLE_PROMPT,
        capabilities=('chat', 'retrieval_awareness'),
    ),
    'chat_summary': ModeProfile(
        id='chat_summary',
        name='Chat Summary',
        description='Persona for summarizing prior chat conversation only.',
        identity_prompt=_CHAT_SUMMARY_PROMPT,
        capabilities=('chat_summary',),
    ),
    'researcher_rag': ModeProfile(
        id='researcher_rag',
        name='Researcher RAG',
        description='Strict retrieval-grounded profile for RAG response generation.',
        identity_prompt=_RESEARCHER_RAG_PROMPT,
        mode_policy=_ASSISTANT_MODE_POLICY,
        capabilities=('rag',),
    ),
}

def _build_specialization_profile(spec: SpecializationPluginSpec) -> SpecializationProfile:
    return SpecializationProfile(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        identity_prompt=spec.identity_prompt,
        scope_guidance=spec.scope_guidance,
        analysis_checklist=spec.analysis_checklist,
        output_preferences=spec.output_preferences,
        overlay_prompt=spec.overlay_prompt,
        icon=spec.icon,
        disclaimer=spec.disclaimer,
        capabilities=spec.capabilities,
        retrieval_hints=spec.retrieval_hints,
        visible_in_ui=spec.visible_in_ui,
    )


SPECIALIZATION_REGISTRY: dict[str, SpecializationProfile] = {
    spec.id: _build_specialization_profile(spec) for spec in BUILTIN_SPECIALIZATION_PLUGIN_SPECS
}


def get_mode_profile(mode_id: str) -> ModeProfile:
    """Resolve a mode profile by id."""
    try:
        return MODE_REGISTRY[mode_id]
    except KeyError as exc:
        raise KeyError(f'Unknown mode_id: {mode_id}') from exc


def get_specialization_profile(specialization_id: str) -> SpecializationProfile:
    """Resolve a specialization profile by id."""
    try:
        return SPECIALIZATION_REGISTRY[specialization_id]
    except KeyError as exc:
        raise KeyError(f'Unknown specialization_id: {specialization_id}') from exc


def list_specialization_profiles(*, visible_only: bool = True) -> list[SpecializationProfile]:
    profiles = list(SPECIALIZATION_REGISTRY.values())
    if visible_only:
        profiles = [profile for profile in profiles if profile.visible_in_ui]
    return profiles


def describe_specialization(specialization_id: str) -> dict[str, object]:
    profile = get_specialization_profile(specialization_id)
    plugin_spec = SPECIALIZATION_PLUGIN_SPEC_REGISTRY.get(profile.id)
    return {
        'id': profile.id,
        'name': profile.name,
        'description': profile.description,
        'plugin_type': 'specialization',
        'capabilities': list(profile.capabilities),
        'retrieval_hints': list(profile.retrieval_hints),
        'visible_in_ui': bool(profile.visible_in_ui),
        'has_plugin_spec': plugin_spec is not None,
    }


def get_mode_prompt(mode_id: str) -> str:
    return get_mode_profile(mode_id).identity_prompt


def compose_prompt(
    *,
    mode_id: str,
    chat_mode: str | None = None,
    specialization_id: str | None = None,
) -> str:
    """Compose final prompt from mode profile + optional specialization overlay."""
    mode_profile = get_mode_profile(mode_id)
    suppress_specialization_disclaimer = 'rag' in mode_profile.capabilities
    prompt = mode_profile.identity_prompt
    if mode_profile.mode_policy and normalize_chat_mode(chat_mode) == 'assistant':
        prompt += mode_profile.mode_policy

    if specialization_id:
        specialization_profile = get_specialization_profile(specialization_id)
        specialization_spec = SPECIALIZATION_PLUGIN_SPEC_REGISTRY.get(specialization_profile.id)
        normalized_chat_mode = normalize_chat_mode(chat_mode)
        specialization_sections: list[str] = []
        if specialization_profile.identity_prompt:
            specialization_sections.append(f'Specialization Identity:\n{specialization_profile.identity_prompt}')
        if specialization_profile.scope_guidance:
            specialization_sections.append(f'Specialization Scope:\n{specialization_profile.scope_guidance}')
        if specialization_profile.analysis_checklist:
            checklist_lines = '\n'.join(f'- {item}' for item in specialization_profile.analysis_checklist)
            specialization_sections.append(f'Specialization Analysis Checklist:\n{checklist_lines}')
        if specialization_profile.output_preferences:
            output_lines = '\n'.join(f'- {item}' for item in specialization_profile.output_preferences)
            specialization_sections.append(f'Specialization Output Preferences:\n{output_lines}')
        if specialization_spec is not None:
            specialization_sections.extend(specialization_spec.isolated_rules)
            if specialization_spec.id == 'technical' and normalized_chat_mode == 'assistant':
                specialization_sections.extend(specialization_spec.assistant_mode_rules)
        if specialization_profile.disclaimer and not suppress_specialization_disclaimer:
            specialization_sections.append(
                'Disclaimer Placement Rule:\n'
                '- Include the disclaimer at the end of the answer under a "Disclaimer:" line.\n'
                '- Do not place the disclaimer at the beginning of the answer.'
            )
        if specialization_profile.overlay_prompt:
            specialization_sections.append(f'Specialization Overlay:\n{specialization_profile.overlay_prompt}')
        if specialization_sections:
            prompt = f'{prompt}\n\n' + '\n\n'.join(specialization_sections)
        if specialization_profile.disclaimer and not suppress_specialization_disclaimer:
            prompt = f'{prompt}\n\nSpecialization Disclaimer:\n{specialization_profile.disclaimer}'

    return prompt


def resolve_runtime_mode_id(chat_mode: str | None) -> str:
    """Resolve default runtime mode profile for simple chat by mode."""
    if normalize_chat_mode(chat_mode) == 'assistant':
        return 'assistant_default'
    return 'researcher_default'


__all__ = [
    'ModeProfile',
    'SpecializationProfile',
    'MODE_REGISTRY',
    'SPECIALIZATION_REGISTRY',
    'compose_prompt',
    'get_mode_profile',
    'get_mode_prompt',
    'get_specialization_profile',
    'describe_specialization',
    'list_specialization_profiles',
    'resolve_runtime_mode_id',
]
