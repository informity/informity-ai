# ==============================================================================
# Informity AI — Prompt Builder Tests
# Tests message construction with context chunks and history
# ==============================================================================

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.personas import compose_prompt
from informity.llm.prompt_builder import build_messages


class _StubProfile:
    def __init__(self, *, context_length: int, rag_context_ratio: float) -> None:
        self.context_length = context_length
        self.rag_context_ratio = rag_context_ratio


class TestPromptBuilder:

    def test_builds_system_message_with_context(self) -> None:
        chunks = [
            {
                'chunk_text': 'This is chunk 1 content.',
                'filename': 'file1.txt',
            },
            {
                'chunk_text': 'This is chunk 2 content.',
                'filename': 'file2.txt',
            },
        ]

        messages = build_messages('What is this?', chunks)

        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert 'chunk 1 content' in messages[0]['content']
        assert 'chunk 2 content' in messages[0]['content']
        assert '[Source: 1]' in messages[0]['content']
        assert '[Source: 2]' in messages[0]['content']

    def test_includes_filename_in_source_label(self) -> None:
        chunks = [
            {
                'chunk_text': 'Content',
                'filename': 'test.pdf',
            },
        ]

        messages = build_messages('Question', chunks)

        assert '[Source: 1] test.pdf' in messages[0]['content']

    def test_includes_page_number_if_present(self) -> None:
        chunks = [
            {
                'chunk_text': 'Content',
                'filename': 'test.pdf',
                'page_number': 5,
            },
        ]

        messages = build_messages('Question', chunks)

        assert 'Page 5' in messages[0]['content']

    def test_includes_section_path_if_present(self) -> None:
        chunks = [
            {
                'chunk_text': 'Content',
                'filename': 'test.pdf',
                'section_path': 'Introduction/Overview',
            },
        ]

        messages = build_messages('Question', chunks)

        assert 'Section: Introduction/Overview' in messages[0]['content']

    def test_includes_year_and_category_in_source_label_when_present(self) -> None:
        chunks = [
            {
                'chunk_text': 'Content',
                'filename': 'financials_2023.pdf',
                'year': 2023,
                'category': 'financial',
            },
        ]

        messages = build_messages('Question', chunks)
        assert 'Year: 2023' in messages[0]['content']
        assert 'Category: financial' in messages[0]['content']

    def test_omits_year_and_category_when_not_present(self) -> None:
        chunks = [
            {
                'chunk_text': 'Content',
                'filename': 'notes.txt',
                'category': '',
            },
        ]

        messages = build_messages('Question', chunks)
        assert 'Year:' not in messages[0]['content']
        assert 'Category:' not in messages[0]['content']

    def test_adds_user_question(self) -> None:
        chunks = []
        messages = build_messages('What is AI?', chunks)

        assert messages[-1]['role'] == 'user'
        assert messages[-1]['content'] == 'What is AI?'

    def test_includes_history(self) -> None:
        chunks = []
        history = [
            ChatMessage(chat_id='test', role='user', content='Previous question'),
            ChatMessage(chat_id='test', role='assistant', content='Previous answer'),
        ]

        messages = build_messages('Follow-up question', chunks, history)

        # Should have system, history messages, and current question
        assert len(messages) >= 3
        assert messages[-2]['content'] == 'Previous answer'
        assert messages[-1]['content'] == 'Follow-up question'

    def test_truncates_long_history(self) -> None:
        chunks = []
        history = [
            ChatMessage(chat_id='test', role='user', content=f'Question {i}')
            for i in range(10)
        ]

        messages = build_messages('Current question', chunks, history)

        # Should only include last 5 history messages
        assert len(messages) <= 7  # system + 5 history + current question

    def test_uses_assistant_history_limit_by_mode(self) -> None:
        original_default = settings.chat_history_messages
        original_assistant = settings.chat_history_messages_assistant
        original_researcher = settings.chat_history_messages_researcher
        try:
            settings.chat_history_messages = 1
            settings.chat_history_messages_assistant = 4
            settings.chat_history_messages_researcher = 2
            history = [
                ChatMessage(chat_id='test', role='user', content=f'Question {i}')
                for i in range(10)
            ]
            messages = build_messages('Current question', [], history, chat_mode='assistant')
            # system + 4 history + current question
            assert len(messages) == 6
            assert messages[1]['content'] == 'Question 6'
            assert messages[-2]['content'] == 'Question 9'
        finally:
            settings.chat_history_messages = original_default
            settings.chat_history_messages_assistant = original_assistant
            settings.chat_history_messages_researcher = original_researcher

    def test_uses_researcher_history_limit_by_mode(self) -> None:
        original_default = settings.chat_history_messages
        original_assistant = settings.chat_history_messages_assistant
        original_researcher = settings.chat_history_messages_researcher
        try:
            settings.chat_history_messages = 1
            settings.chat_history_messages_assistant = 4
            settings.chat_history_messages_researcher = 2
            history = [
                ChatMessage(chat_id='test', role='user', content=f'Question {i}')
                for i in range(10)
            ]
            messages = build_messages('Current question', [], history, chat_mode='researcher')
            # system + 2 history + current question
            assert len(messages) == 4
            assert messages[1]['content'] == 'Question 8'
            assert messages[-2]['content'] == 'Question 9'
        finally:
            settings.chat_history_messages = original_default
            settings.chat_history_messages_assistant = original_assistant
            settings.chat_history_messages_researcher = original_researcher

    def test_uses_fallback_history_limit_when_mode_is_unresolved(self) -> None:
        original_default = settings.chat_history_messages
        original_assistant = settings.chat_history_messages_assistant
        original_researcher = settings.chat_history_messages_researcher
        try:
            settings.chat_history_messages = 3
            settings.chat_history_messages_assistant = 8
            settings.chat_history_messages_researcher = 2
            history = [
                ChatMessage(chat_id='test', role='user', content=f'Question {i}')
                for i in range(10)
            ]
            messages = build_messages('Current question', [], history, chat_mode='future-mode')
            # system + fallback(3) history + current question
            assert len(messages) == 5
            assert messages[1]['content'] == 'Question 7'
            assert messages[-2]['content'] == 'Question 9'
        finally:
            settings.chat_history_messages = original_default
            settings.chat_history_messages_assistant = original_assistant
            settings.chat_history_messages_researcher = original_researcher

    def test_token_budget_trim_limits_history_when_profile_is_provided(self) -> None:
        history = [
            ChatMessage(chat_id='test', role='user', content=f'Long message {i} ' + ('x ' * 300))
            for i in range(8)
        ]
        profile = _StubProfile(context_length=1200, rag_context_ratio=0.75)

        messages = build_messages('Current question', [], history, model_profile=profile)

        # Token budget is effectively exhausted (small context + generation reserve),
        # so builder should keep only the latest history message as a floor.
        assert len(messages) == 3
        assert messages[1]['role'] == 'user'
        assert 'Long message 7' in messages[1]['content']
        assert messages[-1]['content'] == 'Current question'

    def test_custom_system_prompt_override(self) -> None:
        messages = build_messages(
            'Question',
            [],
            system_prompt='You are a direct assistant.',
        )
        assert messages[0]['content'].startswith('You are a direct assistant.')

    def test_empty_chunks_still_builds_messages(self) -> None:
        messages = build_messages('Question', [])

        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert messages[-1]['role'] == 'user'
        assert messages[-1]['content'] == 'Question'

    def test_system_prompt_includes_rules(self) -> None:
        messages = build_messages('Question', [])

        assert 'Answer using ONLY' in messages[0]['content']
        assert 'Use markdown:' in messages[0]['content']
        assert 'insufficient for a complete answer' in messages[0]['content']

    def test_system_prompt_includes_context_anchor(self) -> None:
        messages = build_messages('Question', [])
        content = messages[0]['content']

        assert 'Context:' in content

    def test_builder_does_not_include_research_instructions(self) -> None:
        messages = build_messages('Question', [])
        assert 'Research mode instructions:' not in messages[0]['content']
        assert 'Prefer comprehensive, evidence-grounded coverage over brevity.' not in messages[0]['content']

    def test_builder_keeps_single_route_prompt_contract(self) -> None:
        messages = build_messages('Question', [])
        assert 'Research mode instructions:' not in messages[0]['content']

    def test_builder_system_prompt_matches_rag_persona_composer_exactly(self) -> None:
        messages = build_messages('Question', [], chat_mode='researcher')
        expected_system_prefix = compose_prompt(mode_id='researcher_rag', chat_mode='researcher')
        assert messages[0]['content'] == f'{expected_system_prefix}\n\nContext:\n'

    def test_builder_assistant_mode_system_prompt_matches_composer_exactly(self) -> None:
        messages = build_messages('Question', [], chat_mode='assistant')
        expected_system_prefix = compose_prompt(mode_id='researcher_rag', chat_mode='assistant')
        assert messages[0]['content'] == f'{expected_system_prefix}\n\nContext:\n'

    def test_builder_general_role_parity_when_role_absent(self) -> None:
        messages_no_role = build_messages('Question', [], chat_mode='researcher', role_id=None)
        messages_legacy = build_messages('Question', [], chat_mode='researcher')
        assert messages_no_role[0]['content'] == messages_legacy[0]['content']

    def test_builder_applies_role_overlay_when_role_present(self) -> None:
        messages = build_messages('Question', [], chat_mode='researcher', role_id='legal')
        assert 'Role Identity:' in messages[0]['content']
        assert 'Role Disclaimer:' in messages[0]['content']

    def test_preserves_assistant_history_verbatim(self) -> None:
        history = [
            ChatMessage(chat_id='test', role='user', content='Prior question'),
            ChatMessage(chat_id='test', role='assistant', content='<think>hidden chain</think>\nVisible answer'),
        ]

        messages = build_messages('Follow-up', [], history)

        assert messages[-2]['role'] == 'assistant'
        assert messages[-2]['content'] == '<think>hidden chain</think>\nVisible answer'

    def test_keeps_assistant_history_when_reasoning_only(self) -> None:
        history = [
            ChatMessage(chat_id='test', role='assistant', content='<think>reasoning only</think>'),
        ]

        messages = build_messages('Follow-up', [], history)

        # system + assistant history + current user question
        assert len(messages) == 3
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'assistant'
        assert messages[2]['role'] == 'user'

    def test_includes_output_contract_constraints_when_provided(self) -> None:
        messages = build_messages(
            'Question',
            [],
            output_constraints={'max_words': 180, 'exact_top_level_bullets': 5},
            format_requirements=['use heading order exactly', 'include heading: ## Scope'],
        )
        system_content = messages[0]['content']

        assert 'Output Contract:' in system_content
        assert '- Maximum words: 180' in system_content
        assert '- Exactly 5 top-level bullets when bullets are requested' in system_content
        assert '- use heading order exactly' in system_content
        assert '- include heading: ## Scope' in system_content
