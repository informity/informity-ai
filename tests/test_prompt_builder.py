# ==============================================================================
# Informity AI — Prompt Builder Tests
# Tests message construction with context chunks and history
# ==============================================================================

from informity.db.models import ChatMessage
from informity.llm.prompt_builder import build_messages


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
        assert 'not contain enough information' in messages[0]['content']

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
