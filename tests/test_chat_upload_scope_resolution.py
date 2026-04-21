from __future__ import annotations

from datetime import UTC, datetime

from informity.api.context_scope_manager import resolve_retrieval_context_scope_key
from informity.api import routes_chat
from informity.db.models import ChatMessage, ChatUploadAttachment


def _attachment(
    upload_id: str,
    filename: str,
    *,
    state: str = 'ready',
    file_id: int | None = 1,
    uploaded_at: datetime | None = None,
) -> ChatUploadAttachment:
    return ChatUploadAttachment(
        upload_id=upload_id,
        chat_id='chat-1',
        file_id=file_id,
        filename_at_upload=filename,
        size_bytes=100,
        state=state,
        uploaded_at=uploaded_at,
    )


def test_extract_filename_candidates_reads_dot_tokens_and_quotes() -> None:
    candidates = routes_chat._extract_filename_candidates(
        'Compare "Annual Report 2024.pdf" vs budget_v2.xlsx and ignore plain words.',
    )
    assert 'Annual Report 2024.pdf' in candidates
    assert 'budget_v2.xlsx' in candidates


def test_resolve_upload_scope_exact_case_insensitive_match() -> None:
    attachments = [
        _attachment('up-1', 'Annual Report 2024.pdf'),
        _attachment('up-2', 'Budget.xlsx'),
    ]
    selected, error = routes_chat._resolve_upload_scope_from_filename_candidates(
        candidates=['annual report 2024.pdf'],
        attachments=attachments,
    )
    assert error is None
    assert [item.upload_id for item in selected] == ['up-1']


def test_resolve_upload_scope_unique_partial_match() -> None:
    attachments = [
        _attachment('up-1', 'Quarterly Financials 2025.pdf'),
        _attachment('up-2', 'Budget.xlsx'),
    ]
    selected, error = routes_chat._resolve_upload_scope_from_filename_candidates(
        candidates=['financials 2025'],
        attachments=attachments,
    )
    assert error is None
    assert [item.upload_id for item in selected] == ['up-1']


def test_resolve_upload_scope_ambiguous_duplicate_filename_requires_clarification() -> None:
    attachments = [
        _attachment('up-1', 'summary.pdf'),
        _attachment('up-2', 'summary.pdf'),
    ]
    selected, error = routes_chat._resolve_upload_scope_from_filename_candidates(
        candidates=['summary.pdf'],
        attachments=attachments,
    )
    assert selected == []
    assert error is not None
    assert 'Ambiguous upload reference' in error


def test_build_retrieval_scope_uses_earliest_active_upload_anchor() -> None:
    scope_kind, scope_key = routes_chat._build_retrieval_scope(
        chat_mode='researcher',
        scoped_file_ids=[101, 202],
        upload_attachments=[
            _attachment('up-2', 'b.pdf', uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC)),
            _attachment('up-1', 'a.pdf', uploaded_at=datetime(2026, 4, 19, 4, 12, 20, tzinfo=UTC)),
        ],
    )
    assert scope_kind == 'chat_uploads'
    assert scope_key == 'chat_uploads:up-1'


def test_build_retrieval_scope_keeps_session_anchor_when_original_anchor_deleted() -> None:
    scope_kind, scope_key = routes_chat._build_retrieval_scope(
        chat_mode='researcher',
        scoped_file_ids=None,
        upload_attachments=[
            _attachment('up-2', 'b.pdf', uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC)),
            _attachment('up-3', 'c.pdf', uploaded_at=datetime(2026, 4, 19, 4, 14, 20, tzinfo=UTC)),
        ],
        upload_attachments_all=[
            _attachment(
                'up-1',
                'a.pdf',
                state='deleted',
                uploaded_at=datetime(2026, 4, 19, 4, 12, 20, tzinfo=UTC),
            ).model_copy(update={'removed_at': datetime(2026, 4, 19, 4, 15, 20, tzinfo=UTC)}),
            _attachment('up-2', 'b.pdf', uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC)),
            _attachment('up-3', 'c.pdf', uploaded_at=datetime(2026, 4, 19, 4, 14, 20, tzinfo=UTC)),
        ],
    )
    assert scope_kind == 'chat_uploads'
    assert scope_key == 'chat_uploads:up-1'


def test_build_retrieval_scope_resets_anchor_after_full_depletion() -> None:
    scope_kind, scope_key = routes_chat._build_retrieval_scope(
        chat_mode='researcher',
        scoped_file_ids=None,
        upload_attachments=[
            _attachment('up-4', 'new.pdf', uploaded_at=datetime(2026, 4, 19, 4, 20, 20, tzinfo=UTC)),
        ],
        upload_attachments_all=[
            _attachment(
                'up-1',
                'a.pdf',
                state='deleted',
                uploaded_at=datetime(2026, 4, 19, 4, 12, 20, tzinfo=UTC),
            ).model_copy(update={'removed_at': datetime(2026, 4, 19, 4, 15, 20, tzinfo=UTC)}),
            _attachment(
                'up-2',
                'b.pdf',
                state='deleted',
                uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC),
            ).model_copy(update={'removed_at': datetime(2026, 4, 19, 4, 16, 20, tzinfo=UTC)}),
            _attachment('up-4', 'new.pdf', uploaded_at=datetime(2026, 4, 19, 4, 20, 20, tzinfo=UTC)),
        ],
    )
    assert scope_kind == 'chat_uploads'
    assert scope_key == 'chat_uploads:up-4'


def test_build_retrieval_scope_adds_subset_suffix_for_selected_uploads() -> None:
    scope_kind, scope_key = routes_chat._build_retrieval_scope(
        chat_mode='researcher',
        scoped_file_ids=None,
        upload_attachments=[
            _attachment('up-1', 'a.pdf', uploaded_at=datetime(2026, 4, 19, 4, 12, 20, tzinfo=UTC)),
            _attachment('up-2', 'b.pdf', uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC)),
            _attachment('up-3', 'c.pdf', uploaded_at=datetime(2026, 4, 19, 4, 14, 20, tzinfo=UTC)),
        ],
        upload_attachments_all=[
            _attachment('up-1', 'a.pdf', uploaded_at=datetime(2026, 4, 19, 4, 12, 20, tzinfo=UTC)),
            _attachment('up-2', 'b.pdf', uploaded_at=datetime(2026, 4, 19, 4, 13, 20, tzinfo=UTC)),
            _attachment('up-3', 'c.pdf', uploaded_at=datetime(2026, 4, 19, 4, 14, 20, tzinfo=UTC)),
        ],
        selected_upload_ids=['up-2'],
    )
    assert scope_kind == 'chat_uploads'
    assert scope_key == 'chat_uploads:up-1|sel:up-2'


def test_filter_history_for_scope_isolates_upload_from_indexed_context() -> None:
    history = [
        ChatMessage(
            chat_id='chat-1',
            role='user',
            content='Tell me about uploaded files',
            chat_mode='researcher',
            retrieval_scope_kind='chat_uploads',
            retrieval_scope_key='chat_uploads:up-1',
        ),
        ChatMessage(
            chat_id='chat-1',
            role='assistant',
            content='Upload answer',
            chat_mode='researcher',
            retrieval_scope_kind='chat_uploads',
            retrieval_scope_key='chat_uploads:up-1',
        ),
        ChatMessage(
            chat_id='chat-1',
            role='user',
            content='Now use scanned docs',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:0',
        ),
    ]
    filtered = routes_chat._filter_history_for_scope(
        history=history,
        chat_mode='researcher',
        retrieval_scope_kind='indexed_corpus',
        retrieval_scope_key='indexed_corpus|g:0',
    )
    assert len(filtered) == 1
    assert filtered[0].content == 'Now use scanned docs'


def test_filter_history_for_scope_respects_indexed_generation_boundaries() -> None:
    history = [
        ChatMessage(
            chat_id='chat-1',
            role='user',
            content='Topic A question',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:0',
        ),
        ChatMessage(
            chat_id='chat-1',
            role='assistant',
            content='Topic A answer',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:0',
        ),
        ChatMessage(
            chat_id='chat-1',
            role='user',
            content='Topic B question',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:1',
        ),
    ]
    filtered = routes_chat._filter_history_for_scope(
        history=history,
        chat_mode='researcher',
        retrieval_scope_kind='indexed_corpus',
        retrieval_scope_key='indexed_corpus|g:1',
    )
    assert [item.content for item in filtered] == ['Topic B question']


def test_resolve_retrieval_context_scope_key_resets_on_topic_shift() -> None:
    history = [
        ChatMessage(
            chat_id='chat-1',
            role='assistant',
            content='Prior answer',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:2',
        )
    ]
    resolved_key, meta = resolve_retrieval_context_scope_key(
        chat_mode='researcher',
        retrieval_scope_kind='indexed_corpus',
        retrieval_scope_key='indexed_corpus',
        message_text='OK, new topic: summarize this document',
        history=history,
    )
    assert resolved_key == 'indexed_corpus|g:3'
    assert meta['topic_shift_reset'] is True


def test_resolve_retrieval_context_scope_key_resets_on_upload_to_indexed_transition() -> None:
    history = [
        ChatMessage(
            chat_id='chat-1',
            role='assistant',
            content='Upload answer',
            chat_mode='researcher',
            retrieval_scope_kind='chat_uploads',
            retrieval_scope_key='chat_uploads:up-1',
        ),
        ChatMessage(
            chat_id='chat-1',
            role='assistant',
            content='Prior indexed answer',
            chat_mode='researcher',
            retrieval_scope_kind='indexed_corpus',
            retrieval_scope_key='indexed_corpus|g:1',
        ),
    ]
    resolved_key, meta = resolve_retrieval_context_scope_key(
        chat_mode='researcher',
        retrieval_scope_kind='indexed_corpus',
        retrieval_scope_key='indexed_corpus',
        message_text='What can you tell me about these documents now?',
        history=history[:1],
    )
    assert resolved_key == 'indexed_corpus|g:1'
    assert meta['scope_transition_reset'] is True
