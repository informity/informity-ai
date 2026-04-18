from __future__ import annotations

from informity.api import routes_chat
from informity.db.models import ChatUploadAttachment


def _attachment(
    upload_id: str,
    filename: str,
    *,
    state: str = 'ready',
    file_id: int | None = 1,
) -> ChatUploadAttachment:
    return ChatUploadAttachment(
        upload_id=upload_id,
        chat_id='chat-1',
        file_id=file_id,
        filename_at_upload=filename,
        size_bytes=100,
        state=state,
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


def test_resolve_removed_upload_reference_detects_deleted_file() -> None:
    removed = routes_chat._resolve_removed_upload_reference(
        message_text='Can you compare summary_v1.pdf with the latest version?',
        active_attachments=[_attachment('up-2', 'summary_v2.pdf')],
        deleted_attachments=[_attachment('up-1', 'summary_v1.pdf', state='deleted', file_id=None)],
    )
    assert removed == 'summary_v1.pdf'

