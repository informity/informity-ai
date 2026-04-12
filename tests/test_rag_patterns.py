from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_patterns import (
    evaluate_substantive_evidence,
    is_plot_or_chapter_request,
    is_summary_style_request,
    resolve_followup_scope_anchor_filename,
    should_block_summary_generation_for_structural_only_evidence,
)


def test_is_summary_style_request_allows_coverage_and_focused_intents() -> None:
    coverage = QueryClassification(intent='coverage')
    focused = QueryClassification(intent='focused')
    metadata = QueryClassification(intent='metadata')
    assert is_summary_style_request('Summarize this document', coverage) is True
    assert is_summary_style_request('Summarize this document', focused) is True
    assert is_summary_style_request('Summarize this document', metadata) is False


def test_is_plot_or_chapter_request_detects_plot_and_chapter() -> None:
    assert is_plot_or_chapter_request('What is the plot?') is True
    assert is_plot_or_chapter_request('Summarize chapter 1') is True
    assert is_plot_or_chapter_request('Give me key points') is False


def test_evaluate_substantive_evidence_profiles_structural_only() -> None:
    profile = evaluate_substantive_evidence(
        [
            {'block_type': 'table'},
            {'block_type': 'form'},
        ]
    )
    assert profile['chunk_count'] == 2
    assert profile['structural_count'] == 2
    assert profile['substantive_count'] == 0
    assert profile['substantive_ratio'] == 0.0


def test_should_block_summary_generation_for_structural_only_evidence() -> None:
    classification = QueryClassification(intent='coverage')
    evidence_profile = {
        'chunk_count': 2,
        'structural_count': 2,
        'substantive_count': 0,
    }
    assert should_block_summary_generation_for_structural_only_evidence(
        question='Summarize key points',
        classification=classification,
        evidence_profile=evidence_profile,
    ) is True


def test_should_not_block_plot_request_structural_only_evidence() -> None:
    classification = QueryClassification(intent='coverage')
    evidence_profile = {
        'chunk_count': 2,
        'structural_count': 2,
        'substantive_count': 0,
    }
    assert should_block_summary_generation_for_structural_only_evidence(
        question='What is the plot?',
        classification=classification,
        evidence_profile=evidence_profile,
    ) is False


def test_resolve_followup_scope_anchor_filename_from_history() -> None:
    classification = QueryClassification(intent='focused')
    history = [
        ChatMessage(
            chat_id='chat',
            role='assistant',
            content='Prior answer',
            sources=[{'filename': 'anchored.pdf', 'path': '/docs/anchored.pdf'}],
        )
    ]
    resolved = resolve_followup_scope_anchor_filename(
        question='Summarize this book',
        history=history,
        classification=classification,
    )
    assert resolved == 'anchored.pdf'
