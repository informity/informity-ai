from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_patterns import (
    evaluate_substantive_evidence,
    extract_explicit_title_reference,
    has_explicit_title_reference,
    has_topic_overlap_with_previous_user,
    has_topic_shift_cue,
    is_plot_or_chapter_request,
    is_summary_style_request,
    resolve_followup_scope_anchor_filename,
    should_block_summary_generation_for_structural_only_evidence,
    should_prefer_title_alignment,
)


def test_is_summary_style_request_allows_coverage_and_focused_intents() -> None:
    coverage = QueryClassification(intent='coverage')
    focused = QueryClassification(intent='focused')
    metadata = QueryClassification(intent='metadata')
    assert is_summary_style_request('Summarize this document', coverage) is True
    assert is_summary_style_request('Summarize this document', focused) is True
    assert is_summary_style_request('Summarize this document', metadata) is False


def test_is_summary_style_request_detects_document_about_prompt() -> None:
    focused = QueryClassification(intent='focused')
    assert is_summary_style_request('What is this document about?', focused) is True


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
        question='Summarize this document',
        history=history,
        classification=classification,
    )
    assert resolved == 'anchored.pdf'


def test_has_topic_overlap_with_previous_user_detects_shared_terms() -> None:
    history = [
        ChatMessage(chat_id='chat', role='user', content='List characters in The Three Musketeers'),
        ChatMessage(chat_id='chat', role='assistant', content='Done'),
    ]
    assert has_topic_overlap_with_previous_user(
        question='give character description for each character',
        history=history,
    ) is True


def test_should_prefer_title_alignment_for_compare_prompt() -> None:
    classification = QueryClassification(intent='focused')
    assert should_prefer_title_alignment(
        question="Compare D'Artagnan in The Three Musketeers and Twenty Years After",
        classification=classification,
    ) is True


def test_should_not_prefer_title_alignment_for_generic_prompt() -> None:
    classification = QueryClassification(intent='focused')
    assert should_prefer_title_alignment(
        question='What is the weather today?',
        classification=classification,
    ) is False


def test_should_prefer_title_alignment_when_source_terms_include_title_phrase() -> None:
    classification = QueryClassification(intent='focused', source_terms=['the three musketeers'])
    assert should_prefer_title_alignment(
        question='List all characters in this document',
        classification=classification,
    ) is True


def test_has_explicit_title_reference_detects_prepositional_title_phrase() -> None:
    assert has_explicit_title_reference('What is the general plot of The Three Musketeers book?') is True


def test_has_explicit_title_reference_detects_title_before_document_noun() -> None:
    assert has_explicit_title_reference('What is The Three Musketeers book about?') is True


def test_has_explicit_title_reference_detects_quoted_title_phrase() -> None:
    assert has_explicit_title_reference('Summarize "The Three Musketeers" with key themes.') is True


def test_has_explicit_title_reference_ignores_generic_question() -> None:
    assert has_explicit_title_reference('What is this file about?') is False


def test_extract_explicit_title_reference_returns_normalized_title() -> None:
    title = extract_explicit_title_reference('What is The Three Musketeers book about?')
    assert title == 'The Three Musketeers'


def test_has_topic_shift_cue_excludes_on_another_note_phrase() -> None:
    assert has_topic_shift_cue('On another note, summarize this file.') is False
