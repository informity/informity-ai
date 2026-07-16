"""Test module for tests test rag runtime modules."""

# pylint: disable=line-too-long

from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.rag_runtime.citation_verification import (
    assess_answer_support,
    filter_verified_sources,
)
from informity.llm.rag_runtime.structured_numeric import (
    _derive_format_requirements,
)


def test_structured_numeric_derives_heading_order_requirement() -> None:
    """Test structured numeric derives heading order requirement."""
    requirements = _derive_format_requirements(
        "Create a brief with sections in order: 1) Scope, 2) Method, 3) Findings."
    )
    assert any("requested order" in requirement for requirement in requirements)
    assert any("include heading: Scope" in requirement for requirement in requirements)


def test_structured_numeric_derives_numbered_headings_with_parenthetical_commas() -> None:
    """Test structured numeric derives numbered headings with parenthetical commas."""
    requirements = _derive_format_requirements(
        "Create a brief with sections in order: 1) Executive Summary (max 140 words), "
        "2) Year-by-Year Evidence Map (2022, 2023, 2024), 3) Action Checklist."
    )
    assert any(
        "include heading: Executive Summary (max 140 words)" in requirement
        for requirement in requirements
    )
    assert any(
        "include heading: Year-by-Year Evidence Map (2022, 2023, 2024)" in requirement
        for requirement in requirements
    )
    assert any("include heading: Action Checklist" in requirement for requirement in requirements)


def test_structured_numeric_derives_include_clause_required_terms() -> None:
    """Test structured numeric derives include clause required terms."""
    requirements = _derive_format_requirements(
        "Compare records from different years and include conflict statement, involved documents, "
        "conflicting values, and likely reason grounded in evidence."
    )
    assert "include term: conflict" in requirements
    assert "include term: documents" in requirements
    assert "include term: values" in requirements
    assert "include term: reason" in requirements
    assert "include term: evidence" in requirements


def test_structured_numeric_derives_cover_clause_required_terms() -> None:
    """Test structured numeric derives cover clause required terms."""
    requirements = _derive_format_requirements(
        "Summarize cross-year changes. Cover biggest increase, biggest decrease, and ambiguous deltas, with evidence."
    )
    assert "include term: increase" in requirements
    assert "include term: decrease" in requirements
    assert "include term: evidence" in requirements


def test_structured_numeric_derives_required_markdown_table_columns() -> None:
    """Test structured numeric derives required markdown table columns."""
    requirements = _derive_format_requirements(
        "Under ## Evidence Coverage, include exactly one markdown table with columns: "
        "Group, Years Covered, Key Evidence, Confidence."
    )
    assert any(
        requirement
        == "include markdown table columns: Group | Years Covered | Key Evidence | Confidence"
        for requirement in requirements
    )


def test_structured_numeric_uses_action_hints_for_enumeration() -> None:
    """Test structured numeric uses action hints for enumeration."""
    requirements = _derive_format_requirements(
        "Summarize key findings.",
        action_hints={"should_enumerate": True},
    )
    assert any("numbered or bulleted list" in requirement for requirement in requirements)


def test_structured_numeric_uses_action_hints_for_comparison() -> None:
    """Test structured numeric uses action hints for comparison."""
    requirements = _derive_format_requirements(
        "Summarize key findings.",
        action_hints={"should_compare": True},
    )
    assert any(
        "side-by-side or structured comparison format" in requirement
        for requirement in requirements
    )


def test_structured_numeric_action_hints_do_not_duplicate_existing_requirements() -> None:
    """Test structured numeric action hints do not duplicate existing requirements."""
    requirements = _derive_format_requirements(
        "Provide findings by year and compare key changes across all indexed records.",
        action_hints={"should_compare": True},
    )
    comparison_requirements = [
        requirement
        for requirement in requirements
        if "side-by-side or structured comparison format" in requirement
    ]
    assert len(comparison_requirements) == 1


def test_generation_runtime_has_remaining_scope_false_for_terminal_timeout() -> None:
    """Test generation runtime has remaining scope false for terminal timeout."""
    assert (
        _generation_runtime._has_remaining_scope(
            timeout_reason="queue_wait_timeout",
            stream_recovery_reason=None,
            generation_skipped=False,
            applied_degradations=[],
        )
        is False
    )


def test_generation_closeout_source_references_filter_to_used_chunks() -> None:
    """Test generation closeout source references filter to used chunks."""
    chunks = [
        {
            "filename": "tax_2024.pdf",
            "file_path": "/docs/tax_2024.pdf",
            "chunk_text": "Property tax receipt shows total paid 2024 county bill.",
            "score": 0.81,
        },
        {
            "filename": "bank.pdf",
            "file_path": "/docs/bank.pdf",
            "chunk_text": "Checking account transfer history and unrelated debit card rows.",
            "score": 0.74,
        },
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text="The property tax receipt confirms total paid for 2024.",
        truncate_preview_fn=_identity_text,
        normalize_relevance_score_fn=_float_score,
    )
    assert len(sources) == 1
    assert sources[0].filename == "tax_2024.pdf"


def test_generation_closeout_source_references_fallback_to_top_when_no_overlap() -> None:
    """Test generation closeout source references fallback to top when no overlap."""
    chunks = [
        {
            "filename": f"doc_{idx}.pdf",
            "file_path": f"/docs/doc_{idx}.pdf",
            "chunk_text": f"Chunk text {idx} with archive metadata and unrelated content.",
            "score": 0.9 - idx * 0.01,
        }
        for idx in range(7)
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text="This final answer discusses topics absent from retrieved chunks.",
        truncate_preview_fn=_identity_text,
        normalize_relevance_score_fn=_float_score,
    )
    assert sources == []


def test_generation_closeout_source_references_keep_all_when_answer_empty() -> None:
    """Test generation closeout source references keep all when answer empty."""
    chunks = [
        {
            "filename": "a.pdf",
            "file_path": "/docs/a.pdf",
            "chunk_text": "Alpha chunk",
            "score": 0.5,
        },
        {
            "filename": "b.pdf",
            "file_path": "/docs/b.pdf",
            "chunk_text": "Beta chunk",
            "score": 0.4,
        },
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text="",
        truncate_preview_fn=_identity_text,
        normalize_relevance_score_fn=_float_score,
    )
    assert len(sources) == 2
    assert {source.filename for source in sources} == {"a.pdf", "b.pdf"}


def test_generation_closeout_fail_closed_when_assessment_finds_thin_evidence() -> None:
    """Test generation closeout fail closed when assessment finds thin evidence."""
    chunks = [
        {
            "filename": "a.pdf",
            "file_path": "/docs/a.pdf",
            "chunk_text": "Alpha chunk with unrelated content.",
            "score": 0.6,
        },
        {
            "filename": "b.pdf",
            "file_path": "/docs/b.pdf",
            "chunk_text": "Beta chunk with unrelated content.",
            "score": 0.5,
        },
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text="This answer uses terms that do not appear in the chunks.",
        truncate_preview_fn=_identity_text,
        normalize_relevance_score_fn=_float_score,
    )
    assert sources == []


def test_citation_verification_support_assessment_flags_thin_evidence() -> None:
    """Test citation verification support assessment flags thin evidence."""
    result = assess_answer_support(
        answer_text="The document says the balance is 123 and confirms the owner.",
        source_texts=["A completely unrelated passage about other topics."],
    )
    assert result.evaluated_claim_count >= 1
    assert result.supported_claim_count == 0
    assert result.should_fail_closed is True


def test_filter_verified_sources_keeps_only_matching_sources() -> None:
    """Test filter verified sources keeps only matching sources."""
    sources = [
        _generation_closeout.ChatSourceReference(
            filename="alpha.pdf",
            path="/docs/alpha.pdf",
            chunk_preview="Alpha balance is 123 and due to seller.",
            relevance_score=0.9,
            file_id=1,
        ),
        _generation_closeout.ChatSourceReference(
            filename="beta.pdf",
            path="/docs/beta.pdf",
            chunk_preview="Beta unrelated metadata and boilerplate.",
            relevance_score=0.8,
            file_id=2,
        ),
    ]
    verified = filter_verified_sources(
        sources,
        answer_text="The balance is 123 and due to seller.",
    )
    assert [source.filename for source in verified] == ["alpha.pdf"]
