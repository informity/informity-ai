from informity.diagnostics.grounding_verifier import run_grounding_verifier


def test_grounding_verifier_flags_unsupported_number_and_phrase() -> None:
    result = run_grounding_verifier(
        question='Provide an evidence-grounded answer.',
        answer='- Claim amount is $9,999. Evidence: ledger.md, page 2\n- Some other claim.',
        sources=[{'chunk_preview': 'Ledger shows $1,200 in 2024.'}],
    )

    assert result.get('required') is True
    assert result.get('passed') is False
    assert int(result.get('unsupported_claim_count') or 0) >= 1
    claims = [str(item) for item in result.get('unsupported_claims', [])]
    assert any('9999' in claim for claim in claims)


def test_grounding_verifier_computes_evidence_coverage_and_not_found() -> None:
    result = run_grounding_verifier(
        question='Summarize findings.',
        answer='- Item A. Evidence: file-a.md, page 3\n- Item B.\nMissing Evidence: 2023 mortgage statement was not found.',
        sources=[{'chunk_preview': 'Item A appears in file-a.md page 3.'}],
    )

    assert result.get('required') is False
    assert float(result.get('evidence_coverage_rate') or 0.0) >= 0.5
    assert int(result.get('not_found_count') or 0) >= 1


def test_grounding_verifier_treats_historical_year_as_year_token() -> None:
    result = run_grounding_verifier(
        question='Provide an evidence-grounded answer.',
        answer='- Historical baseline year is 1776. Evidence: archive.md, page 1',
        sources=[{'chunk_preview': 'Archive mentions baseline year context without numeric details.'}],
    )

    claims = [str(item) for item in result.get('unsupported_claims', [])]
    assert all('1776' not in claim for claim in claims)


def test_grounding_verifier_ignores_markdown_table_source_snippet_numbers() -> None:
    result = run_grounding_verifier(
        question='Provide an evidence-grounded answer.',
        answer=(
            '### Deterministic Structured Extraction\n\n'
            '| Field | Value | Source Snippet |\n'
            '| --- | ---: | --- |\n'
            '| mortgage interest | 5,000 | statement line shows 92131 and reference 24768 |\n'
        ),
        sources=[{'chunk_preview': 'Mortgage statement reports mortgage interest of 5,000.'}],
    )

    claims = [str(item) for item in result.get('unsupported_claims', [])]
    assert '5000' not in claims
    assert '92131' not in claims
    assert '24768' not in claims
    assert result.get('passed') is True


def test_grounding_verifier_ignores_bullet_source_snippet_numbers() -> None:
    result = run_grounding_verifier(
        question='Provide an evidence-grounded answer.',
        answer='- mortgage interest | 5,000 | statement line shows 92131 and ref 24768',
        sources=[{'chunk_preview': 'Mortgage statement reports mortgage interest of 5,000.'}],
    )

    claims = [str(item) for item in result.get('unsupported_claims', [])]
    assert '92131' not in claims
    assert '24768' not in claims
    assert result.get('passed') is True
