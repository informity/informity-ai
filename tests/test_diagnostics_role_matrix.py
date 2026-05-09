from tools.diagnostics.evaluate import _validate_query_expectations


def test_validate_query_expectations_required_terms_all_missing() -> None:
    failures = _validate_query_expectations(
        query_item={'required_terms_all': ['liability', 'jurisdiction']},
        sections={},
        summary={},
        answer='This response mentions liability only.',
        sources_count=0,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert 'missing_required_term: jurisdiction' in failures


def test_validate_query_expectations_required_terms_any_missing() -> None:
    failures = _validate_query_expectations(
        query_item={'required_terms_any': ['budget', 'forecast']},
        sections={},
        summary={},
        answer='This response focuses on architecture and dependencies.',
        sources_count=0,
        unsupported_claim_count=0,
        evidence_coverage_rate=0.0,
        not_found_count=0,
    )
    assert 'missing_required_any_term: [budget, forecast]' in failures
