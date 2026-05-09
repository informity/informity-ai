from informity.api.role_evidence_fallback import apply_role_evidence_fallback


def test_role_evidence_fallback_applies_for_specialized_researcher_role_when_evidence_is_weak() -> None:
    answer = (
        "This is definitively SOC 2 compliant with complete controls and no gaps. "
        "It also fully satisfies GDPR obligations."
    )
    source_items = [
        {'filename': 'policy.txt', 'chunk_preview': 'General governance principles and high-level policy scope.'},
        {'filename': 'notes.txt', 'chunk_preview': 'Operational notes without explicit compliance attestations.'},
    ]

    finalized, unsupported_claim_count, evidence_coverage_rate, _not_found_count, applied = (
        apply_role_evidence_fallback(
            answer=answer,
            source_items=source_items,
            chat_mode='researcher',
            role_id='security_compliance',
        )
    )

    assert applied is True
    assert 'Evidence from Retrieved Context' in finalized
    assert 'Uncertainty / Missing Evidence' in finalized
    assert unsupported_claim_count <= 2
    assert evidence_coverage_rate >= 0.45


def test_role_evidence_fallback_does_not_apply_for_general_role() -> None:
    answer = 'Concise summary.'
    source_items = [{'filename': 'doc.txt', 'chunk_preview': 'Concise summary.'}]

    finalized, *_rest, applied = apply_role_evidence_fallback(
        answer=answer,
        source_items=source_items,
        chat_mode='researcher',
        role_id=None,
    )

    assert applied is False
    assert finalized == answer

