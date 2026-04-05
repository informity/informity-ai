from informity.diagnostics.issue_types import IssueType
from informity.diagnostics.observer import EvalMetrics, detect_issues, estimate_evidence_metrics


def test_detect_issues_skips_insufficient_retrieval_for_filename_anchored_focus_with_sources() -> None:
    metrics = EvalMetrics(
        chat_id='c1',
        question='What information is in sample-payment-confirmation.pdf?',
        model_filename='model.gguf',
        query_type='focused',
        raw_chunks_count=2,
        sources_count=2,
        generation_seconds=1.2,
        answer_length=180,
        timeout_occurred=False,
        has_empty_answer=False,
        has_refusal_pattern=False,
    )

    issues = detect_issues('answer', metrics)
    assert IssueType.insufficient_retrieval not in issues


def test_detect_issues_keeps_insufficient_retrieval_for_non_filename_complex_focus_query() -> None:
    metrics = EvalMetrics(
        chat_id='c2',
        question='Please explain all financial implications in detail for this scenario carefully',
        model_filename='model.gguf',
        query_type='focused',
        raw_chunks_count=2,
        sources_count=1,
        generation_seconds=1.2,
        answer_length=180,
        timeout_occurred=False,
        has_empty_answer=False,
        has_refusal_pattern=False,
    )

    issues = detect_issues('answer', metrics)
    assert IssueType.insufficient_retrieval in issues


def test_estimate_evidence_metrics_scores_supported_numeric_claims() -> None:
    answer = (
        '- conflict statement: Balance differs by $1,250 in 2023.\n'
        '- involved documents: Tax Report 2023 and Ledger 2023.\n'
    )
    sources = [
        'Tax Report 2023 shows ending balance $12,500.',
        'Ledger 2023 shows ending balance $11,250.',
    ]

    unsupported_claim_count, evidence_coverage_rate, not_found_count = estimate_evidence_metrics(
        answer=answer,
        source_texts=sources,
    )

    assert unsupported_claim_count == 0
    assert evidence_coverage_rate >= 0.75
    assert not_found_count == 0


def test_estimate_evidence_metrics_handles_no_sources() -> None:
    unsupported_claim_count, evidence_coverage_rate, not_found_count = estimate_evidence_metrics(
        answer='Balance is $1,000 and amount increased in 2024.',
        source_texts=[],
    )

    assert unsupported_claim_count >= 1
    assert evidence_coverage_rate == 0.0
    assert not_found_count == 0


def test_estimate_evidence_metrics_skips_non_numeric_likely_reason_claims() -> None:
    answer = (
        "- conflict statement: 2022 and 2023 totals differ.\n"
        "- likely reason: one file appears to include deferred entries not present in the ledger.\n"
    )
    sources = [
        "2022 totals differ from 2023 totals across ledger snapshots.",
        "Deferred entries are not present in the ledger export.",
    ]

    unsupported_claim_count, evidence_coverage_rate, _ = estimate_evidence_metrics(
        answer=answer,
        source_texts=sources,
    )

    assert unsupported_claim_count == 0
    assert evidence_coverage_rate >= 1.0
