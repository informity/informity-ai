from informity.diagnostics.issue_types import IssueType
from informity.diagnostics.observer import EvalMetrics, detect_issues


def test_detect_issues_skips_insufficient_retrieval_for_filename_anchored_focus_with_sources() -> None:
    metrics = EvalMetrics(
        chat_id='c1',
        question='What information is in 2020 Payment Confirmation - IRS - 4.pdf?',
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
        question='Please explain all retirement tax implications in detail for this scenario',
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
