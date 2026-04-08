from tools.diagnostics.evaluate import _apply_closeout_contract_for_diagnostics


def test_diagnostics_closeout_contract_noop_without_query_item() -> None:
    answer = 'Plain response'
    enforced = _apply_closeout_contract_for_diagnostics(
        question='Summarize this.',
        display_answer=answer,
        query_item=None,
    )
    assert enforced == answer


def test_diagnostics_closeout_contract_applies_required_heading_repair() -> None:
    enforced = _apply_closeout_contract_for_diagnostics(
        question='Use headings in exact order: ## Scope, ## Findings.',
        display_answer='## Scope\n- ok',
        query_item={'id': 'q1'},
    )
    assert '## Scope' in enforced
    assert '## Findings' in enforced


def test_diagnostics_closeout_contract_redacts_ssn() -> None:
    enforced = _apply_closeout_contract_for_diagnostics(
        question='Summarize records.',
        display_answer='SSN: 123-45-6789',
        query_item={'id': 'q2'},
    )
    assert '[REDACTED-SSN]' in enforced
    assert '123-45-6789' not in enforced
