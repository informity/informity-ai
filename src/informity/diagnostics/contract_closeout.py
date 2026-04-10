# ==============================================================================
# Informity AI — Diagnostics Contract Closeout
# Shared helper for diagnostics-only closeout contract enforcement.
# ==============================================================================

from informity.llm.contract_gate import build_contract_spec, enforce_required_sections


def apply_closeout_contract_for_diagnostics(
    *,
    question: str,
    display_answer: str,
    query_item: dict | None,
) -> str:
    if not display_answer or not isinstance(query_item, dict):
        return display_answer
    contract_spec = build_contract_spec(question=question, classification=None)
    enforced_answer, _missing_sections_filled = enforce_required_sections(
        answer=display_answer,
        spec=contract_spec,
    )
    return enforced_answer
