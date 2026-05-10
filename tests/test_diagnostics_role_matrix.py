import importlib.util
from pathlib import Path

_EVALUATE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' / 'evaluate.py'
_SPEC = importlib.util.spec_from_file_location('diagnostics_evaluate_for_tests', _EVALUATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_validate_query_expectations = _MODULE._validate_query_expectations


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
