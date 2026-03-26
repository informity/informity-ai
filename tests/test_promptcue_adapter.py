from informity.llm.promptcue_adapter import _map_intent


def test_map_intent_keeps_coverage_for_analysis_in_focused_scope() -> None:
    assert _map_intent('analysis', 'focused') == 'coverage'


def test_map_intent_promotes_lookup_to_coverage_in_broad_scope() -> None:
    assert _map_intent('lookup', 'broad') == 'coverage'
