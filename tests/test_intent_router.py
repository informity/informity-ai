from __future__ import annotations

from informity.llm.intent_router import _INTENT_SPECS


def test_intent_specs_shape_and_coverage() -> None:
    intents = [spec.intent for spec in _INTENT_SPECS]
    assert set(intents) == {'metadata', 'simple', 'focused', 'coverage'}
    assert len(intents) == 4

    for spec in _INTENT_SPECS:
        assert spec.description.strip()
        assert 2 <= len(spec.examples) <= 20
        assert len(spec.negatives) >= 1
        assert all(example.strip() for example in spec.examples)
        assert all(negative.strip() for negative in spec.negatives)


def test_intent_specs_are_corpus_agnostic() -> None:
    banned_terms = (
        'glenn',
        'perez',
        'fatca',
        'sample-',
        'sample_data',
        '.pdf',
        '.csv',
        '.xlsx',
        'lender',
        '1099',
        'w-2',
    )
    texts: list[str] = []
    for spec in _INTENT_SPECS:
        texts.append(spec.description)
        texts.extend(spec.examples)
        texts.extend(spec.negatives)
    lower = ' '.join(texts).casefold()
    for term in banned_terms:
        assert term not in lower
