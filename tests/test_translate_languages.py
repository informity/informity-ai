from informity.translate_languages import (
    TRANSLATE_LANGUAGE_OPTIONS,
    normalize_translate_language,
    normalize_translate_language_list,
    search_translate_languages,
)


def test_translate_languages_are_alphabetical_and_restricted() -> None:
    labels = [option.label for option in TRANSLATE_LANGUAGE_OPTIONS]
    assert labels == sorted(labels)
    assert labels == [
        'Arabic (Standard)',
        'Bengali',
        'Chinese',
        'Czech',
        'Dutch',
        'English',
        'French',
        'German',
        'Greek',
        'Hebrew',
        'Hindi',
        'Indonesian',
        'Italian',
        'Japanese',
        'Korean',
        'Persian',
        'Polish',
        'Portuguese',
        'Romanian',
        'Russian',
        'Spanish',
        'Swedish',
        'Thai',
        'Turkish',
        'Ukrainian',
        'Vietnamese',
    ]


def test_translate_language_normalization_accepts_codes_and_aliases() -> None:
    assert normalize_translate_language('de') == 'German'
    assert normalize_translate_language('Chinese (Simplified)') == 'Chinese'
    assert normalize_translate_language('Arabic') == 'Arabic (Standard)'
    assert normalize_translate_language('') == 'Spanish'
    assert normalize_translate_language('Unknown Language') == 'Spanish'
    assert normalize_translate_language_list(['de', 'German', 'fr', 'German']) == ['German', 'French']


def test_translate_language_search_is_restricted_to_catalog() -> None:
    labels = [option.label for option in search_translate_languages('span')]
    assert labels == ['Spanish']
    labels = [option.label for option in search_translate_languages('chi')]
    assert labels == ['Chinese']
    assert all(label in {option.label for option in TRANSLATE_LANGUAGE_OPTIONS} for label in labels)
