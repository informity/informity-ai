from __future__ import annotations

import builtins

from informity.config import _get_default_supported_extensions


def test_default_supported_extensions_filter_excluded_extractable_extensions(monkeypatch) -> None:
    from informity.scanner.extractors import base as extractor_base

    monkeypatch.setattr(
        extractor_base,
        'get_all_extractable_extensions',
        lambda: ['.pdf', '.txt', '.json', '.yaml', '.csv', '.toml', '.md'],
    )

    extensions = _get_default_supported_extensions()
    assert '.pdf' in extensions
    assert '.txt' in extensions
    assert '.csv' in extensions
    assert '.md' in extensions
    assert '.json' not in extensions
    assert '.yaml' not in extensions
    assert '.toml' not in extensions


def test_default_supported_extensions_uses_file_types_when_extractor_import_unavailable(monkeypatch) -> None:
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == 'informity.scanner.extractors.base':
            raise ImportError('simulated extractor import failure')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _patched_import)

    extensions = _get_default_supported_extensions()
    assert '.pdf' in extensions
    assert '.txt' in extensions
    assert '.docx' in extensions
    assert '.json' not in extensions
    assert '.yaml' not in extensions
    assert '.yml' not in extensions
    assert '.toml' not in extensions
