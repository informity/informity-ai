from informity.scanner import watcher


def test_watcher_uses_persisted_supported_extensions_source(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watcher.invalidate_watcher_cache()
    monkeypatch.setattr(watcher.settings, 'supported_extensions', ['.txt'])
    monkeypatch.setattr(
        watcher,
        'get_supported_extensions_for_scan',
        lambda: ['.pdf', ' .DOCX ', '.TXT'],
    )

    ext_set = watcher._get_cached_supported_extensions_set()
    assert ext_set == frozenset({'.pdf', '.docx', '.txt'})


def test_watcher_falls_back_to_in_memory_extensions_on_read_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watcher.invalidate_watcher_cache()
    monkeypatch.setattr(watcher.settings, 'supported_extensions', ['.txt', '.md'])

    def _raise() -> list[str]:
        raise OSError('cannot read config')

    monkeypatch.setattr(watcher, 'get_supported_extensions_for_scan', _raise)

    ext_set = watcher._get_cached_supported_extensions_set()
    assert ext_set == frozenset({'.txt', '.md'})
