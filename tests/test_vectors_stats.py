from informity.db.vectors import vector_store


class _FakeCursor:
    def __init__(self, row: dict[str, int]) -> None:
        self._row = row

    def fetchone(self) -> dict[str, int]:
        return self._row


class _FakeConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute(self, query: str) -> _FakeCursor:
        self.queries.append(query)
        if 'COUNT(*)' in query:
            return _FakeCursor({'count': 12})
        if 'dbstat' in query:
            return _FakeCursor({'size_bytes': 3456})
        raise AssertionError(f'unexpected query: {query}')


def test_vector_store_get_stats_uses_dbstat_for_storage_bytes(monkeypatch) -> None:
    fake_conn = _FakeConnection()
    monkeypatch.setattr(vector_store, '_get_thread_connection', lambda: fake_conn)

    stats = vector_store.get_stats()

    assert stats == {'total_vectors': 12, 'storage_bytes': 3456}
    assert any('dbstat' in query for query in fake_conn.queries)
