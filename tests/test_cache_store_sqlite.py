from cache_store_sqlite.store import SQLiteCache


def test_exact_cache_roundtrip(tmp_path):
    db = SQLiteCache(tmp_path / "cache.sqlite")
    key = "abc"
    db.set(key, {"value": 123})
    assert db.get(key) == {"value": 123}
