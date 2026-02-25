import sqlite3

import pytest

from cache_store_sqlite.store import SQLiteCache


def test_exact_cache_roundtrip(tmp_path):
    db = SQLiteCache(tmp_path / "cache.sqlite")
    key = "abc"
    db.set(key, {"value": 123})
    assert db.get(key) == {"value": 123}


def test_close_closes_connection(tmp_path):
    db = SQLiteCache(tmp_path / "cache.sqlite")
    db.close()

    with pytest.raises(sqlite3.ProgrammingError):
        db.set("abc", {"value": 123})
