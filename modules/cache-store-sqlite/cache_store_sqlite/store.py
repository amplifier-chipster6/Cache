import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteCache:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._connection = sqlite3.connect(self._database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._connection.commit()

    def get(self, key: str) -> Any | None:
        cursor = self._connection.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self._connection.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, payload),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
