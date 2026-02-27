import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# Default values mirror behaviors/hybrid-cache.yaml
_DEFAULT_TTL: int = 86400  # seconds (24 h)
_DEFAULT_MAX_ENTRIES: int | None = None  # unlimited


class SQLiteCache:
    """Exact-match cache backed by SQLite.

    Task 9 hardening applied:
    - ``check_same_thread=False`` so the same instance can be used from worker
      threads (e.g. inside asyncio.run_in_executor).
    - WAL journal mode reduces lock contention and improves concurrent read
      performance.
    - Schema init is idempotent (CREATE TABLE IF NOT EXISTS + safe ALTER TABLE
      for the ``created_at`` column added in Task 8).

    Task 8 TTL / eviction:
    - Every ``get`` checks whether the entry has expired; expired rows are
      deleted and ``None`` returned as if the entry never existed.
    - Every ``set`` inserts with the current epoch timestamp and then evicts
      the oldest row(s) when ``max_entries`` is exceeded.

    Config keys (nested dict passed as ``config``):
    - ``config["cache"]["exact"]["ttl"]``         — int seconds, default 86400
    - ``config["cache"]["exact"]["max_entries"]`` — int|None, default None
    """

    def __init__(self, database_path: str | Path, config: dict | None = None) -> None:
        self._database_path = Path(database_path)
        cfg = (config or {}).get("cache", {}).get("exact", {})
        self._ttl: int = cfg.get("ttl", _DEFAULT_TTL)
        self._max_entries: int | None = cfg.get("max_entries", _DEFAULT_MAX_ENTRIES)

        # Task 9: check_same_thread=False allows multi-thread/async access.
        self._connection = sqlite3.connect(self._database_path, check_same_thread=False)

        # Task 9: WAL mode — set before any DDL so all subsequent ops benefit.
        self._connection.execute("PRAGMA journal_mode=WAL")

        # Task 9: idempotent table creation.
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT, created_at INTEGER DEFAULT 0)"
        )

        # Task 8: safe migration — add created_at to any pre-existing table.
        try:
            self._connection.execute(
                "ALTER TABLE cache ADD COLUMN created_at INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # column already exists

        self._connection.commit()

    # ------------------------------------------------------------------
    # Public API (unchanged shape from original)
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` if missing / expired."""
        cursor = self._connection.execute(
            "SELECT value, created_at FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        value_json, created_at = row

        # Task 8: TTL check — treat expired entry as a miss and purge it.
        age = time.time() - (created_at or 0)
        if age > self._ttl:
            self._connection.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._connection.commit()
            return None

        return json.loads(value_json)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` and enforce ``max_entries`` eviction."""
        payload = json.dumps(value)
        now = int(time.time())
        self._connection.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, payload, now),
        )

        # Task 8: evict oldest entries when the table exceeds max_entries.
        if self._max_entries is not None:
            (count,) = self._connection.execute("SELECT COUNT(*) FROM cache").fetchone()
            excess = count - self._max_entries
            if excess > 0:
                self._connection.execute(
                    "DELETE FROM cache WHERE key IN "
                    "(SELECT key FROM cache ORDER BY created_at ASC LIMIT ?)",
                    (excess,),
                )

        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
