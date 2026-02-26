# Hybrid Local Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a hybrid local cache (exact-match + semantic) for Amplifier by wrapping provider and tool modules, using SQLite for exact-match storage and Chroma (local mode) for semantic search with local embeddings.

**Architecture:** Implement provider/tool wrapper modules that normalize inputs, check exact cache, optionally check semantic cache, and fall back to real provider/tool. Storage is local: SQLite for exact entries and Chroma for vectors; cache failures never block execution.

**Tech Stack:** Python, Amplifier module contracts, SQLite, Chroma (local mode), local embeddings (sentence-transformers).

---

### Task 1: Establish repository structure and bundle configuration

**Files:**
- Create: `bundle.md`
- Create: `behaviors/hybrid-cache.yaml`
- Create: `context/hybrid-cache-awareness.md`

**Step 1: Write the failing test**
- Not applicable (config scaffolding). Create a TODO note in the plan to add validation checks later.

**Step 2: Create bundle.md**
```yaml
bundle:
  name: cache
  version: 0.1.0

includes:
  - foundation:bundle
  - cache:behaviors/hybrid-cache.yaml
```

**Step 3: Create behaviors/hybrid-cache.yaml**
```yaml
bundle:
  name: hybrid-cache
  version: 0.1.0

modules:
  include:
    - cache:modules/cache-provider-wrapper
    - cache:modules/cache-tool-wrapper
    - cache:modules/cache-store-sqlite
    - cache:modules/cache-store-chroma
    - cache:modules/cache-normalizer

context:
  include:
    - cache:context/hybrid-cache-awareness.md

config:
  cache:
    enabled: true
    exact:
      ttl: 86400
      max_entries: 50000
    semantic:
      enabled: true
      threshold: 0.90
    storage:
      path: ./cache
```

**Step 4: Create context/hybrid-cache-awareness.md**
```markdown
# Hybrid Cache Behavior

This bundle enables hybrid caching (exact + semantic) for providers and tools using local storage.
```

**Step 5: Commit**
```bash
git add bundle.md behaviors/hybrid-cache.yaml context/hybrid-cache-awareness.md
git commit -m "feat: add hybrid cache bundle scaffolding"
```

---

### Task 2: Implement cache normalization utilities

**Files:**
- Create: `modules/cache-normalizer/pyproject.toml`
- Create: `modules/cache-normalizer/cache_normalizer/__init__.py`
- Create: `modules/cache-normalizer/cache_normalizer/normalize.py`
- Test: `tests/test_cache_normalizer.py`

**Step 1: Write the failing test**
```python
from cache_normalizer.normalize import normalize_request

def test_normalize_request_stable_ordering():
    data = {"b": 2, "a": 1}
    assert normalize_request(data) == "{\"a\":1,\"b\":2}"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cache_normalizer.py::test_normalize_request_stable_ordering -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**
```python
import json

def normalize_request(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cache_normalizer.py::test_normalize_request_stable_ordering -v`
Expected: PASS

**Step 5: Commit**
```bash
git add modules/cache-normalizer tests/test_cache_normalizer.py
git commit -m "feat: add cache normalization utilities"
```

---

### Task 3: Implement exact-match cache store (SQLite)

**Files:**
- Create: `modules/cache-store-sqlite/pyproject.toml`
- Create: `modules/cache-store-sqlite/cache_store_sqlite/__init__.py`
- Create: `modules/cache-store-sqlite/cache_store_sqlite/store.py`
- Test: `tests/test_cache_store_sqlite.py`

**Step 1: Write the failing test**
```python
from cache_store_sqlite.store import SQLiteCache

def test_exact_cache_roundtrip(tmp_path):
    db = SQLiteCache(tmp_path / "cache.sqlite")
    key = "abc"
    db.set(key, {"value": 123})
    assert db.get(key) == {"value": 123}
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cache_store_sqlite.py::test_exact_cache_roundtrip -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**
```python
import json
import sqlite3

class SQLiteCache:
    def __init__(self, path):
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str):
        row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict):
        self._conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self._conn.commit()
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cache_store_sqlite.py::test_exact_cache_roundtrip -v`
Expected: PASS

**Step 5: Commit**
```bash
git add modules/cache-store-sqlite tests/test_cache_store_sqlite.py
git commit -m "feat: add sqlite exact cache store"
```

---

### Task 4: Implement semantic cache store (Chroma, local)

**Files:**
- Create: `modules/cache-store-chroma/pyproject.toml`
- Create: `modules/cache-store-chroma/cache_store_chroma/__init__.py`
- Create: `modules/cache-store-chroma/cache_store_chroma/store.py`
- Test: `tests/test_cache_store_chroma.py`

**Step 1: Write the failing test**
```python
from cache_store_chroma.store import ChromaCache

def test_semantic_cache_insert_query(tmp_path):
    cache = ChromaCache(tmp_path / "chroma")
    cache.add("id1", [0.1, 0.2, 0.3], {"value": "ok"})
    result = cache.query([0.1, 0.2, 0.3])
    assert result[0]["value"] == "ok"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cache_store_chroma.py::test_semantic_cache_insert_query -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**
```python
import chromadb

class ChromaCache:
    def __init__(self, path):
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection("cache")

    def add(self, id: str, embedding: list[float], payload: dict):
        self._collection.add(ids=[id], embeddings=[embedding], metadatas=[payload])

    def query(self, embedding: list[float], n_results: int = 1):
        result = self._collection.query(query_embeddings=[embedding], n_results=n_results)
        return result["metadatas"][0]
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cache_store_chroma.py::test_semantic_cache_insert_query -v`
Expected: PASS

**Step 5: Commit**
```bash
git add modules/cache-store-chroma tests/test_cache_store_chroma.py
git commit -m "feat: add chroma semantic cache store"
```

---

### Task 5: Implement provider cache wrapper module

**Files:**
- Create: `modules/cache-provider-wrapper/pyproject.toml`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/__init__.py`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/provider.py`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/mount.py`
- Test: `tests/test_cache_provider_wrapper.py`

**Step 1: Write the failing test**
```python
from cache_provider_wrapper.provider import CacheProviderWrapper

class DummyProvider:
    name = "dummy"
    async def complete(self, request, **kwargs):
        return "live"

async def test_exact_cache_hit():
    wrapper = CacheProviderWrapper(DummyProvider(), exact_cache=None, semantic_cache=None)
    # set up exact cache hit and verify no provider call
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cache_provider_wrapper.py::test_exact_cache_hit -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**
```python
from amplifier_core.interfaces import Provider

class CacheProviderWrapper(Provider):
    def __init__(self, provider, exact_cache, semantic_cache, normalizer, embedder, config):
        ...

    async def complete(self, request, **kwargs):
        # normalize -> exact lookup -> semantic lookup -> provider
        ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cache_provider_wrapper.py::test_exact_cache_hit -v`
Expected: PASS

**Step 5: Commit**
```bash
git add modules/cache-provider-wrapper tests/test_cache_provider_wrapper.py
git commit -m "feat: add provider cache wrapper"
```

---

### Task 6: Implement tool cache wrapper module

**Files:**
- Create: `modules/cache-tool-wrapper/pyproject.toml`
- Create: `modules/cache-tool-wrapper/cache_tool_wrapper/__init__.py`
- Create: `modules/cache-tool-wrapper/cache_tool_wrapper/tool.py`
- Create: `modules/cache-tool-wrapper/cache_tool_wrapper/mount.py`
- Test: `tests/test_cache_tool_wrapper.py`

**Step 1: Write the failing test**
```python
from cache_tool_wrapper.tool import CacheToolWrapper

class DummyTool:
    name = "dummy"
    description = "dummy"
    async def execute(self, input):
        return {"ok": True}

async def test_exact_cache_hit_tool():
    wrapper = CacheToolWrapper(DummyTool(), exact_cache=None, semantic_cache=None)
    # set up exact cache hit and verify no tool call
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cache_tool_wrapper.py::test_exact_cache_hit_tool -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**
```python
from amplifier_core.interfaces import Tool

class CacheToolWrapper(Tool):
    def __init__(self, tool, exact_cache, semantic_cache, normalizer, embedder, config):
        ...

    async def execute(self, input):
        # normalize -> exact lookup -> optional semantic -> tool execute
        ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cache_tool_wrapper.py::test_exact_cache_hit_tool -v`
Expected: PASS

**Step 5: Commit**
```bash
git add modules/cache-tool-wrapper tests/test_cache_tool_wrapper.py
git commit -m "feat: add tool cache wrapper"
```

---

### Task 7: Wire modules into bundle and add usage docs

**Files:**
- Modify: `context/hybrid-cache-awareness.md`

**Step 1: Update awareness doc**
Add a section describing how to enable the bundle in `amplifier run` and how to configure cache settings.

**Step 2: Commit**
```bash
git add context/hybrid-cache-awareness.md
git commit -m "docs: document hybrid cache usage"
```

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-02-23-hybrid-cache-plan.md`. Two execution options:

1. **Subagent-Driven (this session)** - I dispatch a fresh subagent per task with reviews between tasks.
2. **Parallel Session (separate)** - Open a new session with executing-plans for batch execution.

Which approach?
