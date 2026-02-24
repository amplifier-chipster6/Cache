# Provider Cache Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a provider cache wrapper module with tests and pytest config so provider calls can be cached.

**Architecture:** Create a small wrapper module in modules/cache-provider-wrapper that composes a provider with a cache interface. Expose CacheProviderWrapper via a minimal package API and validate cache-hit behavior via a focused unit test.

**Tech Stack:** Python, pytest

---

### Task 1: Provider cache wrapper

**Files:**
- Create: `modules/cache-provider-wrapper/pyproject.toml`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/__init__.py`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/provider.py`
- Create: `modules/cache-provider-wrapper/cache_provider_wrapper/mount.py`
- Modify: `pytest.ini`
- Test: `tests/test_cache_provider_wrapper.py`

**Step 1: Write the failing test**

```python
from cache_provider_wrapper.provider import CacheProviderWrapper

class DummyCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        return self.store.get(key)
    def set(self, key, value):
        self.store[key] = value

class DummyProvider:
    def __init__(self):
        self.calls = 0
    def fetch(self, key):
        self.calls += 1
        return f"value:{key}"

def test_cache_provider_wrapper_exact_cache_hit():
    cache = DummyCache()
    cache.set("alpha", "cached:alpha")
    provider = DummyProvider()
    wrapper = CacheProviderWrapper(cache=cache, provider=provider)

    assert wrapper.fetch("alpha") == "cached:alpha"
    assert provider.calls == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache_provider_wrapper.py::test_cache_provider_wrapper_exact_cache_hit -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'cache_provider_wrapper'"

**Step 3: Write minimal implementation**

```python
class CacheProviderWrapper:
    def __init__(self, cache, provider):
        self.cache = cache
        self.provider = provider

    def fetch(self, key):
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        value = self.provider.fetch(key)
        self.cache.set(key, value)
        return value
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cache_provider_wrapper.py::test_cache_provider_wrapper_exact_cache_hit -v`
Expected: PASS

**Step 5: Update pytest config**

Add module path so tests can import `cache_provider_wrapper`.

**Step 6: Commit**

```bash
git add pytest.ini modules/cache-provider-wrapper tests/test_cache_provider_wrapper.py
git commit -m "feat: add provider cache wrapper"
```
