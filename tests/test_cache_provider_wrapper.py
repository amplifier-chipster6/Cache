import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "cache-provider-wrapper"))

from cache_provider_wrapper.provider import CacheProviderWrapper  # noqa: E402


class DummyProvider:
    name = "dummy"

    def __init__(self):
        self.called = False

    async def complete(self, request, **kwargs):
        self.called = True
        return "live"


class DummyExactCache:
    def __init__(self, value):
        self.value = value
        self.last_key = None

    def get(self, key):
        self.last_key = key
        return self.value


def test_exact_cache_hit():
    provider = DummyProvider()
    exact_cache = DummyExactCache("cached")
    wrapper = CacheProviderWrapper(provider, exact_cache=exact_cache, semantic_cache=None)

    result = asyncio.run(wrapper.complete("request"))

    assert result == "cached"
    assert provider.called is False
    assert exact_cache.last_key == "request"
