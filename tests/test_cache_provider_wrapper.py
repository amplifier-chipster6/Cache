import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "cache-provider-wrapper"))

from cache_provider_wrapper.provider import CacheProviderWrapper  # type: ignore[import-untyped]  # noqa: E402


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


_SENTINEL = object()


class DummyEmbedder:
    def __init__(self, result=_SENTINEL):
        # Sentinel allows callers to explicitly pass result=None to simulate
        # an embedder failure (returns None → semantic path skipped).
        self._result = [0.1, 0.2, 0.3] if result is _SENTINEL else result
        self.last_input = None

    def __call__(self, key):
        self.last_input = key
        return self._result


class DummySemanticCache:
    """Returns list of (metadata, distance) tuples — matches ChromaCache.query API."""

    def __init__(self, results):
        self.results = results
        self.last_embedding = None

    def query(self, embedding):
        self.last_embedding = embedding
        return self.results


# ---------------------------------------------------------------------------
# Existing exact-cache test (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_exact_cache_hit():
    provider = DummyProvider()
    exact_cache = DummyExactCache("cached")
    wrapper = CacheProviderWrapper(
        provider, exact_cache=exact_cache, semantic_cache=None
    )

    result = asyncio.run(wrapper.complete("request"))

    assert result == "cached"
    assert provider.called is False
    assert exact_cache.last_key == "request"


# ---------------------------------------------------------------------------
# Task 4: provider semantic path tests
# ---------------------------------------------------------------------------


class TestSemanticPath:
    """CacheProviderWrapper must route through semantic cache with threshold guard."""

    def _wrapper(self, semantic_results, embedder=None, threshold=None):
        provider = DummyProvider()
        exact_cache = DummyExactCache(None)
        emb = embedder or DummyEmbedder()
        semantic_cache = DummySemanticCache(semantic_results)
        cfg = {}
        if threshold is not None:
            cfg["semantic_threshold"] = threshold
        return (
            CacheProviderWrapper(
                provider,
                exact_cache=exact_cache,
                semantic_cache=semantic_cache,
                embedder=emb,
                config=cfg,
            ),
            provider,
            emb,
            semantic_cache,
        )

    def test_semantic_hit_above_threshold(self):
        # distance=0.05 → similarity=0.95, above default threshold 0.90
        wrapper, provider, embedder, semantic_cache = self._wrapper(
            [("semantic-payload", 0.05)]
        )
        result = asyncio.run(wrapper.complete("the-request"))
        assert result == "semantic-payload"
        assert provider.called is False
        assert embedder.last_input == "the-request"
        assert semantic_cache.last_embedding == [0.1, 0.2, 0.3]

    def test_semantic_hit_at_threshold(self):
        # distance=0.10 → similarity=0.90, exactly at default threshold — accepted
        wrapper, provider, _, _ = self._wrapper([("edge", 0.10)])
        result = asyncio.run(wrapper.complete("req"))
        assert result == "edge"
        assert provider.called is False

    def test_threshold_rejection_falls_through_to_provider(self):
        # distance=0.15 → similarity=0.85, below default threshold 0.90 → rejected
        wrapper, provider, _, _ = self._wrapper([("too-far", 0.15)])
        result = asyncio.run(wrapper.complete("req"))
        assert result == "live"
        assert provider.called is True

    def test_custom_threshold_respected(self):
        # distance=0.25 → similarity=0.75; threshold=0.70 → accepted
        wrapper, provider, _, _ = self._wrapper([("near", 0.25)], threshold=0.70)
        result = asyncio.run(wrapper.complete("req"))
        assert result == "near"
        assert provider.called is False

    def test_embed_failure_skips_semantic(self):
        # embedder returns None → semantic path bypassed → live provider
        wrapper, provider, _, semantic_cache = self._wrapper(
            [("should-not-hit", 0.0)],
            embedder=DummyEmbedder(result=None),
        )
        result = asyncio.run(wrapper.complete("req"))
        assert result == "live"
        assert provider.called is True
        assert semantic_cache.last_embedding is None

    def test_empty_semantic_results_falls_through(self):
        wrapper, provider, _, _ = self._wrapper([])
        result = asyncio.run(wrapper.complete("req"))
        assert result == "live"
        assert provider.called is True

    def test_exact_hit_bypasses_semantic(self):
        # exact cache returns a value → semantic path never reached
        provider = DummyProvider()
        exact_cache = DummyExactCache("exact-hit")
        embedder = DummyEmbedder()
        semantic_cache = DummySemanticCache([("semantic", 0.0)])
        wrapper = CacheProviderWrapper(
            provider,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=embedder,
        )
        result = asyncio.run(wrapper.complete("req"))
        assert result == "exact-hit"
        assert provider.called is False
        assert semantic_cache.last_embedding is None

    def test_no_embedder_skips_semantic(self):
        # embedder not provided at all → semantic path skipped
        provider = DummyProvider()
        exact_cache = DummyExactCache(None)
        semantic_cache = DummySemanticCache([("semantic", 0.0)])
        wrapper = CacheProviderWrapper(
            provider,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=None,
        )
        result = asyncio.run(wrapper.complete("req"))
        assert result == "live"
        assert provider.called is True
        assert semantic_cache.last_embedding is None
