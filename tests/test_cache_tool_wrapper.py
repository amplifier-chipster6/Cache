import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "cache-tool-wrapper"))

from cache_tool_wrapper.tool import CacheToolWrapper  # type: ignore[import-untyped]  # noqa: E402


class DummyTool:
    name = "dummy"
    description = "dummy"

    def __init__(self):
        self.called = False

    async def execute(self, input):
        self.called = True
        return {"ok": True}


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
        self.last_input = None
        # Use a sentinel so callers can explicitly pass result=None to simulate
        # an embedder that returns None (failed / skipped).
        self._result = ["embedding"] if result is _SENTINEL else result

    def __call__(self, input):
        self.last_input = input
        return self._result


class DummySemanticCache:
    """Returns list of (metadata, distance) tuples — matches ChromaCache.query API."""

    def __init__(self, results):
        # results: list[tuple[dict, float]]
        self.results = results
        self.last_embedding = None

    def query(self, embedding):
        self.last_embedding = embedding
        return self.results


# ---------------------------------------------------------------------------
# Existing exact-cache tests (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_exact_cache_hit_tool():
    tool = DummyTool()
    exact_cache = DummyExactCache({"ok": "cached"})
    wrapper = CacheToolWrapper(tool, exact_cache=exact_cache, semantic_cache=None)

    result = asyncio.run(wrapper.execute({"input": True}))

    assert result == {"ok": "cached"}
    assert tool.called is False
    assert exact_cache.last_key == {"input": True}


def test_semantic_cache_hit_tool():
    tool = DummyTool()
    exact_cache = DummyExactCache(None)
    embedder = DummyEmbedder()
    # distance=0.0 → similarity=1.0, passes default threshold 0.90
    semantic_cache = DummySemanticCache([({"ok": "semantic"}, 0.0)])
    wrapper = CacheToolWrapper(
        tool,
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        normalizer=lambda value: {"normalized": value},
        embedder=embedder,
    )

    result = asyncio.run(wrapper.execute({"input": True}))

    assert result == {"ok": "semantic"}
    assert tool.called is False
    assert embedder.last_input == {"normalized": {"input": True}}
    assert semantic_cache.last_embedding == ["embedding"]


def test_semantic_cache_hit_falsy_payload():
    tool = DummyTool()
    exact_cache = DummyExactCache(None)
    embedder = DummyEmbedder()
    # Empty dict payload; distance=0.0 passes threshold
    semantic_cache = DummySemanticCache([({}, 0.0)])
    wrapper = CacheToolWrapper(
        tool,
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        normalizer=lambda value: {"normalized": value},
        embedder=embedder,
    )

    result = asyncio.run(wrapper.execute({"input": True}))

    assert result == {}
    assert tool.called is False
    assert embedder.last_input == {"normalized": {"input": True}}
    assert semantic_cache.last_embedding == ["embedding"]


# ---------------------------------------------------------------------------
# New Task 5 tests: threshold enforcement
# ---------------------------------------------------------------------------


class TestSemantic:
    """Task 5: CacheToolWrapper must enforce the similarity threshold."""

    def _wrapper(self, semantic_results, threshold=None):
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        embedder = DummyEmbedder(result=[0.1, 0.2, 0.3])
        semantic_cache = DummySemanticCache(semantic_results)
        cfg = {"semantic_threshold": threshold} if threshold is not None else None
        return (
            CacheToolWrapper(
                tool,
                exact_cache=exact_cache,
                semantic_cache=semantic_cache,
                embedder=embedder,
                config=cfg,
            ),
            tool,
        )

    def test_accepts_hit_at_threshold(self):
        # distance=0.10 → similarity=0.90, exactly at default threshold
        wrapper, tool = self._wrapper([({"answer": "semantic"}, 0.10)])
        result = asyncio.run(wrapper.execute("query"))
        assert result == {"answer": "semantic"}
        assert tool.called is False

    def test_accepts_hit_above_threshold(self):
        # distance=0.05 → similarity=0.95, above default threshold 0.90
        wrapper, tool = self._wrapper([({"answer": "close"}, 0.05)])
        result = asyncio.run(wrapper.execute("query"))
        assert result == {"answer": "close"}
        assert tool.called is False

    def test_rejects_hit_below_threshold(self):
        # distance=0.15 → similarity=0.85, below default threshold 0.90
        wrapper, tool = self._wrapper([({"answer": "too-far"}, 0.15)])
        result = asyncio.run(wrapper.execute("query"))
        # Semantic result rejected; falls through to live tool
        assert result == {"ok": True}
        assert tool.called is True

    def test_custom_threshold_rejects_distant(self):
        # distance=0.30 → similarity=0.70, below threshold 0.80 → rejected
        wrapper, tool = self._wrapper([({"answer": "nope"}, 0.30)], threshold=0.80)
        asyncio.run(wrapper.execute("query"))
        assert tool.called is True

    def test_custom_threshold_accepts_within(self):
        # distance=0.30 → similarity=0.70, passes threshold 0.60
        wrapper, tool = self._wrapper([({"answer": "yes"}, 0.30)], threshold=0.60)
        result = asyncio.run(wrapper.execute("query"))
        assert result == {"answer": "yes"}
        assert tool.called is False

    def test_embed_none_skips_semantic(self):
        # embedder returns None → semantic path skipped entirely
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        embedder = DummyEmbedder(result=None)
        semantic_cache = DummySemanticCache([({"answer": "should-not-hit"}, 0.0)])
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=embedder,
        )
        result = asyncio.run(wrapper.execute("query"))
        assert result == {"ok": True}
        assert tool.called is True
        assert semantic_cache.last_embedding is None

    def test_empty_semantic_results_falls_through(self):
        wrapper, tool = self._wrapper([])
        result = asyncio.run(wrapper.execute("query"))
        assert result == {"ok": True}
        assert tool.called is True
