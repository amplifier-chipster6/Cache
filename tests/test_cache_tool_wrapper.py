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
        self.set_calls: list = []

    def get(self, key):
        self.last_key = key
        return self.value

    def set(self, key, value):
        self.set_calls.append((key, value))


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
        self.add_calls: list = []

    def query(self, embedding):
        self.last_embedding = embedding
        return self.results

    def add(self, id, embedding, payload):
        self.add_calls.append((id, embedding, payload))


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


# ---------------------------------------------------------------------------
# Task 2: metrics counters and hit_ratio for CacheToolWrapper
# ---------------------------------------------------------------------------


class TestMetricsCounters:
    """CacheToolWrapper must maintain counters and expose _last_metrics."""

    def _make_wrapper(
        self,
        exact_value=None,
        semantic_results=(),
        embedder=None,
        threshold=None,
        use_exact_cache=True,
        use_semantic_cache=True,
    ):
        """Factory for a fully-wired wrapper; returns (wrapper, tool)."""
        tool = DummyTool()
        exact_cache = DummyExactCache(exact_value) if use_exact_cache else None
        semantic_cache = (
            DummySemanticCache(list(semantic_results)) if use_semantic_cache else None
        )
        cfg = {"semantic_threshold": threshold} if threshold is not None else {}
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=embedder,
            config=cfg,
        )
        return wrapper, tool

    # ------------------------------------------------------------------
    # Exact-hit path
    # ------------------------------------------------------------------

    def test_exact_hit_increments_exact_hits(self):
        wrapper, tool = self._make_wrapper(
            exact_value={"cached": True}, use_semantic_cache=False
        )
        result = asyncio.run(wrapper.execute({"q": 1}))
        assert result == {"cached": True}
        assert wrapper.exact_hits == 1
        assert wrapper.exact_misses == 0
        assert wrapper.tool_calls == 0
        assert tool.called is False

    def test_exact_hit_metrics_flags(self):
        wrapper, _ = self._make_wrapper(
            exact_value={"cached": True}, use_semantic_cache=False
        )
        asyncio.run(wrapper.execute({"q": 1}))
        m = wrapper._last_metrics
        assert m is not None
        assert m["exact_hit"] is True
        assert m["semantic_hit"] is False
        assert m["provider_call"] is False
        assert m["embed_call"] is False

    def test_exact_hit_hit_ratio_is_one(self):
        wrapper, _ = self._make_wrapper(
            exact_value={"cached": True}, use_semantic_cache=False
        )
        asyncio.run(wrapper.execute({"q": 1}))
        assert wrapper._last_metrics["hit_ratio"] == 1.0

    def test_exact_hit_counts_snapshot(self):
        wrapper, _ = self._make_wrapper(
            exact_value={"cached": True}, use_semantic_cache=False
        )
        asyncio.run(wrapper.execute({"q": 1}))
        counts = wrapper._last_metrics["counts"]
        assert counts["exact_hits"] == 1
        assert counts["tool_calls"] == 0

    # ------------------------------------------------------------------
    # Semantic-hit path
    # ------------------------------------------------------------------

    def test_semantic_hit_increments_semantic_hits(self):
        # distance=0.05 → similarity=0.95, above default threshold 0.90
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"sem": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"sem": True}
        assert wrapper.semantic_hits == 1
        assert wrapper.exact_hits == 0
        assert wrapper.tool_calls == 0
        assert tool.called is False

    def test_semantic_hit_increments_embed_calls(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"sem": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 1

    def test_semantic_hit_metrics_flags(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"sem": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        assert m["semantic_hit"] is True
        assert m["exact_hit"] is False
        assert m["provider_call"] is False
        assert m["embed_call"] is True

    def test_semantic_hit_hit_ratio_is_one(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"sem": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper._last_metrics["hit_ratio"] == 1.0

    # ------------------------------------------------------------------
    # Embedder returns None → semantic skip + tool fallback
    # ------------------------------------------------------------------

    def test_embedder_returns_none_increments_semantic_skips(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"would": "hit"}, 0.0)]),
            embedder=DummyEmbedder(result=None),
            config={},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert wrapper.semantic_skips == 1
        assert wrapper.tool_calls == 1
        assert tool.called is True

    def test_embedder_returns_none_still_counts_embed_call(self):
        # Embedder callable was invoked even though it returned None
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),
            embedder=DummyEmbedder(result=None),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 1

    def test_embedder_returns_none_metrics_flags(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"would": "hit"}, 0.0)]),
            embedder=DummyEmbedder(result=None),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        assert m["semantic_skip"] is True
        assert m["provider_call"] is True
        assert m["embed_call"] is True
        assert m["semantic_hit"] is False

    # ------------------------------------------------------------------
    # Threshold reject → semantic skip + tool fallback
    # ------------------------------------------------------------------

    def test_threshold_reject_increments_semantic_skips(self):
        # distance=0.20 → similarity=0.80, below default 0.90
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"too": "far"}, 0.20)]),
            embedder=DummyEmbedder(),
            config={},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert wrapper.semantic_skips == 1
        assert wrapper.semantic_hits == 0
        assert wrapper.tool_calls == 1
        assert tool.called is True

    def test_threshold_reject_increments_embed_calls(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"too": "far"}, 0.20)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 1

    def test_threshold_reject_metrics_flags(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"too": "far"}, 0.20)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        assert m["semantic_skip"] is True
        assert m["provider_call"] is True
        assert m["semantic_hit"] is False

    # ------------------------------------------------------------------
    # No embedder configured → semantic skip + tool fallback
    # ------------------------------------------------------------------

    def test_no_embedder_increments_semantic_skips(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"would": "hit"}, 0.0)]),
            embedder=None,  # no embedder provided
            config={},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert wrapper.semantic_skips == 1
        assert wrapper.tool_calls == 1
        assert tool.called is True

    def test_no_embedder_does_not_increment_embed_calls(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),
            embedder=None,
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 0

    # ------------------------------------------------------------------
    # Semantic miss (empty results) → semantic_misses + tool fallback
    # ------------------------------------------------------------------

    def test_semantic_miss_increments_semantic_misses(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),  # empty → miss
            embedder=DummyEmbedder(),
            config={},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert wrapper.semantic_misses == 1
        assert wrapper.semantic_hits == 0
        assert wrapper.tool_calls == 1
        assert tool.called is True

    def test_semantic_miss_increments_embed_calls(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 1

    def test_semantic_miss_metrics_flags(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        assert m["provider_call"] is True
        assert m["embed_call"] is True
        assert m["semantic_hit"] is False
        # semantic_miss tracked via counts snapshot (no dedicated top-level flag)
        assert m["counts"]["semantic_misses"] == 1

    # ------------------------------------------------------------------
    # hit_ratio zero-guard (no calls made yet)
    # ------------------------------------------------------------------

    def test_hit_ratio_zero_guard_counters_start_at_zero(self):
        """All counters initialise to 0 and _last_metrics is None before any call."""
        wrapper, _ = self._make_wrapper(exact_value=None, use_semantic_cache=False)
        assert wrapper._last_metrics is None
        assert wrapper.exact_hits == 0
        assert wrapper.exact_misses == 0
        assert wrapper.semantic_hits == 0
        assert wrapper.semantic_misses == 0
        assert wrapper.semantic_skips == 0
        assert wrapper.tool_calls == 0
        assert wrapper.embed_calls == 0

    def test_hit_ratio_zero_when_only_provider_calls(self):
        """All requests go to tool → ratio is 0.0 (zero hits / total)."""
        wrapper, _ = self._make_wrapper(
            exact_value=None,
            use_semantic_cache=False,
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper._last_metrics["hit_ratio"] == 0.0

    def test_hit_ratio_accumulates_across_calls(self):
        """Two calls: first exact hit, second tool fallback → ratio 0.5."""
        tool = DummyTool()
        exact_cache_hit = DummyExactCache({"cached": True})
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache_hit,
            semantic_cache=None,
        )
        asyncio.run(wrapper.execute("req1"))
        assert wrapper._last_metrics["hit_ratio"] == 1.0  # 1 hit / 1 total

        # Swap the exact cache to return None → next call falls through to tool
        wrapper.exact_cache = DummyExactCache(None)
        asyncio.run(wrapper.execute("req2"))
        # 1 exact_hit, 1 provider_call → ratio = 1/2
        assert wrapper._last_metrics["hit_ratio"] == 0.5

    # ------------------------------------------------------------------
    # embed_calls accumulates correctly over multiple semantic attempts
    # ------------------------------------------------------------------

    def test_embed_calls_increments_on_each_semantic_attempt(self):
        """embed_calls increases by 1 per call when on the semantic path."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"payload": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req1"))
        assert wrapper.embed_calls == 1
        asyncio.run(wrapper.execute("req2"))
        assert wrapper.embed_calls == 2

    def test_embed_calls_not_incremented_on_exact_hit(self):
        """Exact cache hit short-circuits before embedder is ever called."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache({"cached": True}),
            semantic_cache=DummySemanticCache([({"payload": True}, 0.0)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 0

    # ------------------------------------------------------------------
    # _last_metrics structure completeness
    # ------------------------------------------------------------------

    def test_last_metrics_contains_required_keys(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({"payload": True}, 0.05)]),
            embedder=DummyEmbedder(),
            config={},
        )
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        assert m is not None
        for key in (
            "exact_hit",
            "semantic_hit",
            "semantic_skip",
            "provider_call",
            "embed_call",
            "counts",
            "hit_ratio",
        ):
            assert key in m, f"Missing key in _last_metrics: {key!r}"

    def test_counts_snapshot_contains_all_counter_keys(self):
        wrapper, _ = self._make_wrapper(exact_value=None, use_semantic_cache=False)
        asyncio.run(wrapper.execute("req"))
        counts = wrapper._last_metrics["counts"]
        for key in (
            "exact_hits",
            "exact_misses",
            "semantic_hits",
            "semantic_misses",
            "semantic_skips",
            "embed_errors",
            "threshold_rejects",
            "tool_calls",
            "embed_calls",
        ):
            assert key in counts, f"Missing counter key in counts snapshot: {key!r}"


# ---------------------------------------------------------------------------
# Task 3: metrics.enabled flag
# ---------------------------------------------------------------------------


class TestMetricsEnabledFlag:
    """Counters and _last_metrics always update; log emission respects the flag."""

    def _make_wrapper_with_flag(self, enabled: bool):
        """Return a wrapper whose metrics emission flag is set explicitly."""
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        config = {"metrics": {"enabled": enabled}}
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=None,
            config=config,
        )
        return wrapper, tool

    def test_metrics_enabled_true_by_default_when_config_absent(self):
        """No config → _metrics_enabled defaults to True."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=None, semantic_cache=None)
        assert wrapper._metrics_enabled is True

    def test_metrics_enabled_true_when_flag_set(self):
        wrapper, _ = self._make_wrapper_with_flag(enabled=True)
        assert wrapper._metrics_enabled is True

    def test_metrics_enabled_false_when_flag_set(self):
        wrapper, _ = self._make_wrapper_with_flag(enabled=False)
        assert wrapper._metrics_enabled is False

    def test_last_metrics_updated_when_emission_disabled(self):
        """Even with metrics disabled, _last_metrics is populated after a call."""
        wrapper, _ = self._make_wrapper_with_flag(enabled=False)
        assert wrapper._last_metrics is None  # nothing before any call
        asyncio.run(wrapper.execute("req"))
        assert wrapper._last_metrics is not None

    def test_counters_updated_when_emission_disabled(self):
        """Counters increment normally even when metrics emission is disabled."""
        wrapper, _ = self._make_wrapper_with_flag(enabled=False)
        asyncio.run(wrapper.execute("req"))
        assert wrapper.tool_calls == 1

    def test_last_metrics_has_correct_shape_when_emission_disabled(self):
        """_last_metrics payload structure is intact regardless of the flag."""
        wrapper, _ = self._make_wrapper_with_flag(enabled=False)
        asyncio.run(wrapper.execute("req"))
        m = wrapper._last_metrics
        for key in (
            "exact_hit",
            "semantic_hit",
            "semantic_skip",
            "provider_call",
            "embed_call",
            "counts",
            "hit_ratio",
        ):
            assert key in m, f"Missing key in _last_metrics: {key!r}"


# ---------------------------------------------------------------------------
# Helper: writable exact cache that stores write-backs and serves them back
# ---------------------------------------------------------------------------


class WritableExactCache:
    """Exact cache that persists write-backs; get() returns stored values."""

    def __init__(self):
        self.stored: dict = {}
        self.set_calls: list = []

    def get(self, key):
        return self.stored.get(key)

    def set(self, key, value):
        self.stored[key] = value
        self.set_calls.append((key, value))


# ---------------------------------------------------------------------------
# Task 1: write-back on miss (tool)
# ---------------------------------------------------------------------------


class TestWriteBack:
    """After a live tool call results are written to exact and semantic caches."""

    def test_exact_writeback_set_called(self):
        """exact_cache.set() is called with (key, result) after a live call."""
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        wrapper = CacheToolWrapper(tool, exact_cache=exact_cache, semantic_cache=None)
        asyncio.run(wrapper.execute("req"))
        assert exact_cache.set_calls == [("req", {"ok": True})]

    def test_second_call_hits_exact_after_writeback(self):
        """Result stored by write-back is returned as an exact hit on the next call."""
        tool = DummyTool()
        exact_cache = WritableExactCache()
        wrapper = CacheToolWrapper(tool, exact_cache=exact_cache, semantic_cache=None)

        asyncio.run(wrapper.execute("req"))  # miss -> live -> write-back
        assert len(exact_cache.set_calls) == 1

        tool.called = False
        asyncio.run(wrapper.execute("req"))  # should hit exact cache now
        assert tool.called is False
        assert wrapper.exact_hits == 1

    def test_no_writeback_on_exact_hit(self):
        """No write-back occurs when the exact cache already has the value."""
        tool = DummyTool()
        exact_cache = DummyExactCache({"already": "cached"})
        wrapper = CacheToolWrapper(tool, exact_cache=exact_cache, semantic_cache=None)
        asyncio.run(wrapper.execute("req"))
        assert exact_cache.set_calls == []  # hit -- no live call, no write-back

    def test_no_writeback_on_semantic_hit(self):
        """No write-back occurs when the semantic cache serves the result."""
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        semantic_cache = DummySemanticCache([({"sem": "payload"}, 0.05)])
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=DummyEmbedder(),
        )
        asyncio.run(wrapper.execute("req"))
        assert exact_cache.set_calls == []  # no live call -> no write-back
        assert semantic_cache.add_calls == []

    def test_semantic_writeback_add_called(self):
        """semantic_cache.add() is called with (key, embedding, result) after live call."""
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        semantic_cache = DummySemanticCache([])  # empty -> miss -> live call
        embedder = DummyEmbedder()  # returns ["embedding"]
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=embedder,
        )
        asyncio.run(wrapper.execute("req"))
        assert len(semantic_cache.add_calls) == 1
        id_, emb, payload = semantic_cache.add_calls[0]
        assert id_ == "req"
        assert emb == ["embedding"]
        assert payload == {"ok": True}

    def test_no_semantic_writeback_when_embed_returns_none(self):
        """semantic_cache.add() is skipped when the embedder returns None."""
        tool = DummyTool()
        exact_cache = DummyExactCache(None)
        semantic_cache = DummySemanticCache([])
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embedder=DummyEmbedder(result=None),
        )
        asyncio.run(wrapper.execute("req"))
        assert semantic_cache.add_calls == []

    def test_writeback_error_does_not_fail_call(self):
        """A broken exact cache must not prevent the live result from being returned."""

        class BrokenCache:
            def get(self, key):
                return None

            def set(self, key, value):
                raise RuntimeError("disk full")

        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=BrokenCache(), semantic_cache=None)
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}  # main call still succeeds


# ---------------------------------------------------------------------------
# Task 2: nested threshold config (tool)
# ---------------------------------------------------------------------------


class TestNestedThresholdConfig:
    """Threshold is read from cache.semantic.threshold with flat-key fallback."""

    def _wrapper(self, distance, config):
        tool = DummyTool()
        semantic_cache = DummySemanticCache([({"answer": "hit"}, distance)])
        return (
            CacheToolWrapper(
                tool,
                exact_cache=DummyExactCache(None),
                semantic_cache=semantic_cache,
                embedder=DummyEmbedder(),
                config=config,
            ),
            tool,
        )

    def test_nested_threshold_accepts_within_range(self):
        """Nested threshold 0.5 accepts similarity 0.75 (distance 0.25)."""
        wrapper, tool = self._wrapper(
            distance=0.25,  # similarity = 0.75
            config={"cache": {"semantic": {"threshold": 0.5}}},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"answer": "hit"}
        assert tool.called is False

    def test_default_threshold_rejects_same_similarity(self):
        """Default threshold 0.90 rejects similarity 0.75 (distance 0.25)."""
        wrapper, tool = self._wrapper(distance=0.25, config={})
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert tool.called is True

    def test_flat_semantic_threshold_still_accepted(self):
        """Flat semantic_threshold key still works as a fallback."""
        wrapper, tool = self._wrapper(
            distance=0.25,  # similarity = 0.75
            config={"semantic_threshold": 0.5},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"answer": "hit"}
        assert tool.called is False

    def test_nested_key_takes_precedence_over_flat(self):
        """Nested cache.semantic.threshold wins over flat semantic_threshold."""
        # nested=0.80, flat=0.50; similarity 0.75 < nested 0.80 -> rejected
        wrapper, tool = self._wrapper(
            distance=0.25,  # similarity = 0.75
            config={
                "cache": {"semantic": {"threshold": 0.80}},
                "semantic_threshold": 0.50,
            },
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert tool.called is True

    def test_no_config_uses_default(self):
        """No config at all -> default threshold 0.90 applies."""
        tool = DummyTool()
        # distance=0.05 -> similarity=0.95, above 0.90 -> hit
        semantic_cache = DummySemanticCache([({"answer": "hit"}, 0.05)])
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=semantic_cache,
            embedder=DummyEmbedder(),
            config=None,
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"answer": "hit"}
        assert tool.called is False


# ---------------------------------------------------------------------------
# Task 4: tool contract — name and description delegation
# ---------------------------------------------------------------------------


class TestToolContract:
    """CacheToolWrapper must expose name and description from the wrapped tool."""

    def test_name_delegates_to_wrapped_tool(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=None, semantic_cache=None)
        assert wrapper.name == "dummy"

    def test_description_delegates_to_wrapped_tool(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=None, semantic_cache=None)
        assert wrapper.description == "dummy"

    def test_name_tracks_wrapped_tool_dynamically(self):
        """If the underlying tool's name changes, the wrapper reflects it."""

        class CustomTool:
            name = "my-tool"
            description = "does things"

            async def execute(self, input):
                return {}

        wrapper = CacheToolWrapper(CustomTool(), exact_cache=None, semantic_cache=None)
        assert wrapper.name == "my-tool"
        assert wrapper.description == "does things"


# ---------------------------------------------------------------------------
# Task 5: tool_calls counter and metrics key
# ---------------------------------------------------------------------------


class TestToolCallsMetricKey:
    """Wrapper counter is tool_calls; counts snapshot uses key tool_calls."""

    def test_live_call_increments_tool_calls(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=DummyExactCache(None), semantic_cache=None)
        asyncio.run(wrapper.execute("req"))
        assert wrapper.tool_calls == 1

    def test_tool_calls_in_counts_snapshot(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=DummyExactCache(None), semantic_cache=None)
        asyncio.run(wrapper.execute("req"))
        assert "tool_calls" in wrapper._last_metrics["counts"]
        assert wrapper._last_metrics["counts"]["tool_calls"] == 1

    def test_no_tool_calls_on_exact_hit(self):
        tool = DummyTool()
        wrapper = CacheToolWrapper(tool, exact_cache=DummyExactCache({"hit": True}), semantic_cache=None)
        asyncio.run(wrapper.execute("req"))
        assert wrapper.tool_calls == 0
        assert wrapper._last_metrics["counts"]["tool_calls"] == 0


# ---------------------------------------------------------------------------
# Task 6: semantic skip sub-bucket counters (tool)
# ---------------------------------------------------------------------------


class TestSemanticSkipReasons:
    """embed_errors and threshold_rejects track skip sub-reasons; semantic_skips is aggregate."""

    def test_embed_none_increments_embed_errors(self):
        """Embedder returning None -> embed_errors += 1."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.0)]),
            embedder=DummyEmbedder(result=None),
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_errors == 1
        assert wrapper.threshold_rejects == 0
        assert wrapper.semantic_skips == 1

    def test_embed_none_embed_errors_in_counts(self):
        """embed_errors appears in the counts snapshot."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([]),
            embedder=DummyEmbedder(result=None),
        )
        asyncio.run(wrapper.execute("req"))
        counts = wrapper._last_metrics["counts"]
        assert counts["embed_errors"] == 1
        assert counts["threshold_rejects"] == 0
        assert counts["semantic_skips"] == 1

    def test_below_threshold_increments_threshold_rejects(self):
        """Similarity below threshold -> threshold_rejects += 1."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.20)]),  # similarity=0.80 < 0.90
            embedder=DummyEmbedder(),
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.threshold_rejects == 1
        assert wrapper.embed_errors == 0
        assert wrapper.semantic_skips == 1

    def test_threshold_rejects_in_counts(self):
        """threshold_rejects appears in the counts snapshot."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.20)]),
            embedder=DummyEmbedder(),
        )
        asyncio.run(wrapper.execute("req"))
        counts = wrapper._last_metrics["counts"]
        assert counts["threshold_rejects"] == 1
        assert counts["embed_errors"] == 0
        assert counts["semantic_skips"] == 1

    def test_semantic_hit_does_not_increment_either_sub_bucket(self):
        """A successful semantic hit leaves both sub-counters at zero."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.05)]),  # similarity=0.95 >= 0.90
            embedder=DummyEmbedder(),
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_errors == 0
        assert wrapper.threshold_rejects == 0
        assert wrapper.semantic_skips == 0

    def test_no_embedder_does_not_increment_sub_buckets(self):
        """No-embedder skip increments semantic_skips but not either sub-bucket."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.0)]),
            embedder=None,
        )
        asyncio.run(wrapper.execute("req"))
        assert wrapper.semantic_skips == 1
        assert wrapper.embed_errors == 0
        assert wrapper.threshold_rejects == 0

    def test_sub_bucket_counters_accumulate(self):
        """Multiple calls accumulate sub-bucket counters independently."""
        tool = DummyTool()
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=DummySemanticCache([({}, 0.20)]),  # below threshold
            embedder=DummyEmbedder(),
        )
        asyncio.run(wrapper.execute("req1"))
        asyncio.run(wrapper.execute("req2"))
        assert wrapper.threshold_rejects == 2
        assert wrapper.semantic_skips == 2
        assert wrapper.embed_errors == 0


# ---------------------------------------------------------------------------
# Task 7: cache.semantic.enabled flag (tool)
# ---------------------------------------------------------------------------


class TestSemanticEnabledFlag:
    """cache.semantic.enabled=False bypasses the semantic path entirely."""

    def _wrapper_with_semantic_disabled(self, semantic_results=None):
        tool = DummyTool()
        semantic_cache = DummySemanticCache(semantic_results or [({}, 0.0)])
        return (
            CacheToolWrapper(
                tool,
                exact_cache=DummyExactCache(None),
                semantic_cache=semantic_cache,
                embedder=DummyEmbedder(),
                config={"cache": {"semantic": {"enabled": False}}},
            ),
            tool,
            semantic_cache,
        )

    def test_semantic_disabled_falls_through_to_tool(self):
        """With enabled=False, live tool is always called (no semantic lookup)."""
        wrapper, tool, _ = self._wrapper_with_semantic_disabled()
        result = asyncio.run(wrapper.execute("req"))
        assert result == {"ok": True}
        assert tool.called is True

    def test_semantic_disabled_does_not_query_semantic_cache(self):
        """semantic_cache.query() is never called when enabled=False."""
        wrapper, _, semantic_cache = self._wrapper_with_semantic_disabled()
        asyncio.run(wrapper.execute("req"))
        assert semantic_cache.last_embedding is None  # query never called

    def test_semantic_disabled_increments_semantic_skips(self):
        """Skipping due to enabled=False still counts in semantic_skips."""
        wrapper, _, _ = self._wrapper_with_semantic_disabled()
        asyncio.run(wrapper.execute("req"))
        assert wrapper.semantic_skips == 1

    def test_semantic_disabled_does_not_increment_embed_calls(self):
        """No embedding is attempted when semantic is disabled."""
        wrapper, _, _ = self._wrapper_with_semantic_disabled()
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_calls == 0

    def test_semantic_disabled_does_not_increment_sub_buckets(self):
        """Config-disabled skip doesn't count as embed_error or threshold_reject."""
        wrapper, _, _ = self._wrapper_with_semantic_disabled()
        asyncio.run(wrapper.execute("req"))
        assert wrapper.embed_errors == 0
        assert wrapper.threshold_rejects == 0

    def test_semantic_enabled_true_uses_semantic_path(self):
        """Explicit enabled=True behaves identically to the default."""
        tool = DummyTool()
        semantic_cache = DummySemanticCache([({}, 0.05)])  # similarity=0.95 hit
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=semantic_cache,
            embedder=DummyEmbedder(),
            config={"cache": {"semantic": {"enabled": True}}},
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {}
        assert tool.called is False

    def test_semantic_enabled_default_uses_semantic_path(self):
        """No enabled flag in config -> default True -> semantic path active."""
        tool = DummyTool()
        semantic_cache = DummySemanticCache([({}, 0.05)])
        wrapper = CacheToolWrapper(
            tool,
            exact_cache=DummyExactCache(None),
            semantic_cache=semantic_cache,
            embedder=DummyEmbedder(),
            config=None,
        )
        result = asyncio.run(wrapper.execute("req"))
        assert result == {}
        assert tool.called is False
