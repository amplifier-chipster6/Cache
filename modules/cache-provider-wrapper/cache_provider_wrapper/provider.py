import asyncio
import logging

try:
    from amplifier_core.interfaces import Provider  # type: ignore[import-untyped]
except ModuleNotFoundError:
    from typing import Protocol

    class Provider(Protocol):  # type: ignore[no-redef]
        async def complete(self, request, **kwargs): ...


_DEFAULT_THRESHOLD = 0.90
_logger = logging.getLogger(__name__)


class _BaseCacheWrapper:
    """Shared 3-stage cache flow (exact → semantic → live) used by both wrappers.

    Subclasses must:
    - Call super().__init__() with the wrapped object and cache config.
    - Set self._logger = logging.getLogger(__name__) in their own __init__
      so log records carry the correct module name.
    - Implement the four abstract hooks below.
    """

    def __init__(
        self,
        wrapped,
        exact_cache,
        semantic_cache,
        normalizer=None,
        embedder=None,
        config=None,
    ):
        self._wrapped = wrapped
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.normalizer = normalizer
        self.embedder = embedder
        self.config = config

        # Metrics emission flag (default enabled; set via config["metrics"]["enabled"])
        self._metrics_enabled: bool = (
            config.get("metrics", {}).get("enabled", True) if config else True
        )

        # Shared cumulative counters
        self.exact_hits: int = 0
        self.exact_misses: int = 0
        self.semantic_hits: int = 0
        self.semantic_misses: int = 0
        self.semantic_skips: int = 0
        # Task 6: sub-bucket counters for skip reasons
        self.embed_errors: int = 0
        self.threshold_rejects: int = 0
        self.embed_calls: int = 0

        # Per-call metrics snapshot (no prompt/response content); inspect in tests
        self._last_metrics: dict | None = None

        # Subclasses must set self._logger = logging.getLogger(__name__) after
        # super().__init__() so that _write_back and _maybe_log_metrics use the
        # correct module-level logger name.  Provide a safe default here so that
        # the base class is usable on its own in unit tests.
        self._logger = _logger

    # ------------------------------------------------------------------ hooks

    def _increment_live_calls(self) -> None:
        """Increment the subclass-specific live-call counter (provider_calls / tool_calls)."""
        raise NotImplementedError

    def _live_call_count(self) -> int:
        """Return current value of the subclass-specific live-call counter."""
        raise NotImplementedError

    def _live_calls_count_key(self) -> str:
        """Return the counts-dict key for live calls, e.g. 'provider_calls'."""
        raise NotImplementedError

    def _live_call_metrics_extra(self, live_call: bool) -> dict:
        """Return subclass-specific top-level keys to merge into the metrics dict.

        Provider:  {"provider_call": live_call}
        Tool:      {"provider_call": live_call, "tool_call": live_call}
        """
        raise NotImplementedError

    # --------------------------------------------------------- shared flow

    async def _run_cache_flow(self, key, live_call_factory):
        """Execute the 3-stage cache flow: exact → semantic → live.

        Args:
            key: Normalised cache key for this request.
            live_call_factory: Zero-arg callable returning a coroutine that
                performs the actual live call.  Using a factory (rather than a
                pre-created coroutine) avoids "coroutine never awaited" warnings
                when the result is satisfied by the exact or semantic cache.
        """
        # Per-call flags for the metrics snapshot
        embed_call = False
        semantic_skip = False
        # Hoisted so the live-fallback write-back can reuse it
        embedding = None

        # 1. Exact cache lookup
        if self.exact_cache:
            cached = self.exact_cache.get(key)
            if cached is not None:
                self.exact_hits += 1
                self._last_metrics = self._build_metrics(
                    exact_hit=True,
                    semantic_hit=False,
                    semantic_skip=False,
                    live_call=False,
                    embed_call=False,
                )
                self._maybe_log_metrics(self._last_metrics)
                return cached
            else:
                self.exact_misses += 1

        # Task 7: gate semantic path on config flag (default enabled)
        _semantic_enabled: bool = (
            self.config.get("cache", {}).get("semantic", {}).get("enabled", True)
            if self.config
            else True
        )

        # 2. Semantic cache lookup (requires embedder + semantic_cache + enabled)
        if self.semantic_cache and _semantic_enabled and self.embedder:
            self.embed_calls += 1
            embed_call = True
            # Task 3: run sync embedder in a thread so the event loop stays free
            embedding = await asyncio.get_running_loop().run_in_executor(
                None, self.embedder, key
            )

            if embedding is not None:
                results = self.semantic_cache.query(embedding)
                if results:
                    payload, distance = results[0]
                    # Task 2: read from nested path first, fall back to flat key
                    threshold = (
                        self.config.get("cache", {})
                        .get("semantic", {})
                        .get(
                            "threshold",
                            self.config.get("semantic_threshold", _DEFAULT_THRESHOLD),
                        )
                        if self.config
                        else _DEFAULT_THRESHOLD
                    )
                    similarity = 1.0 - distance
                    if similarity >= threshold:
                        self.semantic_hits += 1
                        self._last_metrics = self._build_metrics(
                            exact_hit=False,
                            semantic_hit=True,
                            semantic_skip=False,
                            live_call=False,
                            embed_call=True,
                        )
                        self._maybe_log_metrics(self._last_metrics)
                        return payload
                    else:
                        # Task 6: below threshold → count as skip + threshold_reject
                        self.semantic_skips += 1
                        self.threshold_rejects += 1
                        semantic_skip = True
                else:
                    # No semantic results: miss, fall through to live call
                    self.semantic_misses += 1
            else:
                # Task 6: embedder returned None → count as skip + embed_error
                self.semantic_skips += 1
                self.embed_errors += 1
                semantic_skip = True

        elif self.semantic_cache and (not _semantic_enabled or not self.embedder):
            # Semantic cache present but disabled by config or no embedder: skip
            self.semantic_skips += 1
            semantic_skip = True

        # 3. Fallback to live call
        self._increment_live_calls()
        result = await live_call_factory()

        # Task 1: write-back — populate caches for future hits
        self._write_back(key, result, embedding)

        self._last_metrics = self._build_metrics(
            exact_hit=False,
            semantic_hit=False,
            semantic_skip=semantic_skip,
            live_call=True,
            embed_call=embed_call,
        )
        self._maybe_log_metrics(self._last_metrics)
        return result

    def _write_back(self, key, result, embedding) -> None:
        """Persist a live result into the caches after a live call.

        Errors are logged and suppressed so write-back never fails the caller.
        """
        if self.exact_cache:
            try:
                self.exact_cache.set(key, result)
            except Exception:
                self._logger.warning(
                    "cache write-back to exact_cache failed", exc_info=True
                )
        if self.semantic_cache and self.embedder and embedding is not None:
            try:
                self.semantic_cache.add(str(key), embedding, result)
            except Exception:
                self._logger.warning(
                    "cache write-back to semantic_cache failed", exc_info=True
                )

    def _maybe_log_metrics(self, metrics: dict) -> None:
        """Emit a concise INFO log line summarising this call's metrics.

        Only fires when metrics emission is enabled (config["metrics"]["enabled"]).
        The log line contains only counts and ratios — never prompt or response text.
        """
        if not self._metrics_enabled:
            return
        c = metrics["counts"]
        lck = self._live_calls_count_key()
        self._logger.info(
            "cache.metrics hit_ratio=%.2f exact_hits=%d semantic_hits=%d"
            " %s=%d embed_calls=%d embed_errors=%d threshold_rejects=%d",
            metrics["hit_ratio"],
            c["exact_hits"],
            c["semantic_hits"],
            lck,
            c[lck],
            c["embed_calls"],
            c["embed_errors"],
            c["threshold_rejects"],
        )

    def _build_metrics(
        self,
        *,
        exact_hit: bool,
        semantic_hit: bool,
        semantic_skip: bool,
        live_call: bool,
        embed_call: bool,
    ) -> dict:
        """Build a per-call metrics snapshot (no prompt/response content)."""
        denom = self.exact_hits + self.semantic_hits + self._live_call_count()
        hit_ratio = (
            0.0 if denom == 0 else (self.exact_hits + self.semantic_hits) / denom
        )
        lck = self._live_calls_count_key()
        base = {
            "exact_hit": exact_hit,
            "semantic_hit": semantic_hit,
            "semantic_skip": semantic_skip,
            "embed_call": embed_call,
            "counts": {
                "exact_hits": self.exact_hits,
                "exact_misses": self.exact_misses,
                "semantic_hits": self.semantic_hits,
                "semantic_misses": self.semantic_misses,
                "semantic_skips": self.semantic_skips,
                "embed_errors": self.embed_errors,
                "threshold_rejects": self.threshold_rejects,
                lck: self._live_call_count(),
                "embed_calls": self.embed_calls,
            },
            "hit_ratio": hit_ratio,
        }
        base.update(self._live_call_metrics_extra(live_call))
        return base


class CacheProviderWrapper(_BaseCacheWrapper, Provider):  # type: ignore[misc]
    def __init__(
        self,
        provider,
        exact_cache,
        semantic_cache,
        normalizer=None,
        embedder=None,
        config=None,
    ):
        super().__init__(
            provider, exact_cache, semantic_cache, normalizer, embedder, config
        )
        self.provider = provider
        self.provider_calls: int = 0
        # Use this module's logger so write-back and metrics records are attributed
        # to cache_provider_wrapper.provider, not the base class module.
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ hooks

    def _increment_live_calls(self) -> None:
        self.provider_calls += 1

    def _live_call_count(self) -> int:
        return self.provider_calls

    def _live_calls_count_key(self) -> str:
        return "provider_calls"

    def _live_call_metrics_extra(self, live_call: bool) -> dict:
        return {"provider_call": live_call}

    # ---------------------------------------------------------- public API

    async def complete(self, request, **kwargs):
        key = self.normalizer(request) if self.normalizer else request
        return await self._run_cache_flow(
            key, lambda: self.provider.complete(request, **kwargs)
        )
