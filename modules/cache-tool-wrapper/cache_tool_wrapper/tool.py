import logging

try:
    from amplifier_core.interfaces import Tool  # type: ignore[import-untyped]
except ModuleNotFoundError:
    from typing import Protocol

    class Tool(Protocol):  # type: ignore[no-redef]
        name: str
        description: str

        async def execute(self, input): ...


_DEFAULT_THRESHOLD = 0.90
_logger = logging.getLogger(__name__)


class CacheToolWrapper(Tool):  # type: ignore[misc]
    def __init__(
        self,
        tool,
        exact_cache,
        semantic_cache,
        normalizer=None,
        embedder=None,
        config=None,
    ):
        self.tool = tool
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.normalizer = normalizer
        self.embedder = embedder
        self.config = config

        # Metrics emission flag (default enabled; set via config["metrics"]["enabled"])
        self._metrics_enabled: bool = (
            config.get("metrics", {}).get("enabled", True) if config else True
        )

        # Cumulative counters
        self.exact_hits: int = 0
        self.exact_misses: int = 0
        self.semantic_hits: int = 0
        self.semantic_misses: int = 0
        self.semantic_skips: int = 0
        self.provider_calls: int = 0
        self.embed_calls: int = 0

        # Per-call metrics snapshot (no prompt/response content); inspect in tests
        self._last_metrics: dict | None = None

    async def execute(self, input):
        key = self.normalizer(input) if self.normalizer else input

        # Per-call flags for the metrics snapshot
        embed_call = False
        semantic_skip = False

        # 1. Exact cache lookup
        if self.exact_cache:
            cached = self.exact_cache.get(key)
            if cached is not None:
                self.exact_hits += 1
                self._last_metrics = self._build_metrics(
                    exact_hit=True,
                    semantic_hit=False,
                    semantic_skip=False,
                    provider_call=False,
                    embed_call=False,
                )
                self._maybe_log_metrics(self._last_metrics)
                return cached
            else:
                self.exact_misses += 1

        # 2. Semantic cache lookup with similarity threshold enforcement
        if self.semantic_cache and self.embedder:
            self.embed_calls += 1
            embed_call = True
            embedding = self.embedder(key)

            if embedding is not None:
                results = self.semantic_cache.query(embedding)
                if results:
                    payload, distance = results[0]
                    threshold = (
                        self.config.get("semantic_threshold", _DEFAULT_THRESHOLD)
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
                            provider_call=False,
                            embed_call=True,
                        )
                        self._maybe_log_metrics(self._last_metrics)
                        return payload
                    else:
                        # Below threshold: treat as skip, fall through to tool
                        self.semantic_skips += 1
                        semantic_skip = True
                else:
                    # No semantic results: miss, fall through to tool
                    self.semantic_misses += 1
            else:
                # Embedder returned None: skip semantic path
                self.semantic_skips += 1
                semantic_skip = True

        elif self.semantic_cache and not self.embedder:
            # Semantic cache present but no embedder configured: skip
            self.semantic_skips += 1
            semantic_skip = True

        # 3. Fallback to live tool
        self.provider_calls += 1
        result = await self.tool.execute(input)
        self._last_metrics = self._build_metrics(
            exact_hit=False,
            semantic_hit=False,
            semantic_skip=semantic_skip,
            provider_call=True,
            embed_call=embed_call,
        )
        self._maybe_log_metrics(self._last_metrics)
        return result

    def _maybe_log_metrics(self, metrics: dict) -> None:
        """Emit a concise INFO log line summarising this call's metrics.

        Only fires when metrics emission is enabled (config["metrics"]["enabled"]).
        The log line contains only counts and ratios — never prompt or response text.
        """
        if not self._metrics_enabled:
            return
        c = metrics["counts"]
        _logger.info(
            "cache.metrics hit_ratio=%.2f exact_hits=%d semantic_hits=%d"
            " provider_calls=%d embed_calls=%d",
            metrics["hit_ratio"],
            c["exact_hits"],
            c["semantic_hits"],
            c["provider_calls"],
            c["embed_calls"],
        )

    def _build_metrics(
        self,
        *,
        exact_hit: bool,
        semantic_hit: bool,
        semantic_skip: bool,
        provider_call: bool,
        embed_call: bool,
    ) -> dict:
        """Build a per-call metrics snapshot (no prompt/response content)."""
        denom = self.exact_hits + self.semantic_hits + self.provider_calls
        hit_ratio = (
            0.0 if denom == 0 else (self.exact_hits + self.semantic_hits) / denom
        )
        return {
            "exact_hit": exact_hit,
            "semantic_hit": semantic_hit,
            "semantic_skip": semantic_skip,
            "provider_call": provider_call,
            "embed_call": embed_call,
            "counts": {
                "exact_hits": self.exact_hits,
                "exact_misses": self.exact_misses,
                "semantic_hits": self.semantic_hits,
                "semantic_misses": self.semantic_misses,
                "semantic_skips": self.semantic_skips,
                "provider_calls": self.provider_calls,
                "embed_calls": self.embed_calls,
            },
            "hit_ratio": hit_ratio,
        }
