import logging

try:
    from amplifier_core.interfaces import Tool  # type: ignore[import-untyped]
except ModuleNotFoundError:
    from typing import Protocol

    class Tool(Protocol):  # type: ignore[no-redef]
        name: str
        description: str

        async def execute(self, input): ...


from cache_provider_wrapper.provider import _BaseCacheWrapper  # noqa: E402

_logger = logging.getLogger(__name__)


class CacheToolWrapper(_BaseCacheWrapper, Tool):  # type: ignore[misc]
    def __init__(
        self,
        tool,
        exact_cache,
        semantic_cache,
        normalizer=None,
        embedder=None,
        config=None,
    ):
        super().__init__(
            tool, exact_cache, semantic_cache, normalizer, embedder, config
        )
        self.tool = tool
        # Task 5: renamed from provider_calls to reflect that this wraps a Tool
        self.tool_calls: int = 0
        # Use this module's logger so write-back and metrics records are attributed
        # to cache_tool_wrapper.tool, not the base class module.
        self._logger = logging.getLogger(__name__)

    # Task 4: delegate name/description to wrapped tool (Tool protocol compliance)
    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description

    # ------------------------------------------------------------------ hooks

    def _increment_live_calls(self) -> None:
        self.tool_calls += 1

    def _live_call_count(self) -> int:
        return self.tool_calls

    def _live_calls_count_key(self) -> str:
        return "tool_calls"

    def _live_call_metrics_extra(self, live_call: bool) -> dict:
        # provider_call is kept as a legacy alias for schema compatibility (Task 5)
        return {"provider_call": live_call, "tool_call": live_call}

    # ---------------------------------------------------------- public API

    async def execute(self, input):
        key = self.normalizer(input) if self.normalizer else input
        return await self._run_cache_flow(key, lambda: self.tool.execute(input))
