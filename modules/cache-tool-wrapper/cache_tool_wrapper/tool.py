try:
    from amplifier_core.interfaces import Tool
except ModuleNotFoundError:
    from typing import Protocol

    class Tool(Protocol):
        name: str
        description: str

        async def execute(self, input): ...


class CacheToolWrapper(Tool):
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

    async def execute(self, input):
        key = self.normalizer(input) if self.normalizer else input
        if self.exact_cache:
            cached = self.exact_cache.get(key)
            if cached is not None:
                return cached
        if self.semantic_cache and self.embedder:
            embedding = self.embedder(key)
            results = self.semantic_cache.query(embedding)
            if results:
                hit = results[0]
                if hit:
                    return hit
        return await self.tool.execute(input)
