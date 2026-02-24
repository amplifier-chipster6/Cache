try:
    from amplifier_core.interfaces import Provider
except ModuleNotFoundError:
    from typing import Protocol

    class Provider(Protocol):
        async def complete(self, request, **kwargs): ...


class CacheProviderWrapper(Provider):
    def __init__(
        self,
        provider,
        exact_cache,
        semantic_cache,
        normalizer=None,
        embedder=None,
        config=None,
    ):
        self.provider = provider
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.normalizer = normalizer
        self.embedder = embedder
        self.config = config

    async def complete(self, request, **kwargs):
        key = self.normalizer(request) if self.normalizer else request
        if self.exact_cache:
            cached = self.exact_cache.get(key)
            if cached is not None:
                return cached
        return await self.provider.complete(request, **kwargs)
