try:
    from amplifier_core.interfaces import Provider  # type: ignore[import-untyped]
except ModuleNotFoundError:
    from typing import Protocol

    class Provider(Protocol):  # type: ignore[no-redef]
        async def complete(self, request, **kwargs): ...


_DEFAULT_THRESHOLD = 0.90


class CacheProviderWrapper(Provider):  # type: ignore[misc]
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

        # 1. Exact cache lookup
        if self.exact_cache:
            cached = self.exact_cache.get(key)
            if cached is not None:
                return cached

        # 2. Semantic cache lookup (requires embedder + semantic_cache)
        if self.semantic_cache and self.embedder:
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
                        return payload

        # 3. Fallback to live provider
        return await self.provider.complete(request, **kwargs)
