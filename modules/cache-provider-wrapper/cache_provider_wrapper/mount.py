from .provider import CacheProviderWrapper


def mount(config):
    if not config or not config.get("provider"):
        return None
    return CacheProviderWrapper(
        provider=config["provider"],
        exact_cache=config.get("exact_cache"),
        semantic_cache=config.get("semantic_cache"),
        normalizer=config.get("normalizer"),
        embedder=config.get("embedder"),
        config=config,
    )
