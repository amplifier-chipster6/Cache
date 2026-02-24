from .tool import CacheToolWrapper


def mount(config):
    if not config or not config.get("tool"):
        return None
    return CacheToolWrapper(
        tool=config["tool"],
        exact_cache=config.get("exact_cache"),
        semantic_cache=config.get("semantic_cache"),
        normalizer=config.get("normalizer"),
        embedder=config.get("embedder"),
        config=config,
    )
