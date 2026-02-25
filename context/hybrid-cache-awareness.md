# Hybrid Cache Behavior

Root bundle: `cache`. Behavior: `hybrid-cache`.

Defaults: storage path `./cache`, semantic threshold `0.90`.

## Enable + Configure
Enable the bundle with `amplifier run --bundle cache`.

Settings:
- `cache.enabled`: master toggle for caching
- `exact.ttl`: exact-cache time-to-live (seconds)
- `exact.max_entries`: maximum exact-cache entries
- `semantic.enabled`: toggle semantic cache
- `semantic.threshold`: similarity threshold for semantic hits
- `storage.path`: local cache directory path

This bundle enables hybrid caching (exact + semantic) for providers and tools using local storage.
