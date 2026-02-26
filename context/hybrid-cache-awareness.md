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
- `semantic.threshold`: similarity threshold for semantic hits (default `0.90`)
- `storage.path`: local cache directory path

## Embedding (embedder-openai)
Semantic cache hits require an embedding vector.  The `embedder-openai` module
generates these vectors using OpenAI's `text-embedding-3-small` model.

- API key is read exclusively from the `OPENAI_API_KEY` environment variable.
  No key material is logged or stored in config.
- Configuration lives under `cache.semantic.embedding` (model, dimensions,
  batch_size, timeout_s, max_retries, retry_backoff_s).
- The embedder returns `None` on any error; the semantic path is then skipped
  and the live provider/tool is called instead.

## Semantic hit decision
Both `CacheProviderWrapper` and `CacheToolWrapper` follow the same flow:

1. Normalize the input key.
2. Check the exact cache (SQLite).
3. If an embedder is wired, embed the key; on `None` skip to step 5.
4. Query Chroma; the store returns `(metadata, distance)` pairs where
   `distance` is cosine distance (`1 − cosine_similarity`).
5. Compute `similarity = 1 − distance`; accept the hit only if
   `similarity ≥ semantic.threshold`.
6. On rejection or no semantic result, call the live provider/tool.

This bundle enables hybrid caching (exact + semantic) for providers and tools using local storage.
