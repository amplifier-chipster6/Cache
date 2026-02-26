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

## Metrics emission

Both wrappers maintain cumulative counters and expose a `_last_metrics` snapshot after each call.
Counters and `_last_metrics` are **always updated** regardless of the emission flag.

### Config flag

`cache.metrics.enabled` (boolean, default `true`) controls whether an INFO log line is emitted
per call.  If the key is absent from the config the flag defaults to `true` for backward
compatibility.

Set it in `behaviors/hybrid-cache.yaml`:

```yaml
config:
  cache:
    metrics:
      enabled: false   # disable log emission; counters still update
```

Or pass `{"metrics": {"enabled": False}}` when constructing a wrapper directly.

### Per-call metrics payload shape

`_last_metrics` is a `dict` with the following structure.  No prompt or response content
is ever included.

```python
{
    # Per-call boolean flags
    "exact_hit":     bool,   # True if the exact cache served this call
    "semantic_hit":  bool,   # True if the semantic cache served this call
    "semantic_skip": bool,   # True if the semantic path was bypassed
    "provider_call": bool,   # True if the live provider/tool was called
    "embed_call":    bool,   # True if the embedder was invoked this call

    # Cumulative counter snapshot (integers, all ≥ 0)
    "counts": {
        "exact_hits":       int,
        "exact_misses":     int,
        "semantic_hits":    int,
        "semantic_misses":  int,
        "semantic_skips":   int,
        "provider_calls":   int,
        "embed_calls":      int,
    },

    # Cumulative hit ratio: (exact_hits + semantic_hits) / (exact_hits + semantic_hits + provider_calls)
    # Returns 0.0 when the denominator is zero (zero-guarded).
    "hit_ratio": float,   # range [0.0, 1.0]
}
```

### Log line format

When `cache.metrics.enabled` is `true` the wrapper emits one `logging.INFO` line per call:

```
cache.metrics hit_ratio=0.75 exact_hits=3 semantic_hits=0 provider_calls=1 embed_calls=0
```

The line contains only numeric metrics — **no prompt text, no response content, no cache keys**.

### Safety guarantee

The payload and log line are constructed exclusively from counter integers and derived
float ratios.  Prompt and response content is never captured, stored, or logged by the
metrics subsystem.

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
