# Hybrid Local Cache for Amplifier (Exact + Semantic)

## Summary
Design a hybrid, single-machine caching system for Amplifier that reduces token costs by combining exact-match and semantic caching. This is a policy layer implemented as **modules** (provider/tool wrappers), not kernel changes. Storage is local and simple: SQLite for exact-match entries and Chroma (local mode) for semantic search.

## Goals
- Reduce token costs by reusing prior responses when prompts match exactly or are semantically similar.
- Remain compatible with Amplifier philosophy (policy in modules, kernel unchanged).
- Keep deployment simple on a single machine.
- Provide safe fallbacks if cache components fail.

## Non-Goals
- Multi-machine cache sharing.
- Kernel modifications.
- Global deduplication across unrelated projects.

## Architecture
**Modules only:**
- **Provider wrapper module**: Intercepts LLM calls, checks cache, returns cached responses, or calls the real provider.
- **Tool wrapper module**: Intercepts expensive tool calls and applies the same cache strategy.

**Storage:**
- **SQLite** for exact-match key/value entries (hash → response + metadata)
- **Chroma (local mode)** for semantic vector index (embedding → entry ID)

## Provider Cache Flow
1. Normalize request (system prompt + user messages + tool schema + model + params).
2. Compute exact-match hash and check SQLite.
3. On miss, embed prompt and query Chroma for semantic similarity.
4. If similarity >= threshold (e.g., 0.90), return cached response.
5. Otherwise call the provider and store response + embedding.

## Tool Cache Flow
1. Normalize tool name + args.
2. Exact-match hash lookup.
3. Optional semantic lookup for tools where similarity is safe.
4. Return cached tool result or execute and store.

## Keying & Normalization Rules
- Stable JSON ordering for inputs.
- Whitespace normalization for prompts.
- Ignore timestamps and volatile fields.
- Any change in system prompt or tool schema → treat as cache miss.

## Storage & Eviction
- **SQLite** for exact-match cache (hash → response + metadata).
- **Chroma** for vector index (embedding → entry ID).
- Eviction: TTL + max entries or disk cap.

## Safety & Validity
- Semantic cache only used when similarity >= threshold.
- If embedding or index fails → fall back to exact-match only.
- Cache failures never block provider/tool execution.

## Observability
Emit events:
- cache_hit
- cache_miss
- cache_error

Include metadata: model, temperature, timestamps, cache strategy used.

## Configuration Surface
- cache.enabled
- exact.ttl
- exact.max_entries
- semantic.enabled
- semantic.threshold
- storage.path (default: ./cache/)

## Testing Strategy
- Unit tests for normalization and hashing.
- Unit tests for semantic threshold behavior.
- Integration tests for provider/tool wrapper with cache and fallback.

## Open Questions
- Which embedding model to use for semantic cache?
- Default TTL and size limits for local storage?
- Which tools are safe for semantic cache reuse?
