# Design: OpenAI Embedder for Semantic Cache

Date: 2026-02-26  
Owner: Amplifier session b63c9cf3-d8ef-4497-8859-8551c2527bb8  
Status: Approved design (implementation pending)

## Goal
Add an OpenAI-based embedder (`text-embedding-3-small`) to power semantic cache hits for both tool and provider wrappers. Use the Amplifier-wide OpenAI API key (no separate secret).

## Scope
- New module `modules/embedder-openai/` that exposes `mount(config) -> embed(key: str) -> list[float] | None`.
- Behavior wiring in `behaviors/hybrid-cache.yaml` to include the module and configure embedding parameters (no api_key in config).
- Wrapper integration: semantic path in both tool and provider wrappers, with threshold enforcement.
- No local model; OpenAI only. No kernel changes.

## Decisions
- **Provider:** OpenAI `text-embedding-3-small` (dimensions: 1536).
- **API key:** Use Amplifier’s standard OpenAI key from environment (`OPENAI_API_KEY`); do not accept a separate key in config.
- **Threshold location:** Enforce similarity threshold in wrappers (policy), not in the store.
- **Failure behavior:** If embedding fails, return `None`; wrappers skip semantic path and proceed to live call.
- **Retry:** Backoff on 429/5xx with 1/2/4s; give up after 3 attempts.

## Module Shape
```
modules/embedder-openai/
  pyproject.toml          # no amplifier-core dep
  embedder_openai/
    __init__.py
    client.py             # OpenAI client + retries
    mount.py              # mount(config) -> embed callable
```

## Behavior Config (hybrid-cache)
Under `cache.semantic.embedding`:
- provider: openai
- model: text-embedding-3-small
- dimensions: 1536
- batch_size: 32
- timeout_s: 10
- max_retries: 3
- retry_backoff_s: [1, 2, 4]
- (no `api_key`; read from env)

Add module include: `cache:modules/embedder-openai`.

## Integration Points
- **Tool wrapper** (`cache-tool-wrapper`): already calls `embedder(key)`; add threshold filter before returning semantic hit.
- **Provider wrapper** (`cache-provider-wrapper`): add semantic path mirroring tool wrapper (normalize -> exact -> embed -> semantic query -> threshold filter -> fallback).
- **Store** (`cache-store-chroma`): unchanged; receives precomputed vectors.

## Error Handling & Observability
- Validate returned vector length matches `dimensions`; if mismatch, treat as failure.
- On embed failure: log once (optional), return `None`, skip semantic path.
- No additional events/hook requirements.

## Testing
- Module: mock OpenAI client; validate dimension check; retry on retryable errors; hard-fail on non-retryables.
- Tool wrapper: semantic hit, threshold rejection, skip when embed returns None.
- Provider wrapper: same semantic path coverage (new).
- Config: behavior parses defaults; uses env key implicitly.

## Out of Scope
- Local or alternate embedding providers.
- Changes to Chroma store internals.
- Multi-tenant key management.