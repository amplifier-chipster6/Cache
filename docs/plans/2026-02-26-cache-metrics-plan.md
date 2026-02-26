---
bundle:
  name: cache-metrics-plan
  version: 0.1.0
---
# Cache Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add cache hit/miss metrics to the cache wrappers so each request reports hit status and cumulative hit ratio, without leaking prompt/response content.

**Architecture:** Instrument the cache provider/tool wrappers to maintain counters (exact/semantic hits, misses, skips, provider fallbacks, embed calls) and emit a structured metrics event/log per call (no prompt content), including cumulative hit ratio. Optional config flag to enable/disable emission.

**Tech Stack:** Python, existing cache wrappers/modules, pytest, hooks/logging in Amplifier bundle.

---

### Task 1: Add metrics counters to CacheProviderWrapper

**Files:**
- Modify: `modules/cache-provider-wrapper/cache_provider_wrapper/provider.py`
- Test: `tests/test_cache_provider_wrapper.py`

**Step 1: Write failing tests (counters and ratio)**
- Add tests to assert:
  - exact cache hit increments `exact_hits` and reports `hit_ratio`
  - semantic hit increments `semantic_hits`, `embed_calls`
  - embedder returns None -> `semantic_skips` increments, provider called
  - threshold reject -> `semantic_skips` increments, provider called
  - semantic miss -> provider called, counters reflect miss
  - hit_ratio guards divide-by-zero (0 when no provider calls and no hits)

**Step 2: Run tests to see them fail**
- `cd /work/projects/cache`
- `pytest tests/test_cache_provider_wrapper.py -q`
- Expect new tests to fail (missing counters/fields).

**Step 3: Implement counters and per-call metrics emission**
- Add counter fields in `__init__`.
- In decision branches, increment:
  - exact_hits / exact_misses
  - semantic_hits / semantic_misses / semantic_skips
  - embed_calls
  - provider_calls
- Compute `hit_ratio` with zero guard.
- Build a small metrics dict per call (no prompt text), include cumulative snapshot.
- Provide a hook/log emission point (e.g., optional callback or logger) gated by config flag; if unavailable, prepare to no-op.

**Step 4: Run tests to verify pass**
- `pytest tests/test_cache_provider_wrapper.py -q`
- Expect pass.

**Step 5: Commit**
- `git add modules/cache-provider-wrapper/cache_provider_wrapper/provider.py tests/test_cache_provider_wrapper.py`
- `git commit -m "feat: add metrics counters to cache provider wrapper"`

---

### Task 2: Add metrics counters to CacheToolWrapper

**Files:**
- Modify: `modules/cache-tool-wrapper/cache_tool_wrapper/tool.py`
- Test: `tests/test_cache_tool_wrapper.py`

**Step 1: Write failing tests**
- Mirror provider tests:
  - exact hit increments exact_hits
  - semantic hit increments semantic_hits and embed_calls
  - embedder None / threshold reject -> semantic_skips, provider_calls
  - semantic miss -> provider_calls
  - hit_ratio computed with zero guard
- If using shared metrics payload shape, assert fields presence and values.

**Step 2: Run tests to see them fail**
- `pytest tests/test_cache_tool_wrapper.py -q`

**Step 3: Implement counters and emission**
- Add counters in `__init__` and increment in execute() branches.
- Compute `hit_ratio`.
- Emit per-call metrics dict (same shape as provider) via hook/log if enabled.

**Step 4: Run tests to verify pass**
- `pytest tests/test_cache_tool_wrapper.py -q`

**Step 5: Commit**
- `git add modules/cache-tool-wrapper/cache_tool_wrapper/tool.py tests/test_cache_tool_wrapper.py`
- `git commit -m "feat: add metrics counters to cache tool wrapper"`

---

### Task 3: Optional hook/log integration and config flag

**Files:**
- Modify: `modules/cache-provider-wrapper/cache_provider_wrapper/provider.py`
- Modify: `modules/cache-tool-wrapper/cache_tool_wrapper/tool.py`
- Modify: `behaviors/hybrid-cache.yaml` (if adding a `metrics.enabled` flag)
- Modify: `context/hybrid-cache-awareness.md` (document flag and payload shape)
- Tests: adjust provider/tool tests if flag affects behavior

**Step 1: Add config toggle**
- Add `cache.metrics.enabled` (default true) in behavior config if desired.
- In wrappers, only emit metrics when enabled; counters still update.

**Step 2: Implement emission path**
- If hooks available: emit event name `cache.metrics` with the metrics dict.
- Else: log INFO one-liner summarizing flags and hit_ratio (no prompt text).
- Ensure payload contains only counts/booleans/lengths; no content, no keys.

**Step 3: Update docs/context**
- Document the flag, event name, fields, and safety (no prompt content) in `context/hybrid-cache-awareness.md`.

**Step 4: Tests**
- If flag is false, ensure no emission path is triggered (can stub a sink/counter).
- Keep counters updating regardless (or clarify behavior in tests).

**Step 5: Run focused tests**
- `pytest tests/test_cache_provider_wrapper.py tests/test_cache_tool_wrapper.py -q`

**Step 6: Commit**
- `git add behaviors/hybrid-cache.yaml context/hybrid-cache-awareness.md modules/cache-provider-wrapper/cache_provider_wrapper/provider.py modules/cache-tool-wrapper/cache_tool_wrapper/tool.py tests/test_cache_provider_wrapper.py tests/test_cache_tool_wrapper.py`
- `git commit -m "feat: add cache metrics emission and config flag"`

---

### Task 4: Full test sweep

**Files/Commands:**
- `pytest -q`

**Step 1: Run full suite**
- `cd /work/projects/cache`
- `pytest -q`
- Expect all tests passing.

**Step 2: Commit (if any remaining changes)**
- `git status` (ensure clean or commit pending changes).

---

### Task 5: (Optional) Surface metrics via hook consumer

**If you want UI/log surfacing beyond raw events:**
- Add/extend a hook consumer to render the one-line status per call.
- Keep payload redacted (counts/lengths only).
- Test with a stub hook to confirm rendering.

---

### Handoff

Plan complete. Choose execution mode:
1) Subagent-Driven (this session) — use superpowers:subagent-driven-development, fresh subagent per task, with reviews.
2) Parallel Session — new session with superpowers:executing-plans to run tasks sequentially with checkpoints.