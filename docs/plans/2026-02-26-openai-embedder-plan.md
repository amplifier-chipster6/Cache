# OpenAI Embedder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an OpenAI `text-embedding-3-small` embedder module and wire semantic cache hits (tool + provider wrappers) using the existing OpenAI API key.

**Architecture:** Embedder is a tool module returning a callable `embed(key) -> list[float] | None`. Wrappers normalize keys, call embedder, query Chroma, and enforce the similarity threshold before returning semantic hits. Failures skip semantic path.

**Tech Stack:** Python, OpenAI embeddings, Chroma, pytest.

---

### Task 1: Module scaffolding
**Files:**
- Create: `modules/embedder-openai/pyproject.toml`
- Create: `modules/embedder-openai/embedder_openai/__init__.py`
- Create: `modules/embedder-openai/embedder_openai/mount.py`
- Create: `modules/embedder-openai/embedder_openai/client.py`

**Step 1: Write failing test**  
Add placeholder test to ensure import + mount exists:  
`tests/test_embedder_openai.py::test_mount_returns_callable` expecting ImportError/AttributeError initially.

**Step 2: Run test to see failure**  
Run: `pytest tests/test_embedder_openai.py::test_mount_returns_callable -q`  
Expect: fails (module not found).

**Step 3: Add scaffolding files (empty mount returning stub)**  
Populate pyproject with setuptools, no amplifier-core dependency. Add empty `mount()` returning lambda that raises NotImplemented.

**Step 4: Run test**  
Expect: still failing until implementation in later tasks.

---

### Task 2: OpenAI client with retries
**Files:**  
- Modify: `modules/embedder-openai/embedder_openai/client.py`
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write failing tests**  
- Mock OpenAI client to simulate 200, 429 (retry), 500 (retry), and non-retryable errors.  
- Validate returned vector length equals dimensions; mismatch raises.

**Step 2: Run tests**  
`pytest tests/test_embedder_openai.py::TestClient -q` (expected fail).

**Step 3: Implement client**  
- Read API key from env `OPENAI_API_KEY` only.  
- `embed_one(text, *, model, dimensions, timeout_s, max_retries, backoff_seq)` -> list[float].  
- Retry on 429/5xx with backoff 1/2/4; raise on exhaustion or non-retryable.  
- Validate length == dimensions; raise ValueError if mismatch.

**Step 4: Run tests**  
`pytest tests/test_embedder_openai.py::TestClient -q` (expected pass).

---

### Task 3: Mount returning embed callable
**Files:**  
- Modify: `modules/embedder-openai/embedder_openai/mount.py`
- Modify: `modules/embedder-openai/embedder_openai/__init__.py`
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write failing tests**  
- `mount(config)` returns callable `embed(key:str)->list[float]|None`.  
- On client error: returns None (does not raise).  
- Uses defaults: model=text-embedding-3-small, dimensions=1536, batch_size=32, timeout_s=10, retries/backoff as config.

**Step 2: Run tests**  
`pytest tests/test_embedder_openai.py::TestMount -q` (expected fail).

**Step 3: Implement mount**  
- Parse config dict (no api_key).  
- Instantiate client once; embed wraps `client.embed_one`.  
- Catch exceptions, return None on failure.

**Step 4: Run tests**  
`pytest tests/test_embedder_openai.py::TestMount -q` (expected pass).

---

### Task 4: Provider wrapper semantic path
**Files:**  
- Modify: `modules/cache-provider-wrapper/cache_provider_wrapper/provider.py`
- Modify: `modules/cache-provider-wrapper/cache_provider_wrapper/mount.py` (ensure embedder passed through)
- Add tests: `tests/test_cache_provider_wrapper.py`

**Step 1: Write failing tests**  
- Semantic hit: embedder returns vec, semantic cache returns (payload, score>=threshold) -> return payload.  
- Threshold rejection: score < threshold -> ignore semantic hit, fall through to provider call.  
- Embed failure (None) -> skip semantic path.  
- Existing exact hit path remains.

**Step 2: Run tests**  
`pytest tests/test_cache_provider_wrapper.py -q` (expected fail).

**Step 3: Implement provider semantic flow**  
- Normalize key, exact lookup, else embed, semantic query, threshold filter, else provider call & store.

**Step 4: Run tests**  
`pytest tests/test_cache_provider_wrapper.py -q` (expected pass).

---

### Task 5: Tool wrapper threshold enforcement
**Files:**  
- Modify: `modules/cache-tool-wrapper/cache_tool_wrapper/tool.py`
- Modify tests: `tests/test_cache_tool_wrapper.py`

**Step 1: Write failing tests**  
- Reject semantic result when score < threshold.  
- Accept when >= threshold.  
- Embed None skips semantic.

**Step 2: Run tests**  
`pytest tests/test_cache_tool_wrapper.py::TestSemantic -q` (expected fail).

**Step 3: Implement threshold filter in tool wrapper**  
- Use existing `self.config.semantic_threshold` (or default 0.90). Filter before returning semantic hit.

**Step 4: Run tests**  
`pytest tests/test_cache_tool_wrapper.py::TestSemantic -q` (expected pass).

---

### Task 6: Behavior wiring
**Files:**  
- Modify: `behaviors/hybrid-cache.yaml`

**Step 1: Add module include**  
`cache:modules/embedder-openai`

**Step 2: Add embedding config block**  
```
cache:
  semantic:
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
      batch_size: 32
      timeout_s: 10
      max_retries: 3
      retry_backoff_s: [1, 2, 4]
```

**Step 3: No api_key field** (uses env).

**Step 4: No tests; verify via existing tests after wiring.**

---

### Task 7: Awareness doc update
**Files:**  
- Modify: `context/hybrid-cache-awareness.md`

**Step 1: Add brief note**  
Mention `embedder-openai` generates vectors; wrappers enforce threshold; key uses shared OpenAI API key via env.

**Step 2: No tests.**

---

### Task 8: Full test sweep
**Files/Commands:**  
- Run: `pytest tests -q`
Expected: PASS.

---

### Execution options
1) Subagent-driven here (recommended): use superpowers:subagent-driven-development to run tasks in this session with checkpoints.  
2) Separate session: run superpowers:executing-plans in a new session/worktree.