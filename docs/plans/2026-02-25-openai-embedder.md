# OpenAI Embedder Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an OpenAI-based embedder module with tests, configuration, and documentation updates.

**Architecture:** Implement a new `modules/embedder-openai` package exposing an embedder implementation plus mount helper, wire it into the hybrid cache behavior, and document usage and settings. Tests cover the embedder behavior and integration points.

**Tech Stack:** Python, pytest

---

### Task 1: Create module scaffolding and failing tests

**Files:**
- Create: `modules/embedder-openai/pyproject.toml`
- Create: `modules/embedder-openai/embedder.py`
- Create: `modules/embedder-openai/mount.py`
- Create: `modules/embedder-openai/__init__.py`
- Create: `tests/test_embedder_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder_openai.py

def test_openai_embedder_returns_embeddings():
    embedder = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-small")
    vectors = embedder.embed(["hello"])
    assert isinstance(vectors, list)
    assert len(vectors) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder_openai.py::test_openai_embedder_returns_embeddings -v`
Expected: FAIL with "OpenAIEmbedder not defined"

**Step 3: Write minimal implementation**

```python
# modules/embedder-openai/embedder.py

class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]
```

```python
# modules/embedder-openai/__init__.py
from .embedder import OpenAIEmbedder

__all__ = ["OpenAIEmbedder"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder_openai.py::test_openai_embedder_returns_embeddings -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/embedder-openai/embedder.py modules/embedder-openai/__init__.py tests/test_embedder_openai.py
git commit -m "feat: add openai embedder skeleton"
```

### Task 2: Add module packaging and mount helper

**Files:**
- Create: `modules/embedder-openai/pyproject.toml`
- Create: `modules/embedder-openai/mount.py`
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder_openai.py

def test_mount_registers_openai_embedder():
    registry = {}
    mount_openai_embedder(registry)
    assert "openai" in registry
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder_openai.py::test_mount_registers_openai_embedder -v`
Expected: FAIL with "mount_openai_embedder not defined"

**Step 3: Write minimal implementation**

```python
# modules/embedder-openai/mount.py
from .embedder import OpenAIEmbedder


def mount_openai_embedder(registry: dict) -> None:
    registry["openai"] = OpenAIEmbedder
```

```toml
# modules/embedder-openai/pyproject.toml
[project]
name = "embedder-openai"
version = "0.1.0"
requires-python = ">=3.11"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder_openai.py::test_mount_registers_openai_embedder -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/embedder-openai/mount.py modules/embedder-openai/pyproject.toml tests/test_embedder_openai.py
git commit -m "feat: add openai embedder mount and packaging"
```

### Task 3: Wire hybrid-cache behavior

**Files:**
- Modify: `behaviors/hybrid-cache.yaml`
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder_openai.py

def test_hybrid_cache_behavior_mentions_openai_embedder():
    content = open("behaviors/hybrid-cache.yaml", "r", encoding="utf-8").read()
    assert "openai" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder_openai.py::test_hybrid_cache_behavior_mentions_openai_embedder -v`
Expected: FAIL with "assert 'openai' in ..."

**Step 3: Write minimal implementation**

```yaml
# behaviors/hybrid-cache.yaml
# (add openai embedder entry to the embedder registry/config)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder_openai.py::test_hybrid_cache_behavior_mentions_openai_embedder -v`
Expected: PASS

**Step 5: Commit**

```bash
git add behaviors/hybrid-cache.yaml tests/test_embedder_openai.py
git commit -m "feat: register openai embedder in hybrid cache behavior"
```

### Task 4: Update README usage

**Files:**
- Modify: `README.md`
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder_openai.py

def test_readme_mentions_openai_embedder_usage():
    content = open("README.md", "r", encoding="utf-8").read()
    assert "openai" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder_openai.py::test_readme_mentions_openai_embedder_usage -v`
Expected: FAIL with "assert 'openai' in ..."

**Step 3: Write minimal implementation**

```markdown
# README.md
# (add usage example showing embedder: openai and required settings)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder_openai.py::test_readme_mentions_openai_embedder_usage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md tests/test_embedder_openai.py
git commit -m "docs: document openai embedder usage"
```

### Task 5: Update docs with settings snippet

**Files:**
- Modify: `docs/**/settings.yaml` (identify correct doc location)
- Modify: `tests/test_embedder_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder_openai.py

def test_docs_include_openai_settings_snippet():
    content = open("docs/settings.yaml", "r", encoding="utf-8").read()
    assert "openai" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder_openai.py::test_docs_include_openai_settings_snippet -v`
Expected: FAIL with "No such file" or "assert 'openai' in ..."

**Step 3: Write minimal implementation**

```yaml
# docs/settings.yaml
# (add snippet showing openai api_key and model fields)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder_openai.py::test_docs_include_openai_settings_snippet -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/settings.yaml tests/test_embedder_openai.py
git commit -m "docs: add openai embedder settings snippet"
```

---

Plan complete and saved to `docs/plans/2026-02-25-openai-embedder.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
