import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "cache-store-chroma"))

from cache_store_chroma.store import ChromaCache  # noqa: E402


def test_semantic_cache_insert_query(tmp_path):
    cache = ChromaCache(tmp_path / "chroma")
    cache.add("id1", [0.1, 0.2, 0.3], {"value": "ok"})
    result = cache.query([0.1, 0.2, 0.3])
    # query() now returns list of (metadata, distance) tuples
    assert len(result) == 1
    payload, distance = result[0]
    assert payload["value"] == "ok"
    # identical vectors → cosine distance ≈ 0
    assert distance == pytest.approx(0.0, abs=1e-5)


def test_query_returns_distance_for_similar_vector(tmp_path):
    cache = ChromaCache(tmp_path / "chroma")
    cache.add("id1", [1.0, 0.0, 0.0], {"label": "unit-x"})
    # Query with a close but not identical vector
    result = cache.query([0.9, 0.1, 0.0])
    assert len(result) == 1
    payload, distance = result[0]
    assert payload["label"] == "unit-x"
    # Should be a small positive distance (not zero, not huge)
    assert 0.0 <= distance < 0.1


def test_query_returns_multiple_results(tmp_path):
    cache = ChromaCache(tmp_path / "chroma")
    cache.add("a", [1.0, 0.0, 0.0], {"k": "a"})
    cache.add("b", [0.0, 1.0, 0.0], {"k": "b"})
    result = cache.query([1.0, 0.0, 0.0], n_results=2)
    assert len(result) == 2
    # First result should be the identical vector (distance ≈ 0)
    payload0, dist0 = result[0]
    assert payload0["k"] == "a"
    assert dist0 == pytest.approx(0.0, abs=1e-5)
