from cache_store_chroma.store import ChromaCache


def test_semantic_cache_insert_query(tmp_path):
    cache = ChromaCache(tmp_path / "chroma")
    cache.add("id1", [0.1, 0.2, 0.3], {"value": "ok"})
    result = cache.query([0.1, 0.2, 0.3])
    assert result[0]["value"] == "ok"
