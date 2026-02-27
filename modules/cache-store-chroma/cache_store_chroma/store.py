import json

import chromadb

# Primitive types that Chroma metadata supports natively as column values.
_FLAT_TYPES = (str, int, float, bool, type(None))


def _is_flat_dict(d: object) -> bool:
    """Return True when *d* is a dict whose values are all Chroma-safe primitives."""
    return isinstance(d, dict) and all(isinstance(v, _FLAT_TYPES) for v in d.values())


class ChromaCache:
    """Semantic cache backed by ChromaDB.

    Task 10 changes:
    - ``add``: if *payload* is not already a flat dict (all primitive values),
      it is JSON-serialised and stored as ``{"payload": "<json>"}`` so Chroma
      never rejects nested structures.
    - ``query``: guards against querying an empty collection (some ChromaDB
      versions raise on ``n_results > count``). Returns ``[]`` immediately when
      the collection is empty or when the result set is empty.
    """

    def __init__(self, path):
        self._client = chromadb.PersistentClient(path=str(path))
        # Use cosine distance so scores map naturally to cosine similarity.
        # distance = 1 - cosine_similarity ∈ [0, 2]; 0 = identical vectors.
        self._collection = self._client.get_or_create_collection(
            "cache", metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, id: str, embedding: list[float], payload: object) -> None:
        """Store *embedding* + *payload* under *id*.

        If *payload* is not a flat dict (all leaf values are str/int/float/bool/None)
        the whole value is JSON-serialised and stored as
        ``{"payload": "<json-string>"}``.  Callers that later retrieve it with
        ``query`` will see this wrapper dict unchanged — they should check for
        the ``"payload"`` key and ``json.loads`` it when needed.
        """
        metadata: dict
        if _is_flat_dict(payload):
            metadata = payload  # type: ignore[assignment]
        else:
            metadata = {"payload": json.dumps(payload)}

        self._collection.add(ids=[id], embeddings=[embedding], metadatas=[metadata])

    def query(
        self, embedding: list[float], n_results: int = 1
    ) -> list[tuple[dict, float]]:
        """Return a list of ``(metadata, distance)`` pairs, closest first.

        ``distance`` is the cosine distance (1 - cosine_similarity).
        A distance of 0 means identical vectors; lower is more similar.
        Callers should accept a hit when ``1 - distance >= threshold``.

        Returns ``[]`` when the collection is empty or yields no results,
        guarding against version-specific ChromaDB errors on empty collections.
        """
        # Task 10: guard — avoid querying when there are no items.
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        raw_metadatas = result.get("metadatas") or []
        raw_distances = result.get("distances") or []
        metadatas: list[dict] = [
            dict(m) for m in (raw_metadatas[0] if raw_metadatas else [])
        ]
        distances: list[float] = list(raw_distances[0] if raw_distances else [])
        return list(zip(metadatas, distances))
