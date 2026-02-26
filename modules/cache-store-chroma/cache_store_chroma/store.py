import chromadb


class ChromaCache:
    def __init__(self, path):
        self._client = chromadb.PersistentClient(path=str(path))
        # Use cosine distance so scores map naturally to cosine similarity.
        # distance = 1 - cosine_similarity  ∈ [0, 2]; 0 = identical vectors.
        self._collection = self._client.get_or_create_collection(
            "cache", metadata={"hnsw:space": "cosine"}
        )

    def add(self, id: str, embedding: list[float], payload: dict):
        self._collection.add(ids=[id], embeddings=[embedding], metadatas=[payload])

    def query(
        self, embedding: list[float], n_results: int = 1
    ) -> list[tuple[dict, float]]:
        """Return a list of ``(metadata, distance)`` pairs, closest first.

        ``distance`` is the cosine distance (1 - cosine_similarity).
        A distance of 0 means identical vectors; lower is more similar.
        Callers should accept a hit when ``1 - distance >= threshold``.
        """
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
