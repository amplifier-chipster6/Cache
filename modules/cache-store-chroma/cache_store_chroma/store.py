import chromadb


class ChromaCache:
    def __init__(self, path):
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection("cache")

    def add(self, id: str, embedding: list[float], payload: dict):
        self._collection.add(ids=[id], embeddings=[embedding], metadatas=[payload])

    def query(self, embedding: list[float], n_results: int = 1):
        result = self._collection.query(
            query_embeddings=[embedding], n_results=n_results
        )
        return result["metadatas"][0]
