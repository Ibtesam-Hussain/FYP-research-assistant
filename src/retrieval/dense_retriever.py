# src/retrieval/dense_retriever.py
"""
dense_retriever.py — thin wrapper giving DenseIndex a uniform interface
for the fusion step. It only does ONE job: text query -> embed -> search.
"""

from src.indexing.dense_index import DenseIndex
from src.ingestion.embedder import Embedder


class DenseRetriever:
    def __init__(self, embedder: Embedder = None, index: DenseIndex = None):
        self.embedder = embedder or Embedder()
        self.index = index or DenseIndex()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Returns results ordered best-first:
        [{"chunk_id", "text", "metadata", "distance"}, ...]
        """
        query_vec = self.embedder.embed_query(query)
        return self.index.query(query_vec, top_k=top_k)


if __name__ == "__main__":
    retriever = DenseRetriever()
    results = retriever.retrieve("How does monocular depth estimation work?", top_k=3)
    for r in results:
        print(f"  {r['chunk_id']}  distance={r['distance']:.4f}")