# src/retrieval/sparse_retriever.py
"""
sparse_retriever.py — thin wrapper giving SparseIndex the same uniform
interface as DenseRetriever, so fusion.py can treat both identically.
"""

from src.indexing.sparse_index import SparseIndex, DEFAULT_INDEX_PATH


class SparseRetriever:
    def __init__(self, index: SparseIndex = None, index_path: str = DEFAULT_INDEX_PATH):
        if index is not None:
            self.index = index
        else:
            self.index = SparseIndex()
            self.index.load(index_path)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Returns results ordered best-first:
        [{"chunk_id", "text", "metadata", "score"}, ...]
        """
        return self.index.query(query, top_k=top_k)


if __name__ == "__main__":
    retriever = SparseRetriever()
    results = retriever.retrieve("How does monocular depth estimation work?", top_k=3)
    for r in results:
        print(f"  {r['chunk_id']}  score={r['score']:.4f}")