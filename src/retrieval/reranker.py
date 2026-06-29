# src/retrieval/reranker.py
"""
reranker.py — cross-encoder reranking of the RRF-fused candidates.

RRF gives you a reasonable merged ordering cheaply (no extra model calls),
but it only knows about RANK POSITION, not actual semantic relevance
between the query and each chunk's full text. A cross-encoder reads the
query and chunk text TOGETHER (not as separate vectors, the way the
dense embedder does) and outputs a direct relevance score for that exact
pair -- generally more accurate than rank-fusion heuristics, at the cost
of being slower (it scores each candidate individually, not via a fast
vector index lookup).

This is why the pipeline does RRF first to cut candidates down (e.g. from
hundreds of chunks to ~15-20), THEN reranks only that smaller shortlist --
reranking the entire corpus would be far too slow.
"""

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        print(f"Loading reranker model: {model_name} (first run downloads weights)")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        candidates: fused results from reciprocal_rank_fusion(), each a
        dict with "text" (and whatever else fusion.py attached).

        Returns the top_k candidates re-ordered by cross-encoder relevance
        score (higher = more relevant), with a new "rerank_score" field.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        output = []
        for candidate, score in scored[:top_k]:
            merged = dict(candidate)
            merged["rerank_score"] = float(score)
            output.append(merged)
        return output


if __name__ == "__main__":
    reranker = Reranker()

    fake_candidates = [
        {"chunk_id": "A", "text": "The cat sat on the mat."},
        {"chunk_id": "B", "text": "Monocular depth estimation predicts depth from a single RGB image using deep learning."},
        {"chunk_id": "C", "text": "MAGSAC is a robust geometric model fitting algorithm."},
    ]

    results = reranker.rerank("How does monocular depth estimation work?", fake_candidates, top_k=2)
    print("Reranked results:")
    for r in results:
        print(f"  {r['chunk_id']}  rerank_score={r['rerank_score']:.4f}  text={r['text'][:60]}")