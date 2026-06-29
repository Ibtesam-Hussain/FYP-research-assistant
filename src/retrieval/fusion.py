# src/retrieval/fusion.py
"""
fusion.py — Reciprocal Rank Fusion (RRF).

This is the actual "hybrid" mechanism: it merges two independently-ranked
result lists (dense + sparse) into one ranked list.

Why RRF specifically, and not just averaging raw scores: dense search
returns DISTANCE (lower = better, roughly 0-2 range for normalized
vectors) and sparse search returns a BM25 SCORE (higher = better,
unbounded, can be 0-30+ depending on term frequency). These two numbers
live on completely different, incompatible scales -- you cannot average
them directly without one dominating arbitrarily.

RRF sidesteps this entirely by ignoring raw scores and using ONLY each
chunk's RANK POSITION in its respective list. A chunk's RRF contribution
is 1 / (k + rank), summed across both lists. Chunks that appear highly
ranked in BOTH lists -- i.e. both retrieval methods agree it's relevant --
naturally float to the top of the fused list. This is the standard,
well-established formula from the original RRF paper (Cormack et al.).

k=60 is the conventional default used in most RRF implementations/papers;
it softens the impact of rank 1 vs rank 2 being a huge gap, smoothing
the curve. You generally don't need to tune this.
"""

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    dense_results, sparse_results: lists of dicts with "chunk_id", each
    ALREADY ordered best-first (rank 0 = most relevant) by their
    respective retriever. This function only looks at list POSITION,
    never at "distance" or "score" values directly.

    Returns a single fused list, ordered best-first, with an added
    "rrf_score" field. Deduplicates chunks that appear in both lists,
    summing their RRF contributions (this is what rewards consensus).
    """
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[cid] = result  # last write wins for text/metadata, fine since identical chunk

    for rank, result in enumerate(sparse_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[cid] = result

    fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    output = []
    for cid, score in fused:
        merged = dict(chunk_lookup[cid])
        merged["rrf_score"] = score
        output.append(merged)
    return output


if __name__ == "__main__":
    # Same sanity check as before: a chunk present in BOTH lists should
    # outrank chunks present in only one, even if it wasn't rank-1 in either.
    dense = [
        {"chunk_id": "A", "text": "dense top result"},
        {"chunk_id": "B", "text": "dense second result"},
    ]
    sparse = [
        {"chunk_id": "B", "text": "sparse top result"},
        {"chunk_id": "C", "text": "sparse second result"},
    ]

    fused = reciprocal_rank_fusion(dense, sparse)
    print("Fused ranking (B should be #1, since both methods agree on it):")
    for r in fused:
        print(f"  {r['chunk_id']}  rrf_score={r['rrf_score']:.5f}")