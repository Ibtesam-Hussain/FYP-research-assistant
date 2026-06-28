# src/indexing/sparse_index.py
"""
sparse_index.py — wraps BM25 as the keyword-based (lexical) search index.

This is the other half of "hybrid" retrieval. Where dense search (Chroma)
catches semantic similarity ("erosion masking" ~ "boundary smoothing"),
BM25 catches exact term matches ("MAGSAC", "SIFT", specific equation
symbols) that an embedding model might blur together with similar-but-
wrong concepts.

BM25 has no persistence built in (unlike ChromaDB) — it's just an
in-memory data structure built from a tokenized corpus. So we handle
saving/loading it ourselves with pickle, otherwise you'd have to rebuild
it from chunks.json every time you start a new session.
"""

import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

DEFAULT_INDEX_PATH = "data/bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.

    BM25 is a bag-of-words method — it doesn't need anything fancier than
    this for matching exact terms. A real NLP tokenizer (stemming, stopword
    removal) could improve results slightly, but adds complexity that isn't
    worth it for a corpus this size. Worth mentioning as a "future
    improvement" in your README rather than over-engineering now.
    """
    return text.lower().split()


class SparseIndex:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def build(self, chunks: list) -> None:
        """
        Build the BM25 index from a list of chunks.

        chunks: list of objects with .text/.metadata (DocumentChunk),
                OR list of dicts with "text"/"metadata" keys
                (e.g. loaded from the chunks.json checkpoint) —
                same flexible input shape as DenseIndex.upsert_chunks,
                so both indexes can be built from the same chunk list.
        """
        self.chunk_ids = []
        self.documents = []
        self.metadatas = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"] if isinstance(chunk, dict) else chunk.text
            meta = chunk["metadata"] if isinstance(chunk, dict) else chunk.metadata

            chunk_id = meta.get("chunk_id") or f"{meta.get('source_file', 'doc')}_{i:05d}"

            self.chunk_ids.append(chunk_id)
            self.documents.append(text)
            self.metadatas.append(meta)

        tokenized_corpus = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Built BM25 index over {len(self.documents)} chunks.")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Run a BM25 keyword search. Returns results ordered by relevance,
        HIGHEST score first (opposite direction from ChromaDB's distance,
        where LOWEST is best — keep this straight when writing fusion.py).
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built yet. Call .build(chunks) first.")

        tokenized_query = _tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk with its score, sort descending, take top_k.
        scored = list(zip(self.chunk_ids, self.documents, self.metadatas, scores))
        scored.sort(key=lambda x: x[3], reverse=True)

        output = []
        for chunk_id, text, meta, score in scored[:top_k]:
            output.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "score": float(score),
            })
        return output

    def save(self, path: str = DEFAULT_INDEX_PATH) -> None:
        """Persist the index to disk so it doesn't need rebuilding every session."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f)
        print(f"Saved BM25 index -> {path}")

    def load(self, path: str = DEFAULT_INDEX_PATH) -> None:
        """Load a previously saved index instead of rebuilding from scratch."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        print(f"Loaded BM25 index from {path} ({len(self.documents)} chunks).")


if __name__ == "__main__":
    # Smoke test: tiny fake corpus, no real PDFs needed.
    fake_chunks = [
        {"text": "Unsupervised monocular depth estimation method using stereo cues.",
         "metadata": {"source_file": "paper1.pdf", "page_num": 3, "chunk_id": "p1_0001"}},
        {"text": "MAGSAC is a robust model fitting algorithm for geometric estimation.",
         "metadata": {"source_file": "paper2.pdf", "page_num": 7, "chunk_id": "p2_0001"}},
        {"text": "The Laplacian pyramid is used for multi-scale image fusion.",
         "metadata": {"source_file": "paper3.pdf", "page_num": 2, "chunk_id": "p3_0001"}},
    ]

    index = SparseIndex()
    index.build(fake_chunks)

    results = index.query("depth estimation method", top_k=2)
    print("\nQuery results for 'depth estimation method':")
    for r in results:
        print(f"  chunk_id={r['chunk_id']}  score={r['score']:.4f}  text={r['text'][:50]}...")

    index.save("data/bm25_index_test.pkl")

    reloaded = SparseIndex()
    reloaded.load("data/bm25_index_test.pkl")
    print(f"\nReloaded index has {len(reloaded.documents)} chunks — matches original: {len(reloaded.documents) == len(fake_chunks)}")