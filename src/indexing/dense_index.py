# src/indexing/dense_index.py
"""
dense_index.py — wraps ChromaDB as the persistent dense (semantic) vector store.

This module ONLY talks to ChromaDB. It knows nothing about parsing,
chunking, or embedding models — it just stores and queries vectors +
metadata + the original text. Keeping it this isolated means you could
swap ChromaDB for FAISS or another vector store later without touching
embedder.py, chunker.py, or retrieval logic upstream/downstream.
"""

import chromadb
from pathlib import Path

DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_COLLECTION_NAME = "fyp_research_assistant"


class DenseIndex:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION_NAME):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: list, embeddings) -> None:
        """
        Write chunks + their embeddings into the collection.

        chunks: list of objects with .text and .metadata (DocumentChunk),
                OR list of dicts with "text" and "metadata" keys
                (e.g. loaded from the chunks.json checkpoint).
        embeddings: numpy array or list of vectors, same length as chunks,
                    same order — produced by embedder.embed_documents(chunks).

        ChromaDB requires string IDs, string/number/bool metadata values
        (no None, no nested dicts) — metadata is sanitized below to avoid
        silent write failures on disallowed types.
        """
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"] if isinstance(chunk, dict) else chunk.text
            meta = chunk["metadata"] if isinstance(chunk, dict) else chunk.metadata

            chunk_id = meta.get("chunk_id") or f"{meta.get('source_file', 'doc')}_{i:05d}"
            clean_meta = {k: v for k, v in meta.items() if v is not None}

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(clean_meta)

        embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"Upserted {len(ids)} chunks into ChromaDB collection '{self.collection.name}'.")

    def query(self, query_embedding, top_k: int = 5) -> list[dict]:
        """
        Run a dense similarity search. Returns a list of result dicts:
        [{"chunk_id", "text", "metadata", "distance"}, ...]
        ordered by relevance (lowest distance first).
        """
        vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=top_k,
        )

        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            output.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "distance": dist,
            })
        return output

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    # Smoke test: build a tiny fake index and query it, no real PDFs needed.
    import numpy as np

    index = DenseIndex(db_path="data/chroma_db_test", collection_name="smoke_test")

    fake_chunks = [
        {"text": "Unsupervised monocular depth estimation method.",
         "metadata": {"source_file": "paper1.pdf", "page_num": 3, "chunk_id": "p1_0001"}},
        {"text": "MAGSAC is a robust model fitting algorithm.",
         "metadata": {"source_file": "paper2.pdf", "page_num": 7, "chunk_id": "p2_0001"}},
    ]
    fake_embeddings = np.random.rand(2, 768)  # dummy vectors, just to test plumbing

    index.upsert_chunks(fake_chunks, fake_embeddings)
    print(f"Collection count: {index.count()}")

    results = index.query(fake_embeddings[0], top_k=2)
    print("\nQuery results:")
    for r in results:
        print(f"  chunk_id={r['chunk_id']}  distance={r['distance']:.4f}  meta={r['metadata']}")